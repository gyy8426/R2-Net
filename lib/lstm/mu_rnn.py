import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from typing import Optional, Tuple

from lib.fpn.box_utils import nms_overlaps
from lib.word_vectors import obj_edge_vectors
from .highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
import numpy as np

def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.autograd.Variable):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.


    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


class MultiLabelRNN(torch.nn.Module):
    def __init__(self, classes, embed_dim, inputs_dim, hidden_dim, recurrent_dropout_probability=0.2,
                 use_highway=True, use_input_projection_bias=True):
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention purposes
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_size: Number of words in the vocab
        :param bos_token: To use during decoding (non teacher forcing mode))
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        """
        super(MultiLabelRNN, self).__init__()

        self.classes = classes
        embed_vecs = obj_edge_vectors(['start'] + self.classes, wv_dim=100)
        self.obj_embed = nn.Embedding(len(self.classes), embed_dim)
        self.obj_embed.weight.data = embed_vecs
        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.nms_thresh = 0.3

        self.recurrent_dropout_probability=recurrent_dropout_probability
        self.use_highway=use_highway
        # We do the projections for all the gates all at once, so if we are
        # using highway layers, we need some extra projections, which is
        # why the sizes of the Linear layers change here depending on this flag.
        if use_highway:
            self.input_linearity = torch.nn.Linear(self.hidden_size, 6 * self.hidden_size,
                                                   bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size,
                                                   bias=True)
        else:
            self.input_linearity = torch.nn.Linear(self.hidden_size, 4 * self.hidden_size,
                                                   bias=use_input_projection_bias)
            self.state_linearity = torch.nn.Linear(self.hidden_size, 4 * self.hidden_size,
                                                   bias=True)

        self.out = nn.Linear(self.hidden_size, len(self.classes))
        self.onehot_embed = nn.Linear(len(self.classes), self.hidden_size)
        self.input_embed = nn.Linear(self.inputs_dim, self.hidden_size)
        self.reset_parameters()

    @property
    def input_size(self):
        return self.inputs_dim + self.obj_embed.weight.size(1)

    def reset_parameters(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.hidden_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def lstm_equations(self, timestep_input, previous_state, previous_memory, dropout_mask=None):
        """
        Does the hairy LSTM math
        :param timestep_input:
        :param previous_state:
        :param previous_memory:
        :param dropout_mask:
        :return:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        if self.use_highway:
            highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                         projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
            highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
            timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(self,  # pylint: disable=arguments-differ
                inputs,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                labels=None, boxes_for_nms=None, obj_num=None, im_inds=None):
        """
        Parameters
        ----------
        inputs : PackedSequence, required.
            A tensor of shape (batch_size, input_size)
            to apply the LSTM over.

        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (1, batch_size, output_dimension).

        Returns
        -------
        A PackedSequence containing a torch.FloatTensor of shape
        (batch_size, num_timesteps, output_dimension) representing
        the outputs of the LSTM per timestep and a tuple containing
        the LSTM state, with shape (1, batch_size, hidden_size) to
        match the Pytorch API.
        """

        batch_size = inputs.size(0)
        max_obj_num = obj_num.max()
        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = Variable(inputs.data.new()
                                                  .resize_(batch_size, self.hidden_size).fill_(0))
            previous_state = Variable(inputs.data.new()
                                                 .resize_(batch_size, self.hidden_size).fill_(0))
        else:
            assert len(initial_state) == 2
            previous_state = initial_state[0].squeeze(0)
            previous_memory = initial_state[1].squeeze(0)

        #previous_embed = self.obj_embed.weight[0, None].expand(batch_size, 100)
        previous_embed = self.input_embed(inputs)
        if self.recurrent_dropout_probability > 0.0:
            dropout_mask = get_dropout_mask(self.recurrent_dropout_probability, previous_memory)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        out_commitments = []
        out_state = []
        end_ind = 0
        for i in range(int(max_obj_num.data.cpu().numpy()) + 1):


            timestep_input = previous_embed

            previous_state, previous_memory = self.lstm_equations(timestep_input, previous_state,
                                                                  previous_memory, dropout_mask=dropout_mask)

            pred_dist = self.out(previous_state)
            #out_dists.append(pred_dist)
            out_state.append(previous_state)


            out_dist_sample = F.softmax(pred_dist, dim=1)
            # if boxes_for_nms is not None:
            #     out_dist_sample[domains_allowed[i] == 0] = 0.0
            out_dists.append(out_dist_sample)
            # Greedily take the max here amongst non-bgs
            max_value, best_ind = out_dist_sample[:, 1:].max(1)
            output_onehot = out_dist_sample == max_value[:, None]
            output_onehot = output_onehot.type_as(max_value)
            previous_embed = self.onehot_embed(output_onehot)
            #previous_embed = timestep_input
            out_commitments.append(best_ind)
            # if boxes_for_nms is not None and i < boxes_for_nms.size(0):
            #     best_int = int(best_ind.data[0])
            #     domains_allowed[i:, best_int] *= (1 - is_overlap[i, i:, best_int])

        # Do NMS here as a post-processing step
        if boxes_for_nms is not None and not self.training:
            is_overlap = nms_overlaps(boxes_for_nms.data).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh
            # is_overlap[np.arange(boxes_for_nms.size(0)), np.arange(boxes_for_nms.size(0))] = False

            out_dists_sampled = F.softmax(torch.cat(out_dists,0), 1).data.cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = out_commitments[0].data.new(len(out_commitments)).fill_(0)

            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            out_commitments = Variable(out_commitments)
        else:
            out_commitments = torch.cat(out_commitments, 0)
        out_dists = torch.stack(out_dists)
        out_state = torch.stack(out_state)
        zero_dists = torch.zeros_like(out_dists)
        zero_sate = torch.zeros_like(out_state)

        for i, i_num in enumerate(obj_num.data.cpu().numpy()):
            i_num = int(i_num)
            out_dists[i_num:,i,:] = zero_dists[i_num:,i,:]
            out_state[i_num:,i,:] = zero_sate[i_num:,i,:]
        # out_dists shape: num_obj, time_step, num_class
        # out_state shape: num_obj, time_step, dim_hid

        return out_dists.permute(1,0,2), out_commitments, out_state.permute(1,0,2)
