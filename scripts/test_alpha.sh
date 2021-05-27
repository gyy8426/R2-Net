#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
export PYTHONPATH=/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn:/home/guoyuyu/lib/python_lib/coco/PythonAPI
export CUDA_VISIBLE_DEVICES=0
export SAVE_MODEL_PATH=/mnt/data/guoyuyu/datasets/visual_genome/model/neural-motifs_graphcnn/
#for i in {0..10}
if [ $1 == "0" ]; then
for i in {0..10}
do
    echo "TRAINING MOTIFNET"
    echo $i
    alpha_i=$(echo "0.1 * $i"|bc)
    echo $alpha_i
	python3 -u models/eval_rels_1.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -p 100  -lr 2e-2 -ngpu 1\
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_AEloss_graphloss_nolossw/vgrel-10.tar \
        -nepoch 50  -resnet -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -test_alpha $alpha_i;
    python3 -u models/eval_rels_1.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -p 100  -lr 2e-2 -ngpu 1\
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_AEloss_graphloss_nolossw/vgrel-10.tar \
        -nepoch 50  -resnet -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -test_alpha $alpha_i;
    python3 -u models/eval_rels_1.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -p 100  -lr 2e-2 -ngpu 1\
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_AEloss_graphloss_nolossw/vgrel-10.tar \
        -nepoch 50  -resnet -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred -test_alpha $alpha_i;
    python3 -u models/eval_rels_1.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512\
        -p 100  -lr 2e-2 -ngpu 1\
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_AEloss_graphloss_nolossw/vgrel-10.tar \
        -nepoch 50  -resnet -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred -test_alpha $alpha_i;
done
elif [ $1 == "1" ]; then
for i in {5}
do
    echo "TRAINING MOTIFNET"
    echo $i
    alpha_i=$(echo "0.1 * $i"|bc)
    echo $alpha_i
    python3 -u models/eval_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 6 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512\
        -lr 1e-4 -ngpu 1 \
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/sgdet_obj2_edg4_nore_gcn1_adj0_graph2_rel1_adjsoft_embed_stack_softadjmat_nopassedggcnembed_lr2e2_lr2e3/vgrel-11.tar \
        -nepoch 50 -resnet -p 100 -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn 'stack' -test_alpha $alpha_i ; 
done
fi
