#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
export PYTHONPATH=/home/guoyuyu/guoyuyu/code/code_by_other/neural-motifs_graphcnn:/home/guoyuyu/guoyuyu/lib/python_lib/coco/PythonAPI
export CUDA_VISIBLE_DEVICES=0
export SAVE_MODEL_PATH=/mnt/data1/guoyuyu/datasets/visual_genome/model/neural-motifs_graphcnn/
if [ $1 == "0" ]; then
    echo "visualization!" 
    export CUDA_VISIBLE_DEVICES=0
    python3 -u models/visualization.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 6 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512\
        -lr 1e-4 -ngpu 1 -test\
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/sgdet_obj2_edg4_nore_gcn1_adj0_graph2_rel1_adjsoft_embed_stack_softadjmat_nopassedggcnembed_lr2e2_topgcn_onlyadjzeros/vgrel-18.tar \
        -nepoch 50 -resnet -p 100 -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack
elif [ $1 == "1" ]; then
    echo "visualization!" 
    export CUDA_VISIBLE_DEVICES=1
    python3 -u models/visualization.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -p 100  -lr 6e-3 -ngpu 1 -test \
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_nore_gcn1_adj0_graph2_rel1_adjsoft_embed_stack_softadjmat_nopassedggcnembed_lr2e2/vgrel-5.tar \
        -nepoch 50  -resnet -use_bias \
        -cache sgcls_gcn2_lstm2\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
fi
