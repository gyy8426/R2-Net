#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export PYTHONPATH=/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn:/home/guoyuyu/lib/python_lib/coco/PythonAPI
export CUDA_VISIBLE_DEVICES=0
export SAVE_MODEL_PATH=/mnt/data/guoyuyu/datasets/visual_genome/model/neural-motifs_graphcnn/

echo "TRAINING MOTIFNET"
python3 -u models/eval_rels_coco.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
    -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
    -p 100  -lr 1e-3 -ngpu 1 -test \
    -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_nore_gcn1_adj0_graph2_rel1_adjsoft_embed_stack_softadjmat_nopassedggcnembed_lr2e2/vgrel-5.tar \
    -nepoch 50  -resnet -use_bias \
    -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn 'stack';



