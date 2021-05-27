#!/usr/bin/env bash

# This is a script that will evaluate all the models for SGDET
export PYTHONPATH=/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn:/home/guoyuyu/lib/python_lib/coco/PythonAPI
export CUDA_VISIBLE_DEVICES=0
export SAVE_MODEL_PATH=/mnt/data/guoyuyu/datasets/visual_genome/model/neural-motifs_graphcnn/
if [ $1 == "0" ]; then
    echo "EVALING THE BASELINE"
    python3 models/eval_rels.py -m sgdet -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -ngpu 1 -ckpt checkpoints/baseline-sgdet/vgrel-17.tar \
    -nepoch 50 -use_bias -cache baseline_sgdet.pkl -test
elif [ $1 == "1" ]; then
    echo "EVALING MESSAGE PASSING"
    python3 models/eval_rels.py -m sgdet -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgdet/vgrel-18.tar -cache stanford_sgdet.pkl -test
elif [ $1 == "2" ]; then
    echo "EVALING MOTIFNET"
    python3 models/eval_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgdet/vgrel-14.tar -nepoch 50 -cache motifnet_sgdet.pkl -use_bias
elif [ $1 == "3" ]; then
    echo "EVALING MOTIFNET"
    export CUDA_VISIBLE_DEVICES=0
    python3 -u models/eval_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -p 100  -lr 2e-2 -ngpu 1 -test\
        -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar \
        -nepoch 50 -resnet -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack
        
    python3 -u models/eval_rels_1.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -p 100  -lr 2e-2 -ngpu 1 -test\
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_AEloss/vgrel-8.tar \
        -nepoch 50 -resnet -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack

    
    # python3 -u models/eval_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 6 -clip 5\
        # -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512\
        # -lr 1e-4 -ngpu 1 -test\
        # -ckpt  ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_nore_gcn1_adj0_graph2_rel1_adjsoft_embed_stack_softadjmat_nopassedggcnembed_lr2e2/vgrel-5.tar\
        # -nepoch 50 -resnet -use_bias\
        # -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack
    
fi



