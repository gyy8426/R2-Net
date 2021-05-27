#!/usr/bin/env bash

# This is a script that will evaluate all models for SGCLS
export PYTHONPATH=/home/guoyuyu/guoyuyu/code/code_by_other/neural-motifs_graphcnn:/home/guoyuyu/guoyuyu/lib/python_lib/coco/PythonAPI
export CUDA_VISIBLE_DEVICES=0
export SAVE_MODEL_PATH=/mnt/data1/guoyuyu/datasets/visual_genome/model/neural-motifs_graphcnn/
if [ $1 == "0" ]; then
    echo "EVALING THE BASELINE"
    python models/eval_rels.py -m sgcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar \
    -nepoch 50 -use_bias -test -cache baseline_sgcls
    python models/eval_rels.py -m predcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar \
    -nepoch 50 -use_bias -test -cache baseline_predcls
elif [ $1 == "1" ]; then
    echo "EVALING MESSAGE PASSING"
    python models/eval_rels.py -m sgcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -test -cache stanford_sgcls
    python models/eval_rels.py -m predcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -test -cache stanford_predcls
elif [ $1 == "2" ]; then
    echo "EVALING MOTIFNET"
    python3 models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 2048 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet2_res101_layer4/vgrel-7.tar -nepoch 50 -use_bias -cache motifnet_res101_layer4_sgcls -resnet
    python3 models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 2048 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet2_res101_layer4/vgrel-7.tar -nepoch 50 -use_bias -cache motifnet_res101_layer4_predcls -resnet
elif [ $1 == "3" ]; then
    echo "TRAINING MOTIFNET"
    export CUDA_VISIBLE_DEVICES=0
    
    
    python3 -u models/eval_rels.py -m sgcls -model motifnet -order leftright \
        -nl_obj 12 -nl_edge 12 -nh_obj 12 -nh_edge 12 \
        -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 768 \
        -lr 1e-3 -ngpu 1 -test\
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/nlobj_12_nledge_12_nhobj_12_nlobj_12_hiddim_768_resnet_parall/vgrelbest.tar \
        -nepoch 50 -resnet -p 100 -use_bias \
        -bg_num_graph 0 -bg_num_rel 2 -with_biliner_score
    
    # python3 -u models/eval_rels_1.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        # -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        # -p 100  -lr 2e-2 -ngpu 1 -test\
        # -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_gcnregfeat_AELOSS2lstm/vgrel-10.tar \
        # -nepoch 50 -resnet -use_bias \
        # -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
 
    # python3 -u models/eval_rels_1.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        # -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        # -p 100  -lr 2e-2 -ngpu 1 -test\
        # -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_gcnregfeat_AELOSS2lstm/vgrel-10.tar \
        # -nepoch 50  -resnet -use_bias \
        # -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
        
    # python3 -u models/eval_rels_1.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        # -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512\
        # -p 100  -lr 2e-2 -ngpu 1 -test\
        # -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_gcnregfeat_AELOSS2lstm/vgrel-10.tar \
        # -nepoch 50  -resnet -use_bias \
        # -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
        
    # python3 -u models/eval_rels_1.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 1 -clip 5\
        # -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512\
        # -p 100  -lr 2e-2 -ngpu 1 -test\
        # -ckpt ${SAVE_MODEL_PATH}/checkpoints/obj2_edg4_gcn1_adj0_graph2_rel1_adjsoft_edgeregfeat_gcnregfeat_AELOSS/vgrel-7.tar \
        # -nepoch 50  -resnet -use_bias \
        # -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
        
fi



