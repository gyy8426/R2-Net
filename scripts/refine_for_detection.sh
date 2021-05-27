#!/usr/bin/env bash

# Refine Motifnet for detection

export PYTHONPATH=/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn:/home/guoyuyu/lib/python_lib/coco/PythonAPI
export CUDA_VISIBLE_DEVICES=1
export SAVE_MODEL_PATH=/mnt/data/guoyuyu/datasets/visual_genome/model/neural-motifs_graphcnn/
if [ $1 == "0" ]; then
     echo "TRAINING THE BASELINE"
    python3 models/train_rels.py -m sgdet -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar  -save_dir checkpoints/baseline-sgdet \
    -nepoch 50 -use_bias
elif [ $1 == "1" ]; then
    echo "TRAINING STANFORD"
    python3 models/train_rels.py -m sgdet -model stanford -b 6 -p 100 -lr 1e-4 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -save_dir checkpoints/stanford-sgdet
elif [ $1 == "2" ]; then
    echo "Refining Motifnet for detection!"
    python3 models/train_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 2048 -lr 1e-4 -ngpu 1 -ckpt checkpoints/motifnet2/vgrel-7.tar \
        -save_dir checkpoints/motifnet-sgdet_res101 -nepoch 10 -use_bias
elif [ $1 == "3" ]; then
    echo "TRAINING MOTIFNET"
    python3 models/train_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 2 -b 6 -clip 5 \
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -p 100  -lr 1e-3 -ngpu 1 -ckpt checkpoints/motifnet2_res101_layer4_24_adj_mat/vgrel-best.tar \
        -save_dir checkpoints/motifnet2-sgdet_res101_layer4_24_adj_mat -nepoch 50 -use_bias -resnet -with_adj_mat -need_bg
elif [ $1 == "4" ]; then
    echo "TRAINING MOTIFNET"
    python3 models/train_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 6 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512\
        -p 100  -lr 1e-4 -ngpu 1 -ckpt  checkpoints/motifnet2_sgdet_res101_layer4_24_adj_mat_graph2_rel0_nl_adj0/vgrel-13.tar\
        -save_dir checkpoints/motifnet2_sgdet_res101_layer4_24_adj_mat_graph2_rel0_nl_adj0 -nepoch 50 -use_bias -resnet -with_adj_mat -bg_num_graph 2 -bg_num_rel 0
elif [ $1 == "5" ]; then
    echo "TRAINING MOTIFNET"
    python3 -u models/train_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 6 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-3 -ngpu 1 \
        -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar \
        -save_dir  ${SAVE_MODEL_PATH}/ablation_study/sgdet_rel_model_edgelstm_regfeat_objedgepos/ \
        -nepoch 50 -resnet -p 100 -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack
elif [ $1 == "6" ]; then
    python3 -u models/train_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 12 -clip 5\
        -pooling_dim 4096 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 6e-6 -ngpu 1 \
        -ckpt ${SAVE_MODEL_PATH}/checkpoints/sgdet_obj2_edg4_nore_gcn1_adj0_graph2_rel1_adjsoft_embed_stack_softadjmat_nopassedggcnembed_lr6e5_2graphbias_vgg/vgrel-27.tar \
        -save_dir  ${SAVE_MODEL_PATH}/checkpoints/sgdet_obj2_edg4_nore_gcn1_adj0_graph2_rel1_adjsoft_embed_stack_softadjmat_nopassedggcnembed_lr6e5_2graphbias_vgg/ \
        -nepoch 50 -p 100 -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack
fi
