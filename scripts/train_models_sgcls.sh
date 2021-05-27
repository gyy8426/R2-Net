#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
export PYTHONPATH=/home/guoyuyu/code/scene_graph_gen/r2net:/home/guoyuyu/guoyuyu/lib/python_lib/coco/PythonAPI
export CUDA_VISIBLE_DEVICES=0
export SAVE_MODEL_PATH=/mnt/data1/guoyuyu/datasets/visual_genome/model/r2net/
if [ $1 == "0" ]; then
    echo "TRAINING THE BASELINE"
    python3 models/train_rels.py -m sgcls -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/baseline2 \
    -nepoch 50 -use_bias
elif [ $1 == "1" ]; then
    echo "TRAINING MESSAGE PASSING"
    python3 models/train_rels.py -m sgcls -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/stanford2
elif [ $1 == "2" ]; then
    echo "TRAINING MOTIFNET"
    python3 models/train_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 2048 -lr 1e-3 -ngpu 1 -ckpt /home/guoyuyu/code/code_by_other/neural-motifs_nongt/checkpoints/vgdet_res101_layer4_finished/vg-24.tar \
        -save_dir checkpoints/motifnet2_sgcls_res101_layer4_24 -nepoch 50 -use_bias -resnet -need_bg -bg_num_rel 0
elif [ $1 == "3" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "TRAINING MOTIFNET ResNet" 
    python3 -u models/train_rels.py -m sgcls -model motifnet -order leftright \
        -nl_obj 12 -nl_edge 12 -nh_obj 12 -nh_edge 12 \
        -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 768 \
        -lr 1e-3 -ngpu 1 \
        -ckpt /mnt/data1/guoyuyu/datasets/visual_genome/model/neural-motifs_nongt/checkpoints/vgdet_res101_layer4_finished/vg-24.tar \
        -save_dir  ${SAVE_MODEL_PATH}/checkpoints/nlobj_12_nledge_12_nhobj_12_nlobj_12_hiddim_768_resnet_parall \
        -nepoch 50 -resnet -p 100 -use_bias \
        -bg_num_graph 0 -bg_num_rel 2 -with_biliner_score
fi
