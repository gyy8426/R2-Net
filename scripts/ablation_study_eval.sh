#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
export PYTHONPATH=/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn:/home/guoyuyu/lib/python_lib/coco/PythonAPI
export CUDA_VISIBLE_DEVICES=0
export SAVE_MODEL_PATH=/mnt/data/guoyuyu/datasets/visual_genome/model/neural-motifs_graphcnn/
if [ $1 == "0" ]; then
    echo "rel_model_nolstm" 
    export CUDA_VISIBLE_DEVICES=0
    python3 -u models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack

	'''
	python3 -u models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
    python3 -u models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
    python3 -u models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
    python3 -u models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgepos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred
	'''
elif [ $1 == "1" ]; then
    echo "rel_model_notop2gcn"
    export CUDA_VISIBLE_DEVICES=1
    python3 -u models/eval_rels_1.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgenopos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels_1.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgenopos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels_1.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgenopos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
    python3 -u models/eval_rels_1.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgenopos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
    python3 -u models/eval_rels_1.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgenopos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels_1.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgenopos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels_1.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgenopos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
    python3 -u models/eval_rels_1.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat_objedgenopos/vgrel-8.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred
elif [ $1 == "2" ]; then
    echo "rel_model_nolstm2"
    export CUDA_VISIBLE_DEVICES=0
    python3 -u models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 0 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_nolstm2/vgrel-4.tar \
        -nepoch 50 -resnet -p 100 -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack
elif [ $1 == "3" ]; then
    echo "rel_model_nogcn_withpriorlabel"
    export CUDA_VISIBLE_DEVICES=1
    python3 -u models/train_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt /mnt/data/guoyuyu/datasets/visual_genome/model/neural-motifs_nongt/checkpoints/vgdet_res101_layer4_finished/vg-24.tar \
        -save_dir  ${SAVE_MODEL_PATH}/ablation_study/rel_model_nogcn_withpriorlabel/ \
        -nepoch 50 -resnet -p 100 -use_bias \
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1
elif [ $1 == "4" ]; then
    echo "rel_model_nolstm" 
    export CUDA_VISIBLE_DEVICES=1
    python3 -u models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat/vgrel-7.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat/vgrel-7.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack;
    python3 -u models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat/vgrel-7.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred;
    python3 -u models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -nl_adj 0 -b 24 -clip 5\
        -pooling_dim 2048 -hidden_dim 512 -adj_embed_dim 512 \
        -lr 2e-2 -ngpu 1 -ckpt ${SAVE_MODEL_PATH}/ablation_study/rel_model_edgelstm_regfeat/vgrel-7.tar  \
        -nepoch 50 -resnet -p 100 -use_bias -test -gt_adj_mat\
        -with_adj_mat -bg_num_graph 2 -bg_num_rel 1 -with_gcn -with_biliner_score -gcn_adj_type soft -where_gcn stack -multipred
fi
