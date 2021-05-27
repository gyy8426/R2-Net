# R2-Net
Code for Relation Regularized Scene Graph Generation



### Bibtex

```
@ARTICLE{9376912,
  author={Guo, Yuyu and Gao, Lianli and Song, Jingkuan and Wang, Peng and Sebe, Nicu and Shen, Heng Tao and Li, Xuelong},
  journal={IEEE Transactions on Cybernetics}, 
  title={Relation Regularized Scene Graph Generation}, 
  year={2021},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TCYB.2021.3052522}}
```
# Setup


0. Install python3.6 and pytorch 3. I recommend the [Anaconda distribution](https://repo.continuum.io/archive/). To install PyTorch if you haven't already, use
 ```conda install pytorch torchvision cuda90 -c pytorch```.
1. Update the config file with the dataset paths. Specifically:
    - Visual Genome (the VG_100K folder, image_data.json, VG-SGG.h5, and VG-SGG-dicts.json). See data/stanford_filtered/README.md for the steps I used to download these.
    - You'll also need to fix your PYTHONPATH: ```export PYTHONPATH=/home/guoyuyu/code/scene-graph``` 

2. Compile everything. run ```make``` in the main directory: this compiles the Bilinear Interpolation operation for the RoIs as well as the Highway LSTM.

3. Pretrain VG detection. The old version involved pretraining COCO as well, but we got rid of that for simplicity. Run ./scripts/pretrain_detector.sh
Note: You might have to modify the learning rate and batch size, particularly if you don't have 3 Titan X GPUs (which is what I used). [You can also download the pretrained detector checkpoint here.](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX)

4. Train VG scene graph classification: run ./scripts/train_models_sgcls.sh. 
5. Refine for detection: run ./scripts/refine_for_detection.sh.
6. Evaluate: Refer to the scripts ./scripts/eval_models_sg[cls/det].sh.

# help

Feel free to ping me if you encounter trouble getting it to work!
