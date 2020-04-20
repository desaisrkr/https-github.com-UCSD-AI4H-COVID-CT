# Self-Trans 


**We provide our current best model `Self-Trans` in this repo**

The `Self-Trans` model are trained by two steps:

*First step*: Load the model pretrained on ImageNet. Call main.py to run MoCo on LUNA dataset. Then run MoCo on COVID-CT by change the path for dataset in line 48 and 238 of `main_coco.py`. To do MoCo, 4 or 8 GPUs are needed.

*Second step*: Load MoCo pretrained model in line [17] of `CT_predict-efficient-pretrain.ipynb` and do training.
 
### Environment
The code is based on Python 3.7 and PyTorch 1.3

The MoCO code is run on four GTX1080Ti with batch size 128. The pretrained model is finetuned on one GTX1080Ti.


### Dataset
Use the split in `Data-split`.


### Pretrained model
See `Self-Trans.pt` with DenseNet-169 backbone.


### How to use our Pretrained model
We provide an example notebook file `CT_predict-efficient-pretrain.ipynb`, the pretrained model is loaded in [30] . Change the name and path to our provided Self-Trans.pt to load correctly. The model achieves an F1-score of 0.85 on the test set.

### Results
F1: 0.85

Accuracy: 0.86

AUC: 0.94

### Reference 
The details of this approach are described in this [preprint](https://www.medrxiv.org/content/10.1101/2020.04.13.20063941v1).

If you find the code useful, please cite:


    @Article{he2020sample,
      author  = {He, Xuehai and Yang, Xingyi and Zhang, Shanghang, and Zhao, Jinyu and Zhang, Yichen and Xing, Eric, and Xie, Pengtao},
      title   = {Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans},
      journal = {medrxiv},
      year    = {2020},
    }

The code of the MoCo part can refer to [https://github.com/facebookresearch/moco](https://github.com/facebookresearch/moco)

 


