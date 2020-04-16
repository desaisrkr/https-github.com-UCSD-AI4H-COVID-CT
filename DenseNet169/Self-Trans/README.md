# COVID-CT


**We provide our current best model `Self-Trans` in this repo**

The `Self-Trans` model are trained by two steps:

*First step*: Load the model pretrained on ImageNet. Call main.py to run MoCo on LUNA dataset. Then run MoCo on COVID-CT by change the path for dataset in line 48 and 238 of `main_coco.py`. To do MoCo, 4 or 8 GPUs are needed.

*Second step*: Load MoCo pretrained model in line 17 of `CT_predict-efficient-pretrain.ipynb` and do training.
 

### How to use our Pretrained model
In `CT_predict-efficient-pretrain.ipynb`, the pretrained model is loaded in [30]. Change the name and path to our provided Self-Trans.pt to load correctly. The model achieves an F1-score of 0.85 on the test set.


### Pretrained model
Self-Trans.pt with DenseNet-169 backbone.


### Reference 
The MoCo method thanks to the work in 

    @Article{chen2020mocov2,
      author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
      title   = {Improved Baselines with Momentum Contrastive Learning},
      journal = {arXiv preprint arXiv:2003.04297},
      year    = {2020},
    }

The code of the MoCo part can refer to [https://github.com/facebookresearch/moco](https://github.com/facebookresearch/moco)

 


