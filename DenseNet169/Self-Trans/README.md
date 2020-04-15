# COVID-CT


**We provide our current best model `Self-Trans` in this repo**
The `Self-Trans` model are trained by two steps:
First step: call main.py to run MoCo on LUNA dataset. Then change the dataset to COVID-CT and run MoCo on COVID-CT. To do MoCo correctly, you need to change the path for dataset in line 48 and 238 of `main_coco.py`.
Second stop: Load MoCo pretrained model in line 17 of `CT_predict-efficient-pretrain.ipynb` and do training.
 

### How to use
In `CT_predict-efficient-pretrain.ipynb`, you can load our pretrained model in [30], change the name and path to our provided Self-Trans.pt. The result of this model achieves an F1-score of 0.85.


### Pretrained model
Self-Trans.pt


### Reference 
The MoCo method refers to the work in 

    @Article{chen2020mocov2,
      author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
      title   = {Improved Baselines with Momentum Contrastive Learning},
      journal = {arXiv preprint arXiv:2003.04297},
      year    = {2020},
    }


 


