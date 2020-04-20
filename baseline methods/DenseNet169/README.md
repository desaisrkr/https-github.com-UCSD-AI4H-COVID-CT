# DenseNet169 baseline

We provide here the PyTorch training script to train and test the model in this repo.

## Requirements

The main requirements are listed below:

* Pytorch
* re
* skimage
* torchvision
* Python 3.6
* Numpy
* OpenCV
* Scikit-Learn
* skimage

<!---
# Dataset Split
See Data-split. Patient distribution in each set will be updated soon.
--->
# Steps to generate the dataset used to do Training and Testing
1. Download images from repo `Images-processed`
2. Download txt files for image names in train, val, and test set from Data-split repo
3. Use the dataloader defined in line `80` of the script `DenseNet_predict.py` and load the dataset


# Dataset Distribution
<!---
--->
Images distribution
|  Type | NonCOVID-19 | COVID-19 |  Total |
|:-----:|:-----------:|:--------:|:------:|
| train |      234    |    191   |   425  |
|  val  |       58    |     60   |   118  |
|  test |      105    |     98   |   203  |

Patients distribution
|  Type |    NonCOVID-19   | COVID-19 |  Total |
|:-----:|:----------------:|:--------:|:------:|
| train |        105       |  1-130   |   235  |
|  val  |         24       | 131-162  |    56  |
|  test |         42       | 163-216  |    96   |



* Max CT scans per patient in COVID: 16.0 (patient 2)
* Average CT scans per patient in COVID: 1.6
* Min CT scans per patient in  COVID: 1.0
<!---
Patients frequency ('ID:number')
* train: 12:18  13:9  14:2  15:12  17:20  18:16  19:12  21:8  23:40  24:22  25:11  34:12
* val: 6:26  16:10  27:22 
* test: 7:4  8:8  10:8  11:3  20:12
--->


## Training and Evaluation
   In [144] of the script.
   Train process is defined in line `190` of the script and val process is defined in line `241`. 
   Loading the pretrained DenseNet model in line `488` and start training in line `535`, can either train from scratch by `pretrain = false` or transfered from the ImageNet pretrained model.  the performance on val set is observed in line `561`. It will predict the target value and the predict value per epoch, and print the F1-score, accuray and AUC of 10 model major vote per 10 epoch. 

## Test
   In [145] of the script. Line `617`. 

## Initial result
   See test_Dense169.txt
