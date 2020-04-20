## Requirements

The main requirements are listed below:

* Pytorch
* re
* skimage
* torchvision
* OpenCV 4.2.0
* Python 3.6
* Numpy
* OpenCV
* Scikit-Learn
* skimage

<!---
# Dataset Split
See Data-split. Patient distribution in each set will be updated soon.
--->
# Dataset Split
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
   In [144] of the script

## Test
   In [145] of the script. Do a 10 major vote.

### Steps for training
   Follow the script, can either train from scratch or do transfer learning. 

## Initial result
   See test_Dense169.txt
