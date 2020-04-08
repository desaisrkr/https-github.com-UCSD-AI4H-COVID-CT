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
Images distribution
|  Type | Normal | COVID-19 |  Total |
|:-----:|:------:|:--------:|:------:|
| train |  146   |    182   |   328  |
|  val  |   15   |     58   |    73  |
|  test |   34   |     35   |    69  |
--->
Patients distribution
|  Type |    NonCOVID-19   | COVID-19 |  Total |
|:-----:|:----------------:|:--------:|:------:|
| train |  See data_split  |  1-130   |    -   |
|  val  |  See data_split  | 131-162  |    -   |
|  test |  See data_split  | 163-216  |    -   |


<!---
* Max CT scans per patient: 40
* Average CT scans per patient: 13.8
* Min CT scans per patient: 2
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
