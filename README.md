# COVID-CT


**We are continuously adding new COVID CT images and we would like to invite the community to contribute COVID CTs as well.**

 

### Data Description

The COVID-CT-Dataset has 349 CT images containing clinical findings of COVID-19 from 216 patients. They are in `./Images-processed/CT_COVID.zip`

Non-COVID CT scans are in `./Images-processed/CT_NonCOVID.zip`

We provide a data split in `./Data-split`.
Data split information see `README for DenseNet_predict.md`

The meta information (e.g., patient ID, patient information, DOI, image caption) is in `COVID-CT-MetaInfo.xlsx`


The images are collected from COVID19-related papers from medRxiv, bioRxiv, NEJM, JAMA, Lancet, etc. CTs containing COVID-19 abnormalities are selected by reading the figure captions in the papers. All copyrights of the data belong to the authors and publishers of these papers.

The dataset details are described in this preprint: [COVID-CT-Dataset: A CT Scan Dataset about COVID-19](https://arxiv.org/pdf/2003.13865.pdf)

If you find this dataset and code useful, please cite:

    @article{zhao2020COVID-CT-Dataset,
      title={COVID-CT-Dataset: a CT scan dataset about COVID-19},
      author={Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao},
      journal={arXiv preprint arXiv:2003.13865}, 
      year={2020}
    }

### Baseline Performance
We developed two baseline methods for the community to benchmark with.
The code are in the "baseline methods" folder and the details are in the readme files under that folder. The methods are described in [Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans](https://www.medrxiv.org/content/10.1101/2020.04.13.20063941v1)

If you find the code useful, please cite:

    @Article{he2020sample,
      author  = {He, Xuehai and Yang, Xingyi and Zhang, Shanghang, and Zhao, Jinyu and Zhang, Yichen and Xing, Eric, and Xie,       Pengtao},
      title   = {Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans},
      journal = {medrxiv},
      year    = {2020},
    }


### Contribution Guide
 - To contribute to our project, please email your data to jiz077@eng.ucsd.edu with the corresponding meta information (Patient ID, DOI and Captions).
 - We recommend you also extract images from publications or preprints. Make sure the original papers you crawled have different DOIs from those listed in `COVID-CT-MetaInfo.xlsx`.
 - In `COVID-CT-MetaInfo.xlsx`, images with the form of `2020.mm.dd.xxxx` are crawled from bioRxiv or medRxiv. The DOIs for these preprints are `10.1101/2020.mm.dd.xxxx`.
 


