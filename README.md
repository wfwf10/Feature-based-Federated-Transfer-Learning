# FbFTL: Communication-Efficient Feature-based Federated Transfer Learning

This is the offical implementation for Python simulation of Feature-based Federated Transfer Learning (FbFTL), from the following conference paper and journal paper: 

  Communication-Efficient Feature-based Federated Transfer Learning.([Globecom2022](https://ieeexplore.ieee.org/abstract/document/10000612), [arXiv](https://arxiv.org/abs/2209.05395))  
Feng Wang, M. Cenk Gursoy and Senem Velipasalar  
Department of Electrical Engineering and Computer Science, Syracuse University

  Feature-based Federated Transfer Learning: Communication Efficiency, Robustness and Privacy.([TMLCN], [arXiv], comming soon)  
Feng Wang, M. Cenk Gursoy and Senem Velipasalar  
Department of Electrical Engineering and Computer Science, Syracuse University

---

<img src="https://github.com/wfwf10/Feature-based-Federated-Transfer-Learning/blob/main/diagrams/FbFTL_diagram.png" width="644" height="501">

We propose the FbFTL as an innovative federated learning approach that upload features and outputs instead of gradients to reduce the uplink payload by more than five orders of magnitude. Please refer to the journal paper for explicit explaination on learning structure, system design, robustness analysis, and privacy analysis.


# Results on CIFAR-10 Dataset with VGG16 Model
In the following table, we provide comparison between federated learning with [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html) (FL), federated transfer learning with FedAvg that updating full model (FTL<sub>f</sub>), federated transfer learning with FedAvg that updating task-specific sub-model(FTL<sub>c</sub>), and FbFTL. All of them learn [VGG16](https://arxiv.org/abs/1409.1556) model on [CIFAR-10](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf) dataset. For transfer learning approaches, the source models are trained on [ImageNet](https://ieeexplore.ieee.org/abstract/document/5206848?casa_token=QncCRBM1tzAAAAAA:QuoJhjJAHRplmLJ4jcFw5JWdfASjmbIVlvpCrHgTPIFu63gpSUlBeACB78S0AH34qqQnsBOdoQ) dataset. Compared to all other methods, FbFTL reduces the uplink payload by up to five orders of magnitude. 

| | FL | FTL<sub>f</sub> | FTL<sub>c</sub> | FbFTL  |
| ---- | ----- | ---- | ---- | ---- |
| upload batches | 656250 | 193750 | 525000 | 50000 |
| upload parameters per batch | 153144650 | 153144650 | 35665418 | 4096 |
| uplink payload per batch | **4.9 Gb** | **4.9 Gb** | **1.1 Gb** | **131 Kb**  |
| total uplink payload | **3216 Tb** | **949 Tb** | **599 Tb** | **6.6 Gb** |
| total downlink payload | 402 Tb | 253 Tb | 322 Tb | 3.8 Gb |
| test accuracy | 89.42\% | 93.75\% | 86.51\% | 86.51\% |

# Results on SAMSum summary task with FLAN-T5-small language model
In the following table, we consider [FLAN-T5-small](https://www.jmlr.org/papers/volume25/23-0870/23-0870.pdf) as a pre-trained language model, and fine-tune on [SAMSum](https://www.aclweb.org/anthology/D19-5409) summary task. As a fine-tuning task, this experiment does not include an FL setting, and we provide comparison between federated transfer learning with FedAvg that updating full model (FTL<sub>f</sub>), federated transfer learning with FedAvg that updating task-specific sub-model(FTL<sub>c</sub>), and FbFTL.  Compared to all other methods, FbFTL reduces the uplink payload by up to five orders of magnitude. 

|  | FTL<sub>f</sub> | FTL<sub>c</sub> | FbFTL  | FTL<sub>c</sub> | FbFTL  | FTL<sub>c</sub> | FbFTL  |
| ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- |
| number of trained encoders | 8 | 8 | 8 | 4 | 4 | 2 | 2 |
| number of upload batches | 132588 | 36830 | 7366 | 88392 | 7366 | 103124 | 7366 |
| upload parameters per batch | 109860224 | 60511616 | 1024 | 51070144 | 1024 | 46349504 | 1024 |
| uplink payload per batch  | **3.5 Gb** | **1.9 Gb** | **32.7 Kb** | **1.6 Gb** | **32.7 Kb** | **1.5 Gb** | **32.7 Kb** |
| total uplink payload | **466.1 Tb** | **71.3 Tb** | **241.4 Mb** | **144.5 Tb** | **241.4 Mb** | **152.9 Tb** | **241.4 Mb** |
| total downlink payload | 116.0 Tb | 32.2 Tb | 1.58 Gb | 77.3 Tb | 1.88 Gb | 90.2 Tb | 2.03 Gb |
| test ROUGE-1 | 45.9249 | 45.4680 | 45.4680 | 45.2827 | 45.2827 | 44.9862 | 44.9862  |

# Required packages installation
We use python==3.6.9, numpy==1.19.5, torch==1.4.0, torchvision==0.5.0, and CUDA version 11.6 for the experiments on CIFAR-10 with VGG16. The dataset and the source model will be automatically downloaded.

Additionally, for the experiments on SAMSUM with FLAN-T5, we use transformers==4.30.2, torchinfo==1.8.0, datasets==2.13.2, nltk==3.8.1, evaluate==0.4.1, huggingface_hub==0.16.4

# Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{wang2022communication,
  title={Communication-Efficient and Privacy-Preserving Feature-based Federated Transfer Learning},
  author={Wang, Feng and Gursoy, M Cenk and Velipasalar, Senem},
  booktitle={GLOBECOM 2022-2022 IEEE Global Communications Conference},
  pages={3875--3880},
  year={2022},
  organization={IEEE}
}
```
Journal version comming soon
