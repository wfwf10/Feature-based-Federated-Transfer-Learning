# FbFTL: Communication-Efficient Feature-based Federated Transfer Learning

This is the offical implementation for Python simulation of Feature-based Federated Transfer Learning (FbFTL), from the following paper: 

  [Communication-Efficient Feature-based Federated Transfer Learning.](https://www.google.com) Globecom2022.  
  Feng Wang, M. Cenk Gursoy and Senem Velipasalar  
Department of Electrical Engineering and Computer Science, Syracuse University

---

<img src="https://github.com/wfwf10/Feature-based-Federated-Transfer-Learning/blob/main/diagrams/FbFTL_diagram.png" width="644" height="501">

We propose the FbFTL as an innovative federated learning approach that upload features and outputs instead of gradients to reduce the uplink payload by more than five orders of magnitude. Please refer to the paper for explicit explaination on learning structure, system design, and privacy analysis.


# Results on CIFAR-10 Dataset
In the following table, we provide comparison between federated learning with [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html) (FL), federated transfer learning with FedAvg that updating full model (FTL<sub>f</sub>), federated transfer learning with FedAvg that updating task-specific sub-model(FTL<sub>c</sub>), and FbFTL. All of them learn [VGG16](https://arxiv.org/abs/1409.1556) model on [CIFAR-10](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf) dataset. For transfer learning approaches, the source models are trained on [ImageNet](https://ieeexplore.ieee.org/abstract/document/5206848?casa_token=QncCRBM1tzAAAAAA:QuoJhjJAHRplmLJ4jcFw5JWdfASjmbIVlvpCrHgTPIFu63gpSUlBeACB78S0AH34qqQnsBOdoQ) dataset. Compared to all other methods, FbFTL reduces the uplink payload by up to five orders of magnitude. 

| | FL | FTL<sub>f</sub> | FTL<sub>c</sub> | FbFTL  |
| ---- | ----- | ---- | ---- | ---- |
| upload batches | 656250 | 193750 | 525000 | 50000 |
| upload parameters per batch | 153144650 | 153144650 | 35665418 | 4096 |
| uplink payload per batch | **4.9 Gb** | **4.9 Gb** | **1.1 Gb** | **131 Kb**  |
| total uplink payload | **3216 Tb** | **949 Tb** | **599 Tb** | **6.6 Gb** |
| total downlink payload | 402 Tb | 253 Tb | 322 Tb | 3.8 Gb |
| test accuracy | 89.1\% | 91.68\% | 85.59\% | 85.59\% |

# Required packages installation
We use python==3.6.9, numpy==1.19.5, torch==1.4.0, torchvision==0.5.0, and CUDA version 11.6. The dataset and the source model will be automatically downloaded.

# Citation
If you find our work useful in your research, please consider citing:
```
@INPROCEEDINGS{Gurs2212:Communication,
AUTHOR="Feng Wang and M. Cenk Gursoy and Senem Velipasalar",
TITLE="{Communication-Efficient} and {Privacy-Preserving} Feature-based Federated
Transfer Learning",
BOOKTITLE="2022 IEEE Global Communications Conference: Selected Areas in
Communications: Machine Learning for Communications (Globecom2022 SAC MLC)",
ADDRESS="Rio de Janeiro, Brazil",
DAYS="4",
MONTH=dec,
YEAR=2022,
KEYWORDS="federated learning; transfer learning; communication efficiency; payload;
privacy",
ABSTRACT="Federated learning has attracted growing interest as it preserves the
clients' privacy. As a variant of federated learning, federated transfer
learning utilizes the knowledge from similar tasks and thus has also been
intensively studied. However, due to the limited radio spectrum, the
communication efficiency of federated learning via wireless links is
critical since some tasks may require thousands of Terabytes of uplink
payload. In order to improve the communication efficiency, we in this paper
propose the feature-based federated transfer learning as an innovative
approach to reduce the uplink payload by more than five orders of magnitude
compared to that of existing approaches. We first introduce the system
design in which the extracted features and outputs are uploaded instead of
parameter updates, and then determine the required payload with this
approach and provide comparisons with the existing approaches.
Subsequently, we analyze the random shuffling scheme that preserves the
clients' privacy. Finally, we evaluate the performance of the proposed
learning scheme via experiments on an image classification task to show its
effectiveness."
}
```
