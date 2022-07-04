# FEDVSSL
This is a general purpose repository for Federated Self-Superivised Learning build on top of [MMCV](https://mmcv.readthedocs.io/en/latest/) and [Flower](https://flower.dev/)

The paper can be found at 

# Dataset:
we use [Kinetics-400](https://www.deepmind.com/open-source/kinetics) for Centralized and FL pretraining. For evaluation, we use [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

One can generate the non-iid version of kinetics-400 with 100 clients by running the ````python scripts/k400_non_iid.py```` and the iid version of kinetics-400 with 100 clients by running ````python scripts/kinetics_json_splitter.py````. Note that these two codes assume that you have already downloaded the official trainlist of kineticss-400. 

As a part of reproducibility, we have provideded the Federated version of Kinetics-400 dataset in the  [Data](https://github.com/yasar-rehman/FEDVSSL/tree/master/DATA)
folder with iid and non-iid data distribution. 

# FL pretrained Models
We provide a series of federated-SSL pretrined models of [VCOP](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.pdf), [Speed](https://arxiv.org/pdf/2004.06130.pdf), and [Ctp](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Unsupervised_Visual_Representation_Learning_by_Tracking_Patches_in_Video_CVPR_2021_paper.pdf)

The models can be downloaded at ... (TBA)

# News
# Dependencies
- [x] [Anaconda environment with python 3.8](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) 
- [x] [Flower](https://flower.dev/) <br>
- [x] Microsoft [Ctp Framework](https://github.com/microsoft/CtP)
- [x] [MMCV](https://mmcv.readthedocs.io/en/latest/)

# Instructions
We recommend installing Microsoft [Ctp Framework](https://github.com/microsoft/CtP) as it contain all the Self-supervised learning frameworks build on top of MMCV framework. Once intsalled, you can use the "config" file to create 
# Running Experiments
The abstract definition of classes are provided by ````reproduce_papers/fedssl/videossl.py````. <br>
FedVSSL $(\alpha=0, \beta=0)$, run ````python main_cam_st_theta_b_wo_moment.py````. <br>
FedVSSL $(\alpha=0, \beta=0)$ is the implementation of FedAvg but with only aggregating the backbone network. If you want to run federate the SSL method using the conventional FedAvg method, then run ````python main.py````.
# Citations
