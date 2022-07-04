# FEDVSSL
This is a general purpose repository for Federated Self-Superivised Learning for video understanding build on top of [MMCV](https://mmcv.readthedocs.io/en/latest/) and [Flower](https://flower.dev/).
![https://github.com/yasar-rehman/FEDVSSL/blob/master/FVSSL.png](FVSSL)

The paper can be found at 

# Authors
[Yasar Abbas Ur Rehman](https://yasar-rehman.github.io/yasar/)
[Gao Yan]
[pedro porto buarque de gusmão](https://www.linkedin.com/in/pedropgusmao/?originalSubdomain=uk)
[JiaJun Shen](https://www.linkedin.com/in/jiajunshen/)
[Nicholas D. Lane](http://niclane.org/)

# Dataset:
we use [Kinetics-400](https://www.deepmind.com/open-source/kinetics) for Centralized and FL pretraining. For evaluation, we use [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

One can generate the non-iid version of kinetics-400 with 100 clients by running the ````python scripts/k400_non_iid.py```` and the iid version of kinetics-400 with 100 clients by running ````python scripts/kinetics_json_splitter.py````. Note that these two codes assume that you have already downloaded the official trainlist of kineticss-400. 

As a part of reproducibility, we have provideded the Federated version of Kinetics-400 dataset in the  [Data](https://github.com/yasar-rehman/FEDVSSL/tree/master/DATA)
folder with iid and non-iid data distribution. 

# FL pretrained Models
We provide a series of federated-SSL pretrined models of [VCOP](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.pdf), [Speed](https://arxiv.org/pdf/2004.06130.pdf), and [Ctp](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Unsupervised_Visual_Representation_Learning_by_Tracking_Patches_in_Video_CVPR_2021_paper.pdf)

The models can be downloaded at ... (TBA)

# News
- [x] Checkout the teaser of our work at the [Flower.dev-2022](https://flower.dev/conf/flower-summit-2022/) summit. 
# Dependencies
- [x] [Anaconda environment with python 3.8](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) 
- [x] [Flower](https://flower.dev/) <br>
- [x] Microsoft [Ctp Framework](https://github.com/microsoft/CtP)
- [x] [MMCV](https://mmcv.readthedocs.io/en/latest/)

# Instructions
We recommend installing Microsoft [Ctp Framework](https://github.com/microsoft/CtP) as it contain all the Self-supervised learning frameworks build on top of MMCV framework. Here we provided a modifed version of that framework for FedVSSL, in particular.
# Running Experiments
The abstract definition of classes are provided by ````reproduce_papers/fedssl/videossl.py````. <br>
FedVSSL $(\alpha=0, \beta=0)$, run ````python main_cam_st_theta_b_wo_moment.py````. <br>
FedVSSL $(\alpha=0, \beta=0)$ is the implementation of FedAvg but with only aggregating the backbone network. If you want to run federate the SSL method using the conventional FedAvg method, then run ````python main.py````.
# Evaluation
After FL pretraining one can use the following code to fine-tune the model on UCF or HMDB.
````
import subprocess
import os
process_obj = subprocess.run(["bash", "/home/root/yasar/reproduce_papers/tools/dist_train.sh",\
"/configs/vcop/r3d_18_kinetics/finetune_hmdb51.py", "4",\
f"--work_dir /centralized_vcop/centralized_finetune/hmdb51/",
f"--data_dir /home/data3/DATA/",\
f"--pretrained /path to the pretrained checkpoint",\
f"--validate"])

````
The complete list of such methods are provided under the Task folder
# Issues: 
If you encounter any issues, feel free to open an issue in the github 


# Citations

# Acknowledgement
We would like to thank [Daniel J. Beutel](https://github.com/danieljanes) for providing the initial blueprint of Federated self-supervised learning with flower. Also thanks to [Akhil Mathur](https://akhilmathurs.github.io/index.html) for providing the useful suggestions.
