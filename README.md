# FEDVSSL
This is a general purpose repository for Federated Self-Superivised Learning for video understanding build on top of [MMCV](https://mmcv.readthedocs.io/en/latest/) and [Flower](https://flower.dev/).
<p align="center">
<img src="https://github.com/yasar-rehman/FEDVSSL/blob/master/FVSSL.png"/>
 </p>

The paper can be found at (link to the ArXiv will be added soon!)

# Authors
- [Yasar Abbas Ur Rehman](https://yasar-rehman.github.io/yasar/) <br>
- Gao Yan <br>
- [pedro porto buarque de gusmão](https://www.linkedin.com/in/pedropgusmao/?originalSubdomain=uk) <br>
- [Jiajun Shen](https://www.linkedin.com/in/jiajunshen/) <br>
- [Nicholas D. Lane](http://niclane.org/) <br>

# Dataset:
For both centralized and federated video SSL pretraining, we use [Kinetics-400](https://www.deepmind.com/open-source/kinetics). We evaluate the quality of learned representations by applying them on two downstream datasets: [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

One can generate the non-iid version of kinetics-400 with 100 clients by running the ````python scripts/k400_non_iid.py```` and the iid version of kinetics-400 with 100 clients by running ````python scripts/kinetics_json_splitter.py````. Note that these two codes assume that you have already downloaded the official trainlist of kineticss-400. 

As a part of reproducibility, we have provideded the dataset partitions of the Kinetics-400 dataset for federated learning in the [Data](https://github.com/yasar-rehman/FEDVSSL/tree/master/DATA)
folder with iid and non-iid data distribution. 

# FL pretrained Models
We provide a series of federated-SSL pretrined models of [VCOP](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.pdf), [Speed](https://arxiv.org/pdf/2004.06130.pdf), and [Ctp](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Unsupervised_Visual_Representation_Learning_by_Tracking_Patches_in_Video_CVPR_2021_paper.pdf) (* TODO: need to be specific what dataset and what data partition (iid or non-iid) are used for this model pretraining)

The models can be downloaded at ... (TBA)

# News
- [x] Checkout the teaser of our work on the [YouTube](https://www.youtube.com/watch?v=ZLqst0lVte8&list=PLNG4feLHqCWni5zfOBaZNtaPlCce0OnJ6&index=8). 
# Dependencies (* TODO: add dependency to a requirements.txt file so users can pip install)
- [x] [Anaconda environment with python 3.8](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) 
- [x] [Flower](https://flower.dev/) <br>
- [x] Microsoft [Ctp Framework](https://github.com/microsoft/CtP)
- [x] [MMCV](https://mmcv.readthedocs.io/en/latest/)

For a complete list of the required packages please see the [requirement.txt](https://github.com/yasar-rehman/FEDVSSL/blob/master/requirement.txt) file. One can easily install, all the requirement by running ````pip install -r requirement.txt````.  

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

## Expected Results
| Method  | Checkpoint file | UCF R@1 | HMDB R@1|
|---------|-----------------|---------|---------|
|FedVSSL$(\alpha=0, \beta=0)$ |[round-540.npz](https://drive.google.com/file/d/15EEIQay5FRBMloEzt1SQ8l8VjZFzpVNt/view?usp=sharing) | 34.34 |15.82  |
|FedVSSL$(\alpha=1, \beta=0)$ |[round-540.npz](https://drive.google.com/file/d/1OUj8kb0ahJSKAZEB-ES94pOG5-fB-28-/view?usp=sharing) | 34.23 |16.73  |
|FedVSSL$(\alpha=0, \beta=1)$ |[round540.npz](https://drive.google.com/file/d/1N62kXPcLQ_tM45yd2kBYjNOskdHclwLM/view?usp=sharing)  | 35.61 | 16.93 |
|FedVSSL$(\alpha=1, \beta=1)$ |[round540.npz](https://drive.google.com/file/d/1SKb5aXjpVAeWbzTKMFN9rjHW_LQsmUXj/view?usp=sharing)  | 35.66 | 16.41 |


# Issues: 
If you encounter any issues, feel free to open an issue in the github 


# Citations

# Acknowledgement
We would like to thank [Daniel J. Beutel](https://github.com/danieljanes) for providing the initial blueprint of Federated self-supervised learning with flower. Also thanks to [Akhil Mathur](https://akhilmathurs.github.io/index.html) for providing the useful suggestions.
