# FEDVSSL
This is a general purpose repository for Federated Self-Superivised Learning build on top of [MMCV](https://mmcv.readthedocs.io/en/latest/) and [Flower](https://flower.dev/).

The paper can be found at 

# Dataset:
we use [Kinetics-400](https://www.deepmind.com/open-source/kinetics) for Centralized and FL pretraining. For evaluation, we use [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

One can generate the non-iid version of kinetics-400 with 100 clients by running the ````python scripts/k400_non_iid.py```` and the iid version of kinetics-400 with 100 clients by running ````python scripts/kinetics_json_splitter.py````. Note that these two codes assume that you have already downloaded the official trainlist of kineticss-400. 

As a part of reproducibility, we have provideded the Federated version of Kinetics-400 dataset in the  [Data](https://github.com/yasar-rehman/FEDVSSL/tree/master/DATA)
folder with iid and non-iid data distribution. 
