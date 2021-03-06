Neural Clarity Learning
==
Implementation of [A Neural Local Coherence Analysis Model for Text Clarity Scoring](https://www.aclweb.org/anthology/2020.coling-main.194)

## Installation
1. Download and install [Anaconda](https://www.anaconda.com/products/individual)
2. Create environment
```
conda env create -f environment.yml
conda activate local_coh
```

## Dataset
1. [Discourse Graphbank](https://catalog.ldc.upenn.edu/LDC2005T08)
2. [PeerRead](https://github.com/allenai/PeerRead)

## Training

1. Training Coherence Model with Discourse Graphbank dataset
```
cd discourse
python preprocessing.py -i ./path/to/discourse/data
python train.py
```
2. Training Single Clarity Score Prediction Model with PeerRead dataset
```
cd ../peerread
python preprocessing.py -i ./path/to/PeerRead/data/acl_2017 ./path/to/PeerRead/data/iclr_2017 -l 6
python single.py
```
3. Training Combined Clarity Score Prediction Model with PeerRead dataset
```
python combined.py
```
