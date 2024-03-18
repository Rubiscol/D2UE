# D2UE
Official PyTorch implementation for our paper "Revisiting Deep Ensemble Uncertainty for Enhanced Medical Anomaly Detection"

**An overview of our method:**
<p align="center">
  <img width="800"  src="./images/method.png">
</p>

Overview of D2UE: (a) During training, a junior learns through a feature path that is distinct from those of all its seniors. (b) During the inference stage, DSU incorporates both output and input-gradient information into the uncertainty estimation. (c) An illustration of over-generalization problem in the output space in 1D regression with two neural networks. The red region represents anomaly while the blue region represents normal data. Two functions output the same at the bottom point, despite divergent directions.

**A simple illustration of the redundancy-aware repulsion:**
<div align="center">
  <img width="800"  src="./images/intro.png">
</div>

Illustration of redundancy-aware repulsion (RAR). <b>Left</b>: During inference, all learners reconstruct samples from repulsed feature spaces to output space. In output space, normal features converge to agreement guided by the reconstruction training while the anomaly's disagreement is amplified. <b>Right</b>: A t-SNE plot of feature spaces from three learners on the anomaly. Feature spaces are repulsed by the redundancy-aware repulsion during training.

**Visualization results**
<div align="center">
  <img width="800"  src="./images/visualizations.png">
</div>

## Data Preparation

Organize the Med-AD benchmarks manually follow the [guidance](https://github.com/caiyu6666/DDAD-ASR/tree/main/data).


## Environment

- NVIDIA GeForce RTX 3090
- Python 3.8.16
- Cuda 11.7
```
conda create --name d2ue python=3.8.16
conda activate d2ue
pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117
```
### Packages

```
pip install -r requirements.txt
```



## Train and Evaluate

All scripts are available in `scripts/`, and configuration files are in `cfgs/`.
For example, you can choose to train and evaluate the method on RSNA dataset using AE as the backbone: `./scripts/RSNA_AE.sh`


The trained models and results are available [here](https://github.com/Rubiscol/D2UE/releases/tag/publish).



## Acknowledgement

We appreciate these open-source codes and datasets for implementing our paper:

### Codes

1. https://github.com/caiyu6666/DDAD-ASR
2. https://github.com/jayroxis/CKA-similarity

### Datasets

1. [RSNA Pneumonia Detection Challenge dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
2. [Vin-BigData Chest X-ray Abnormalities Detection dataset (VinDr-CXR)](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
3. [Brain Tumor MRI dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
4. [Large-scale Attention-based Glaucoma (LAG) dataset](https://github.com/smilell/AG-CNN)


## Citation

If this work is helpful for you, please cite our papers:

```

```
