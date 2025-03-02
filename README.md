# M4Semantic
Official implementation of the single semantic segmentation network presented in the paper "Co-SemDepth: Fast Joint Semantic Segmentation and Depth Estimation on Aerial Images"
![alt text](https://github.com/Malga-Vision/M4Semantic/blob/main/m4semantic_.png?raw=true)

## Overview
M4Semantic is a lightweight deep architecture for semantic segmentation given an input of RGB image captured in outdoor environments by a camera moving with 6 degrees of freedom (6 DoF). 

Please refer to [Co-SemDepth](https://github.com/Malga-Vision/Co-SemDepth/tree/main) for the implementation of the joint architecture.

## Citation

## Dependencies
Starting from a fresh Anaconda environment, you can install the required depndencies to run our code with:
```shell
conda install -c conda-forge tensorflow-gpu=2.7 numpy pandas
```

### Datasets

#### Mid-Air [[1](#ref_1)]

To download the Mid-Air dataset necessary for training and testing our architecture, do the following:
> 1. Go on the [download page of the Mid-Air dataset](https://midair.ulg.ac.be/download.html)
> 2. Select the "Left RGB", "Semantic seg." and "Stereo Disparity" image types
> 3. Move to the end of the page and press "Get download links"

When you have the file, execute this script to download and extract the dataset:
```shell
bash  scripts/0a-get_midair.sh path/to/desired/dataset/location path/to/download_config.txt
```

Apply the semantic classes mapping on MidAir by running the following script:
```shell
python scripts/data_class_mapping.py
```

#### Aeroscapes [[2](#ref_2)]

## Reproducing paper results

### Training from scratch
To train on MidAir:
```shell
bash  scripts/1a-train-midair.sh path/to/desired/weights/location
```

To train on Aeroscapes:
```shell
bash  scripts/1a-train-aeroscapes.sh path/to/desired/weights/location
```

### Evaluation and Pretrained weights
The pre-trained weights can be downloaded from here and extracted:
[weights trained on MidAir](https://drive.google.com/file/d/1C8uqqsv8sXYF15QvdLfw3G2S8JZ5vmQf/view?usp=sharing)

[weights trained on Aeroscapes](https://drive.google.com/file/d/1C8uqqsv8sXYF15QvdLfw3G2S8JZ5vmQf/view?usp=sharing)

For evaluation:
```shell
bash  scripts/2-evaluate.sh dataset path/to/weights/location
```

where `dataset` can be `midair` or `aeroscapes`
### Other operations

### Processing outputs

## Prediction on your own images

## Baseline methods performance reproduction
