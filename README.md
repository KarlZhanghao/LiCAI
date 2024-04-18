### Supplementary software for the manuscript "Fast imaging of 15 intracellular compartments and interactions by deep learning segmentation of super-resolution data". Currently, the package contains the prediction part of the code and the test dataset of two cells.

`LiCAI` is a Python package containing tools for segmenting intracellular structures from the super-resolution ratiometric images.
*This repository is still under construction*

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demonstration](#license)

# System Requirements
## Hardware requirements
`LiCAI` package requires a standard computer. However, a GPU card is prefered to run the deep learning networks.

## Software requirements
This package is implemented with PyTorch. It has been tested with the Anaconda in the Windows 10 OS. LiCAI depends on the following packages.

```
python = 3.8
pytorch = 1.6
torchvision
jupyter
scipy
pandas
tifffile
tqdm
```

# Installation Guide

We recommend to use anaconda to install the python environment and dependencies.The typical install time should be less than one hour. After setup the running environment, download and unzip the code. 

# Model and data download

|Data   |url	|des folder |
|:----:|:----:|:----:| 
|pretrained models |[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7724778.svg)](https://doi.org/10.5281/zenodo.7724778)  |./Models   |
|foreground detection model |[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10989849.svg)](https://doi.org/10.5281/zenodo.10989849). |./Models   |
|test examples for prediction|[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7724778.svg)](https://doi.org/10.5281/zenodo.7724778)|./Data/prediction|
|illastic model|[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7653696.svg)](https://doi.org/10.5281/zenodo.7653696)|./Illastic|
|lipid droplet training dataset|[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10979607.svg)](https://doi.org/10.5281/zenodo.10979607)|./Data/LD|
|mitochondria training dataset|[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7724798.svg)](https://doi.org/10.5281/zenodo.7724798)|./Data/MITO|
|Golgi training dataset|[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10979124.svg)](https://doi.org/10.5281/zenodo.10979124)|./Data/GOLGI|
|ER training dataset|[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10978822.svg)](https://doi.org/10.5281/zenodo.10978822)|./Data/ER|
|lysosome training dataset|[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10978939.svg)](https://doi.org/10.5281/zenodo.10978939)|./Data/LYSO|


For example, make the Models folder and put the pretrained .pth files into the directory.

# Demonstration of inference

To run demo notebooks:
- cd the code directory in "conda.exe"
- `jupyter notebook`
- copy the url jupyter generates, it looks something like this: `http://127.0.0.1:8888/`, open the link in your browser
- open `Demo1_prediction.ipynb` and run
- The datasets in the "Data" folder will be processed and the predicted masks are within the same directory
- The prediction of a single dataset is within seconds for our GTX 2080 GPU and takes several minutes for the CPU. There are two datasets in the "Data" folder for demonstration

The network outputs include:
- pred_GOLGIm: the segmented binary mask of Golgi apparatus
- pred_LDm: the segmented binary mask of lipid droplets
- pred_MITOm: the segmented binary mask of mitochondria
- pred_LYSOm: the segmented binary mask of lysosomes
- pred_ERm: the segmented binary mask of ER, nuclear membrane, and nuclear reticulum
- pred_NMNRm_0: the segmented binary mask of ER only
- pred_NMNRm_1: the segmented binary mask of nuclear membrane only
- pred_NMNRm_2: the segmented binary mask of nuclear reticulum only
- pred_PMFPm_1: the segmented binary mask of plasma membrane
- pred_PMFPm_2: the segmented binary mask of filopodia
- pred_PEROm: the segmented binary mask of peroxisomes
- pred_EEm: the segmented binary mask of early endosomes
- pred_LEm: the segmented binary mask of late endosomes
- pred_VOLm_0: the segmented binary mask of nucleus
- pred_VOLm_1: the segmented binary mask of cytosol
- pred_VOLm_2: the segmented binary mask of extracellular space

# Model training
- `Demo2_model_training.ipynb` for model training
- `Demo3_transfer_learning.ipynb` for model training
