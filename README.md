### Supplementary software for the manuscript "Fast imaging of 15 intracellular compartments and interactions by deep learning segmentation of super-resolution data". Currently, the package contains the prediction part of the code and the test dataset of two cells.

`LiCAI` is a Python package containing tools for segmenting intracellular structures from the super-resolution ratiometric images.

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

# Installation Guide:

We recommend to use anaconda to install the python environment and dependencies.The typical install time should be less than one hour. After setup the running environment, download and unzip the code. 

# Demonstration

To run demo notebooks:
- cd the code directory in "conda.exe"
- `jupyter notebook`
- copy the url jupyter generates, it looks something like this: `http://127.0.0.1:8888/`, open the link in your browser
- open `Prediction.ipynb` and run
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

# Run your data
The dataset for prediction can be put under the "Data" folder as a subfolder. In the subfolder, two images are required: the intensity image "fluoa.tif" and the spectrum ratio image "ratio.tif". Currently the dataset subfolder should end with "_NZ_NumberOfZSlices". More details on the data training and prediction of custom dataset will be updated upon acception of the manuscript.