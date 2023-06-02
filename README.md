<img src = "./images/LiVaS_Logo.png" width="160">  

# Liver Vaculature Segmentation (LiVaS) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7989974.svg)](https://doi.org/10.5281/zenodo.7989974)  

## Introduction

Automated liver segmentation methods often struggle to accurately segment the internal structures of the liver due to its complex anatomy and variable vascular spatial orientations. Manual segmentation and labeling of liver vascularity is time-consuming and requires expert input. The development of automated methods for liver vascular anatomy segmentation has been limited due to the difficulty and cost of producing large annotated datasets.

LiVaS is a semi-automated pipeline and user interface that allows for the quick segmentation and labeling of 3D liver vasculature from multiphase MR images.

## Prerequisites

Prior to using LiVaS software, ensure that the following are true for each case:
- Pre-contrast, arterial, portal venous and delayed phase images are present for the case (Figure 1, 1st row)
- The liver outer contour is segmented on the pre- and post-contrast MR images (Figure 1, 2nd row)
- Post-contrast liver masks (populated with liver signal intensities) are registered to the pre-contrast liver mask (Figure 1, 3rd row)
- Each pre- and post- contrast series are all located in their own directory within a parent case directory (*/path/to/case/directory* in further text). Here is a conceptual diagram of a directory structure:
```mermaid
graph TD

/path/to/case/directory --> CaseID_PRE_PHASE_DIR
CaseID_PRE_PHASE_DIR --> 1.dcm
CaseID_PRE_PHASE_DIR --> 2.dcm
CaseID_PRE_PHASE_DIR --> 3.dcm
CaseID_PRE_PHASE_DIR --> ...
/path/to/case/directory --> CaseID_ARTERIAL_PHASE_DIR
/path/to/case/directory --> CaseID_PHASE_DIR
/path/to/case/directory --> CaseID_DELAYED_PHASE_DIR
```
- Each contrast series direcotry is prefixed with unique case (subject) identifier separated by the underscore *'_'* from the rest of the directory name. Here is an example case subdirectory structure where patient ID is 117370:
```
117370_1219655612_MR_2016-09-17_102928_._S.10.ARTERIAL.MASKED.DEF.ALIGNED_n120__00000
117370_1219655612_MR_2016-09-17_102928_._S.10.DELAYED.1.MASKED.DEF.ALIGNED_n120__00000
117370_1219655612_MR_2016-09-17_102928_._S.10.PRE.MASKED_n120__00000
117370_1219655612_MR_2016-09-17_102928_._S.10.PVP.MASKED.DEF.ALIGNED_n120__00000
```

![fig1](./images/Figure1_phase_images_case106428.png)  
*Figure 1. Liver segmentation and registration. First row, left to right: pre-contrast, arterial phase, portal venous phase and delayed phase MR images. Second row: phase MR images after liver outer contour segmentation. Third row: phase MR images after post-contrast liver masks are registered to pre-contrast liver masks.*

### Python modules requirements:
- python 3.8.5
- numpy 1.19.2
- pydicom 2.2.2
- scipy 1.5.2
- SimpleITK 2.0.2
- faiss 1.6.5
- matplotlib 3.3.2

## Installation

### Setting up the LiVaS conda environment:
```bash
conda create --name LiVaS # create the LiVaS environment
conda env list # list all of the conda environments
conda activate LiVaS # activate the new environment
conda install python=3.8.5 # install python 3.8.5
python --version # confirm that the python version is 3.8.5
conda install numpy=1.19.2 # install numpy
conda install matplotlib=3.3.2
conda install scipy=1.5.2
pip install pydicom==2.2.2
pip install simpleitk==2.0.2
conda install -c conda-forge faiss=1.6.5
conda list # list all the packages in the environment
```
### Clone the LiVaS GitHub repository:
Crete a project directory on your computer (*/path/to/project/directory* in further text) and clone the project repository:
```bash
mkdir /path/to/project/directory # create project directory if not already present
cd /path/to/project/directory
git clone https://github.com/mladenze/LiVaS.git
```

## Usage
The LiVaS usage framework consists of two steps:
1. Voxel signal intensity trajectories are clustered into a preestablished number of groups and saved to disk.
2. Using a customized user interface (UI): a) select voxel clusters that accurately segment the liver veins, b) label the selected clusters as portal veins (PV) or hepatic veins (HV), c) save the labels to disk.
 
### 1) Voxel clustering  
1. Navigate to local project repository:
```bash
cd /path/to/project/directory/LiVaS/
```
2. An example python code for voxel clustering:
```python
import sys
sys.path.append('/path/to/project/directory/LiVaS/')
from LiVaS_helper import *

#-------------------------------------------------------------------------
# List containing search kewords to be used in sorting phase series.
# The order of search keys in the list should reflect the AP timing,
#  and should contain substrings of the relevant phase-directories' names.
sort_order_key_list = ["PRE","ARTERIAL", "PVP", "DELAYED"]

# directory where phase dicom subdirectories are located
dicom_dir = '/path/to/case/directory'

# run clustering for the case
voxel_clustering_pipeline( dicom_dir, sort_order_key_list )
```

### 2) User interface (UI) - Cluster selection and vascular labeling  
The LiVaS UI allows users to quickly browse through precomputed cluster configurations (see *Voxel clustering* above) and match cluster labels to corresponding vessel groups.
The UI can be launched with the following shell command:
```bash
cd /path/to/project/directory/LiVaS/
python LiVaS_UI.py
```
The main UI window presents liver images and clustering outputs as three axial slices that can be scrolled through (Figure 2).  Once the user selects the number of clusters that best segment the liver vasculature, vessel groups are labeled using default keys for Portal Veins and Hepatic Veins, after which segmentation and labels are saved to disk.

![fig2](./images/LiVaS_UI_3.png)  
*Figure 2. LiVaS User Interface (UI) window with 3 arterial axial slices: reference image (left), reference image with overlaid candidate cluster (center), and reference image with overlaid selected and labeled anatomy (right).*

### Output
LiVaS voxel clustering, as well as cluster selection and labeling, generate the following output files:  
- */path/to/case/directory*  
	- *phase_array.npz* (4D numpy array of the original images)  
	- *phase_corrected_arr.npz* (4D image array after bias field correction)  
	- *clusters.npz* (voxel clusters array )  
	- *labels.npz* (3D labels array)  

To use original phase image array along with vasculature labels array, the relevant files must be first loaded from disk:
```python
import numpy as np

def load_NPZs(dicom_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(dicom_dir + '/phase_array.npz') as data:
        phase_array = data['phase_array']
    with np.load(dicom_dir + '/labels.npz') as data:
        labels = data['labels']
    return (phase_array, labels)

# directory where phase dicom subdirectories are located
dicom_dir = '/path/to/case/directory'
phase_array, labels = load_NPZs(dicom_dir)
```

## Authors
- Mladen Zečević
- Kyle Hasenstab
- Guilherme Moura Cunha

## Cite LiVaS software
> 1. Zečević, Mladen, Hasenstab, Kyle, Cunha, Guilherme Moura. Liver Vaculature Segmentation (LiVaS). Published online May 31, 2023. doi:10.5281/zenodo.7989974

#### BibTeX  
```
@software{zecevic_mladen_2023_7989974,
  author       = {Zečević, Mladen and
                  Hasenstab, Kyle and
                  Cunha, Guilherme Moura},
  title        = {Liver Vaculature Segmentation (LiVaS)},
  month        = may,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1.0},
  doi          = {10.5281/zenodo.7989974},
  url          = {https://doi.org/10.5281/zenodo.7989974}
}
```
