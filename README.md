# Nanotube Image Annotation and Region Statistics

This repository contains code used to support the analysis of wetting dynamics inside nanotubes in the manuscript:
"Capturing atomic wetting dynamics in real time" by
George T. Tebbutt, Christopher S. Allen, Mohsen Danaie, Anna Fabijańska, Barbara M. Maciejewska, Nicole Grobert

A custom image segmentation model is trained to identify annotated regions in microscopy images, allowing automated counting of region statistics. The machine learning model is used purely as a tool within the broader physical analysis and is not the main research contribution.

---

## 🖥️ System Requirements

- OS: Windows 10 Pro
- Environment: Python 3.10+ (recommended via Anaconda)
- RAM: ≥ 8 GB
- GPU: optional (for faster training/prediction)

All required Python packages are listed in `requirements.txt`. Installation in a virtual environment is strongly recommended.

---

## ⚙️ Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/afabijanska/nanotube_project.git
cd nanotube_project
conda create -n nanotube_env python=3.10
conda activate nanotube_env
pip install -r requirements.txt


# nanotube_project
Source code of the ML model associated with submission "Capturing atomic wetting dynamics in real time" by
George T. Tebbutt, Christopher S. Allen, Mohsen Danaie, Anna Fabijańska, Barbara M. Maciejewska, Nicole Grobert

To run the code:
1. Create directory train containing subdirectories: 
- org -> containing original images
- markings -> containing images with manual annotations
- labels2 -> containing labels from training extracted from manual markings

2. Create directory test containing subdidectories
- org -> containing images for prediction
- preds -> for predicted images
- tube_masks -> containing binary masks of tube regions 

2. Run patch_filter_train.py to train the model
3. Run patch_filter_predict_with_mask.py to get the predictions
4. Run count_regions to get the region stats

A minimal data sample required to run the code is suplemented. Please unzip minimal_data_sample.zip to the directory containing source code.

The code was developed and executed on a Windows 10 Pro system within an Anaconda environment. All software dependencies are specified in the accompanying requirements.txt file.
