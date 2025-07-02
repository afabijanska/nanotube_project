# Nanotube Image Annotation and Region Statistics

This repository contains code used to support the analysis of wetting dynamics inside nanotubes in the manuscript:
"Capturing atomic wetting dynamics in real time" by
George T. Tebbutt, Christopher S. Allen, Mohsen Danaie, Anna Fabijańska, Barbara M. Maciejewska, Nicole Grobert

A custom image segmentation model is trained to identify nanotube regions in microscopy images, allowing automated counting of region statistics. The machine learning model is used purely as a tool within the broader physical analysis and is not the main research contribution.

---

## System Requirements

- OS: Windows 10 Pro
- Environment: Python 3.9+ (recommended via Anaconda)
- RAM: ≥ 8 GB
- GPU
  
All required Python packages are listed in `requirements.txt`. Installation in a virtual environment is strongly recommended.

---

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/afabijanska/nanotube_project.git
cd nanotube_project
conda create -n nanotube_env python=3.9
conda activate nanotube_env
pip install -r requirements.txt
```

---

## Demo / Minimal Example

A minimal working dataset is provided in `minimal_data_sample.zip`.  
Unzip it into the root directory of the repository:

```bash
unzip minimal_data_sample.zip
```

This will create the following structure:

```
.
├── train/
│   ├── org/          # original images for training
│   ├── markings/     # manual annotation images, kept for clarity, but do not used by the code
│   └── labels2/      # training labels extracted from markings
│
├── test/
│   ├── org/          # original images for prediction
│   ├── preds/        # will contain predicted masks
│   └── tube_masks/   # binary masks of the tube region
```

---

## Usage Instructions

Run the following scripts in order:

1. **Train the segmentation model**

```bash
python patch_filter_train.py
```

2. **Generate predictions on test images**

```bash
python patch_filter_predict_with_mask.py
```

3. **Count region statistics from predicted masks**

```bash
python count_regions.py
```

The resulting region image labels are saved in `test/preds` directory, similar to statistics saved as csv file in the same directory.

---

## Reproducibility Notes

- The provided minimal dataset allows testing the workflow end-to-end.
- For full-scale experiments used in the manuscript, replace the `train/` and `test/` directories with the full dataset.

---

 ## License

This project is covered under the Apache 2.0 License.  
