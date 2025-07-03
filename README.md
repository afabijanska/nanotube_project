# Nanotube Image Segmentation and Region Tracking: CNN-Based Patch Classifier for in situ HRTEM datasets

This repository contains code used to support the analysis of wetting dynamics inside nanotubes in the manuscript:
"Capturing atomic wetting dynamics in real time" by
George T. Tebbutt, Christopher S. Allen, Mohsen Danaie, Anna Fabijańska, Barbara M. Maciejewska, Nicole Grobert

Purpose: 

This repository provides a patch-based convolutional neural network (CNN) model developed to process high-resolution transmission electron microscopy (HRTEM) datasets that capture *in situ* atomic-scale processes across multiple frames.

The CNN architecture is specifically designed to preserve native atomic resolution, using a 64x64 patch that rasters across micrographs of up to 2500x5000 pixels in size. Each central pixel is classified based on the local atomic texture surrounding it, allowing the model to resolve subtle structural variations without image downsampling or interpolation.

Once trained, the model enables automated, large-scale identification of structural regions based on their atomic texture. The model can distinguish between amorphous domains, crystalline regions (including intermediate oxides phases), liquid phases, long-range ordered features (e.g., carbon nanotube walls), and the background TEM vacuum.

By applying the model across large *in situ* datasets, allows for the extraction of statistical trends within and across samples, including tracking phase transformations, quantifying wetting behaviour, and analysing nanowire growth dynamics at atomic resolution as presented here.

---

## System Requirements

- OS: Windows 10 Pro
- Environment: Python 3.9+ (recommended via Anaconda)
- RAM: ≥ 64 GB (tested on 128 MB)
- GPU (tested on NVIDIA Quadro RTX 6000)
  
All required Python packages are listed in `requirements.txt`. Installation in a virtual environment is strongly recommended.

The code was tested with GPU support. CPU-only runs are not recommended and have not been validated.

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

Typical installation time on a standard computer with a stable internet connection is several minutes.

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

Expected run time of demo:
- Training time: ~30 minutes
- Prediction time: ~5 minutes per image

Actual times will vary with hardware used, dataset size and image resolution.

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

## Using a Pre-Trained Model

If you want to skip training and directly test the pipeline using a pre-trained model:

1. **Copy the provided model weights**  
   Place the file `trained_model_weights.h5` into the **root directory** of the repository.

2. **Update the prediction script**
   Open `patch_filter_predict_with_mask.py` and modify line 87 as follows:

   ```python
   best_weights_file = 'trained_model_weights.h5'
   
3. **Run prediction as usual**

  ```bash
  python patch_filter_predict_with_mask.py
  ```
---

## Reproducibility Notes

- The provided minimal dataset allows testing the workflow end-to-end.
- For full-scale experiments used in the manuscript, replace the `train/` and `test/` directories with the full dataset.

---

 ## License

This project is covered under the Apache 2.0 License.  
