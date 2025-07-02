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
