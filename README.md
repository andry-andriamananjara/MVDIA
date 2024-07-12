# Image-Level Micro-Gesture Classification via Visual-text Contrastive Learning

## Overview

![Visual-Text Contrastive Learning](visualtextcontrastive.png)

## File Structure

    001690286/
     ├── cecaptioncheckpoints       # checkpoints for a contrastive learning model trained on the cross-entropy loss
     ├── kl2captioncheckpoints      # checkpoints for a contrastive learning model trained on the Jensen-Shannon (JS) divergence loss
     |   ├── best_0.81.pth          # best-performing model checkpoint with an accuracy of 0.81           
     ├── training                   # training folder with subdirectories labeled from 1 to 32, representing different micro-gesture classes
     |   ├── 1                      # directory containing class 1 images 
     |   ├── 2                      # directory containing class 2 images
     |   ...                        ...
     |   ├── 31                     # directory containing class 31 images
     |   ├── 32                     # directory containing class 32 images
     |   ├── train.xlsx             # excel file containing training metadata [filepaths to the images, class (number), video id (video from which image was extracted), frame id and caption (class as text)]
     |   ├── validsubset.csv        # subset of train.xlsx made for illustrative purposes of the testing procedures
     ├── Clip_label.csv             # file containing labels/class names for the data
     ├── environment.yaml           # yaml file containing information about the environment
     ├── README.md                  # markdown file containing information about the project, how to set it up, and how to use it
     ├── report.pdf                 # project report providing explanations on the implementation
     ├── requirements.txt           # file listing all the python dependencies required for this project
     ├── test.py                    # python script used for testing the model
     ├── test_script.sh             # shell script used to execute the `test.py` script
     ├── train_valid.py             # python script used for training and validating the model
     ├── train_valid_script.sh      # shell script used to execute the `train_valid.py` script
     └── visualtextcontrastive.png  # framework of the model


## Prerequisites

### Install Environment

Python 3.10.9

### Requirements

- opencv-contrib-python
- scikit-learn
- pandas
- numpy
- moviepy
- glob2
- torch 
- torchvision
- transformers
- tdqm
- decord
- pytorchvideo
- openpyxl
- albumentations
- timm
- matplotlib
- tensorboard


## Data 
The dataset of images has been extracted from the [iMiGUE] (https://github.com/linuxsino/iMiGUE) video dataset and comprises 32 distinct classes of micro gestures.
 
## Implementation

Our method is based on OpenAI's model, Contrastive Language-Image Pre-training [(CLIP)] (https://github.com/openai/CLIP). Our implementation closely follows [this] (https://github.com/moein-shariatnia/OpenAI-CLIP) implementation with modifications to the loss function and a few parameters. For more detailed information see [report.pdf](report.pdf)

## Performance

The performance on the validation test is as follows:

#### Image-level Micro-gesture Classification
| top-1 Acc(%) | top-5 Acc(%)                                                   |
| :-------------: |:---------------------------------------------------------: | 
| 81.13%          | 94.33% 

## Testing 
To test the trained models, you can run:
```
# test
## imagepathscsv -> Path to the CSV file containing the test images file paths.
## columnname -> Name of the column in the CSV file containing the paths to the test images.
python test.py --imagepathscsv training/validsubset.csv --columnname Path

```
Notably, `Clip_label.csv` should be in the same directory as `test.py`

# Acknowledgments
Our code is based on the Pytorch implementation of [CLIP](https://github.com/moein-shariatnia/OpenAI-CLIP).
