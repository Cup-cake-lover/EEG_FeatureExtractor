# Epilepsy Detection using EEG Data

This repository contains code for detecting epileptic events using EEG data from the Bonn dataset. The project includes feature extraction (both statistical and chaotic) and classifier evaluation using various machine learning models.



## Table of Contents
- [Directory Structure](#project-structure)
- [EpilepsyFeatureExtractor Class](#epilepsyfeatureextractor-class)
- [Classifier Evaluation](#classifier-evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [References](#references)

## Directory Structure

- `BonnData/` - Main directory containing subdirectories and scripts.
  - `A/` - Subdirectory containing dataset A.
  - `B/` - Subdirectory containing dataset B.
  - `D/` - Subdirectory containing dataset D.
  - `E/` - Subdirectory containing dataset E.
- `EpilepsyFeatureExtractor.py` - Script for extracting features from the epilepsy datasets.
- `classifier_evaluation.py` - Script for evaluating classifiers on the extracted features.
- `README.md` - This file.


## EpilepsyFeatureExtractor Class

### Description
The `EpilepsyFeatureExtractor` class is designed to load EEG data, perform denoising, and extract statistical and chaotic features.

### Methods
- `__init__()`: Initializes the class with paths and data structures.
- `load_data()`: Loads and denoises the EEG data.
- `denoise(arr)`: Denoises the data using the TQWT method.
- `select_samples()`: Selects and reshapes samples for feature extraction.
- `statistical_feature_extractor()`: Extracts statistical features using mean, standard deviation, kurtosis, and skewness.
- `chaotic_feature_extractor()`: Extracts chaotic features including Lyapunov Exponent, Hurst Exponent, Sample Entropy, and DFA.
- `plot_stat_features()`: Plots the extracted statistical features.
- `plot_chaotic_features()`: Plots the extracted chaotic features.
- `plot_2d_features()`: Plots 2D feature comparisons for chaotic features.

### Usage

from EpilepsyFeatureExtractor import EpilepsyFeatureExtractor

- extractor = EpilepsyFeatureExtractor()

(Load and preprocess the data)
- extractor.load_data()

(Extract and plot statistical features)
- extractor.plot_stat_features()

(Extract and plot chaotic features)
- extractor.plot_chaotic_features()

(Plot 2D feature comparisons for chaotic features)
- extractor.plot_2d_features() 



## Classifier Evaluation

### Description
The `classifier_evaluation.py` script evaluates various classifiers (Random Forest, SVM, K-Nearest Neighbors) on both statistical and chaotic features extracted from EEG data. The evaluation is performed using Stratified K-Fold cross-validation to ensure balanced splits across different classes.

### Usage
1. **Run the script**: `python3 classifier_evaluation.py` -- The results are stored in a pandas dataframe.
   

# References

-- The dataset used here is the Bonn epilepsy data --

@article{PhysRevE.64.061907,
  title = {Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state},
  author = {Andrzejak, Ralph G. and Lehnertz, Klaus and Mormann, Florian and Rieke, Christoph and David, Peter and Elger, Christian E.},
  journal = {Phys. Rev. E},
  volume = {64},
  issue = {6},
  pages = {061907},
  numpages = {8},
  year = {2001},
  month = {Nov},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.64.061907},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.64.061907}
}


## The TQWT implementation is adapted from: 

https://github.com/jollyjonson/tqwt_tools.git 


