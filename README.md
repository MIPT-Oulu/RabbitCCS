# RabbitCCS
*Calcified Cartilage Segmentation from Histological Sections of Rabbit Knee Joints*

[![DOI](https://zenodo.org/badge/201437633.svg)](https://zenodo.org/badge/latestdoi/201437633)

(c) Santeri Rytky, University of Oulu, 2019-2021

![Analysis pipeline](https://github.com/MIPT-Oulu/RabbitCCS/blob/master/images/Flowchart.PNG)

## Background

This repository is used to create deep learning segmentation models for identifying the calcified cartilage layer
in rabbit histopathological images. 
The method is used for segmentation of 2D histological color images and 3D micro-computed tomography images (slice-by-slice).
For detailed description of the method and example of the results, please refer to the publication: 
[Rytky SJO, et al. Automated analysis of rabbit knee calcified cartilage morphology using micro-computed tomography](https://doi.org/10.1111/joa.13435)

## Prerequisites

- [Anaconda installation](https://docs.anaconda.com/anaconda/install/) 
```
git clone https://github.com/MIPT-Oulu/RabbitCCS.git
cd RabbitCCS
conda env create -f environment.yml
```

## Usage

### Model training

- Create a training dataset: input images in folder `images` and target masks in `masks`. 
For 2D data, just add the images to the corresponding folders, making sure that the image and mask names match.
For 3D data, create a subfolder for each scan (sample name for the folder), and include the slices in the subfolder.

- Set the path name for training data in [session.py](../master/rabbitccs/training/session.py) (`init_experiment()` function)

- Create a configuration file to the `experiments/run` folder. Four example experiments are included. 
All experiments are conducted subsequently during training.

```
conda activate cc_segmentation
python train.py
```

### Inference

For 2D prediction, use `inference_tiles.py`. For 3D data, use `inference_tiles_3d.py`. 
Update the `snap` variable, image path and save directory.

### Thickness analysis

For 2D prediction, use `thickness_analysis_2d.py`. For 3D data, use `thickness_analysis_3d.py`. 
The code will create a thickness map using circle-fitting (2D) or sphere -fitting (3D). 
Note that the 3D thickness analysis is very slow for large volumes. 
In such cases, slice-by slice thickness analysis or external software is recommended.

## License

This software is released under the Creative commons license (CC-BY).

## Citation
If you use the source code in your work, please cite [our paper.](https://doi.org/10.1111/joa.13435)

```
@article {Rytky13453,
title = {Automated analysis of rabbit knee calcified cartilage morphology using micro-computed tomography and deep learning},
author = {Rytky, S.J.O. and Huang, L. and Tanska, P. and Tiulpin, A. and Panfilov, E. and Herzog, W. and Korhonen, R.K. and Saarakkala, S. and Finnil{\"a}, M.A.J.},
journal = {Journal of Anatomy}
volume = {239},
number = {2},
pages = {251-263},
doi = {https://doi.org/10.1111/joa.13435},
year = {2021},
}
```
