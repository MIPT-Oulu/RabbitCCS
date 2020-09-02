# RabbitCCS
Calcified Cartilage Segmentation from Histological Sections and µCT images of Rabbit Knee Joints

![Analysis pipeline](https://github.com/MIPT-Oulu/RabbitCCS/blob/master/images/Flowchart.PNG)

## Summary
In this paper, we utilized state-of-the-art deep learning segmentation methods to detect the calcified cartilage (CC) layer in rabbit knees. The method is used for two imaging modalities: three-dimensional micro-computed tomography, as well as two-dimensional histology. Our pipeline allows further analyzing the local thickness maps based on the predicted CC segmentation masks. Our results show that the pipeline allows separating the CC thickness properties between tibia, femur and patella, on both imaging modalities.

## Prerequisites

```conda create --name rabbitccs```

```conda activate rabbitccs```

```pip install -r requirements.txt```

## Usage
Model training is conducted using ```train.py``` in the ```scripts``` directory. The training codes runs a sequence of experiments, based on the experiment configuration files at the ```experiments/run``` directory. Update the configuration files with desired experiment parameters, as well as the general arguments in ```session.py```, ```init_experiment()```.

The out-of-fold inference and metric evaluation can be conducted automatically with the inference parameter set to true.

Inference on new images can be conducted using ```inference_tiles_3d.py``` or ```inference_tiles.py``` in the ```scripts``` (on 3D or 2D data).

Thickness maps can be calculated using ```thickness_analysis_EP.py``` or ```thickness_analysis_EP_2D.py``` in the ```scripts``` (on 3D or 2D data).

## Outputs

- Training and segmentation metrics (training and validation loss, validation Dice score, validation Jaccard index)
- Predicted CC segmentation maps
- Thickness maps and map statistics (mean, std, max...)

## License
This software is distributed under the MIT License. This software and the pretrained models can be used only for research purposes.

## Citation
If you use the software or the source code in your work, please cite [our paper.](https://doi.org/10.1101/2020.08.21.260992)

```
@article {RytkyCCThickness,
		title = {Automated analysis of rabbit knee calcified cartilage morphology using micro-computed tomography and deep learning segmentation},
 author = {Rytky, S.J.O. and Huang L. and Tanska P. and Tiulpin, A. and Panfilov E. and Herzog W. and Korhonen R.K. and Saarakkala S. and Finnil{\"a}, M.A.J.},
 journal = {bioRxiv},
 doi = {10.1101/2020.08.21.260992},
 year = {2020},
}
```
