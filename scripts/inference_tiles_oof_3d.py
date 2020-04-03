import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from time import time
import argparse
import dill
import torch
import yaml
from tqdm import tqdm
from glob import glob

from collagen.core.utils import auto_detect_device
from rabbitccs.data.utilities import load_images as load, save_images as save, bounding_box, largest_object
from rabbitccs.inference.model_components import InferenceModel, load_models
from rabbitccs.inference.pipeline_components import inference, largest_object

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='../../../Data/µCT/images')
    parser.add_argument('--save_dir', type=Path, default='../../../Data/µCT/predictions')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--threshold', type=float, default=0.8)
    # µCT snapshot
    parser.add_argument('--snapshots', type=Path, default='../../../workdir/snapshots/')
    args = parser.parse_args()

    # Snapshots to be evaluated
    snaps = ['dios-erc-gpu_2020_04_02_08_42_39_Unet_resnet18',
              'dios-erc-gpu_2020_04_02_11_18_28_Unet_resnet34',
              'dios-erc-gpu_2020_04_02_14_24_27_FPN_resnet34',
              'dios-erc-gpu_2020_04_03_07_25_01_FPN_resnet18']
    snaps = [args.snapshots / snap for snap in snaps]

    # Iterate through snapshots
    args.save_dir.mkdir(exist_ok=True)
    for snap in snaps:
        save_dir = args.save_dir / str(snap.stem + '_oof')
        save_dir.mkdir(exist_ok=True)

        # Load snapshot configuration
        with open(snap / 'config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        with open(snap / 'args.dill', 'rb') as f:
            args_experiment = dill.load(f)

        with open(snap / 'split_config.dill', 'rb') as f:
            split_config = dill.load(f)

        device = auto_detect_device()

        # Load models
        model_list = load_models(str(snap), config, unet=args_experiment.model_unet, n_gpus=args_experiment.gpus)
        model = InferenceModel(model_list).to(device)
        model.eval()
        print(f'Found {len(model_list)} models.')

        threshold = args.threshold

        # Create directories
        save_dir.mkdir(exist_ok=True)
        try:
            input_x = args_experiment.crop_size[0]
            input_y = args_experiment.crop_size[1]
        except AttributeError:
            input_x = config['training']['crop_size'][0]
            input_y = config['training']['crop_size'][1]

        # Loop for all images
        for fold in range(len(model_list)):
            # List validation images
            validation_files = split_config[f'fold_{fold}']['val'].fname.values

            for file in tqdm(validation_files, desc=f'Running inference for fold {fold}'):

                img_full = cv2.imread(str(file))
                # img_full = np.flip(img_full, axis=0)

                with torch.no_grad():  # Do not update gradients
                    merged_mask = inference(model, args, config, img_full)

                mask_final = (merged_mask >= threshold).astype('uint8') * 255

                # Save largest mask
                largest_mask = largest_object(mask_final)
                # When saving 3D stacks, file structure should be preserved
                (save_dir / file.parent.stem).mkdir(exist_ok=True)
                cv2.imwrite(str(save_dir / file.parent.stem / file.stem) + '.bmp', largest_mask)

                # Free memory
                torch.cuda.empty_cache()
                gc.collect()

        dur = time() - start
        print(f'Inference completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds.')