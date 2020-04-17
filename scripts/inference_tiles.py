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
from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.core.utils import auto_detect_device
from rabbitccs.data.transforms import numpy2tens
from rabbitccs.data.utilities import load_images as load, save_images as save, bounding_box
from rabbitccs.inference.pipeline_components import inference, largest_object
from rabbitccs.inference.model_components import InferenceModel, load_models

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=Path, default='/media/dios/dios2/RabbitSegmentation/Histology/Rabbits/Images_CTRL')
    parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/Histology/Rabbits/Predictions_resnet34_UNet')
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--snapshot', type=Path, default='../../../workdir/snapshots/dios-erc-gpu_2020_04_07_15_41_37_2D_resnet34_UNet/')
    args = parser.parse_args()
    threshold = 0.8

    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    with open(args.snapshot / 'split_config.dill', 'rb') as f:
        split_config = dill.load(f)

    # Load models
    unet = config['model']['decoder'].lower() == 'unet'
    device = auto_detect_device()
    model_list = load_models(str(args.snapshot), config, unet=unet, n_gpus=args.gpus)
    model = InferenceModel(model_list).to(device)
    model.eval()

    # Find image files
    files = glob(str(args.dataset_root) + '/*.[tp][in][fg]')
    files.sort()
    args.save_dir.mkdir(exist_ok=True)
    (args.save_dir / 'visualization').mkdir(exist_ok=True)
    # Loop for all images

    input_x = config['training']['crop_size'][0]
    input_y = config['training']['crop_size'][1]
    for file in tqdm(files, desc='Running inference'):

        img_full = cv2.imread(file)

        x, y, ch = img_full.shape
        mask_full = np.zeros((x, y))

        # Cut large image into overlapping tiles
        tiler = ImageSlicer(img_full.shape, tile_size=(input_x, input_y),
                            tile_step=(input_x // 2, input_y // 2), weight=args.weight)

        # HCW -> CHW. Optionally, do normalization here
        tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

        # Allocate a CUDA buffer for holding entire mask
        merger = CudaTileMerger(tiler.target_shape, channels=1, weight=tiler.weight)

        # Loop evaluating inference on every fold
        masks = []

        # Run predictions for tiles and accumulate them
        for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=args.bs, pin_memory=True):
            # Move tile to GPU
            tiles_batch = (tiles_batch.float() / 255.).to(device)
            # Predict and move back to CPU
            pred_batch = model(tiles_batch).detach()

            # Merge on GPU
            merger.integrate_batch(pred_batch, coords_batch)

            # Plot
            if args.plot:
                for i in range(args.bs):
                    if args.bs != 1:
                        plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze()[i, :, :])
                    else:
                        plt.imshow(pred_batch.cpu().detach().numpy().astype('float32').squeeze())
                    plt.show()

        # Normalize accumulated mask and convert back to numpy
        merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype('float32')
        merged_mask = tiler.crop_to_orignal_size(merged_mask)
        # Plot
        if args.plot:
            for i in range(args.bs):
                if args.bs != 1:
                    plt.imshow(merged_mask)
                else:
                    plt.imshow(merged_mask.squeeze())
                plt.show()

        # Average of predictions
        mask_final = (merged_mask >= threshold).astype('uint8') * 255

        # Save largest mask
        largest_mask = largest_object(mask_final)
        cv2.imwrite(str(args.save_dir) + '/' + file.split('/')[-1][:-4] + '.bmp', largest_mask)

        # Save reference images (slows down the script)

        cv2.imwrite(str(args.save_dir / 'visualization' / file.split('/')[-1][:-4]) + '_input.png', img_full)
        m = largest_mask.squeeze()
        fig = plt.figure(); ax = fig.add_subplot(111); ax.set_axis_off()
        ax.imshow(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)); ax.imshow(np.ma.masked_array(m, m == 0), cmap='autumn', alpha=0.4)
        fig.savefig(str(args.save_dir / 'visualization' / file.split('/')[-1][:-4]) + '_reference.png', dpi=800,
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

    dur = time() - start
    print(f'Inference completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds.')