import cv2
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import dill
#from torch2trt import torch2trt
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import yaml
from time import sleep, time
from skimage import measure
from tqdm import tqdm
from glob import glob
from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.core.utils import auto_detect_device

from rabbitccs.data.utilities import load, save, print_orthogonal
from rabbitccs.data.visualizations import render_volume

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class InferenceModel(nn.Module):
    def __init__(self, models_list):
        super(InferenceModel, self).__init__()
        self.n_folds = len(models_list)
        modules = {}
        for idx, m in enumerate(models_list):
            modules[f'fold_{idx}'] = m

        self.__dict__['_modules'] = modules

    def forward(self, x):
        res = 0
        for idx in range(self.n_folds):
            fold = self.__dict__['_modules'][f'fold_{idx}']
            #res += torch2trt(fold, [x]).sigmoid()
            res += fold(x).sigmoid()

        return res / self.n_folds


def inference(inference_model, img_full, device='cuda', mean=None, std=None):
    x, y, ch = img_full.shape

    input_x = config['training']['crop_size'][0]
    input_y = config['training']['crop_size'][1]

    # Cut large image into overlapping tiles
    tiler = ImageSlicer(img_full.shape, tile_size=(input_x, input_y),
                        tile_step=(input_x // 2, input_y // 2), weight=args.weight)

    # HCW -> CHW. Optionally, do normalization here
    tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(img_full)]

    # Allocate a CUDA buffer for holding entire mask
    merger = CudaTileMerger(tiler.target_shape, channels=1, weight=tiler.weight)

    # Run predictions for tiles and accumulate them
    for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=args.bs, pin_memory=True):
        # Move tile to GPU
        if mean is not None and std is not None:
            tiles_batch = tiles_batch.float()
            for ch in range(len(mean)):
                tiles_batch[:, ch, :, :] = ((tiles_batch[:, ch, :, :] - mean[ch]) / std[ch])
            tiles_batch = tiles_batch.to(device)
        else:
            tiles_batch = (tiles_batch.float() / 255.).to(device)
        # Predict and move back to CPU
        pred_batch = inference_model(tiles_batch)

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

    torch.cuda.empty_cache()
    gc.collect()

    return merged_mask.squeeze()


def largest_object(input_mask):
    """
    Keeps only the largest connected component of a binary segmentation mask.
    """

    output_mask = np.zeros(input_mask.shape, dtype=np.uint8)

    # Label connected components
    binary_img = input_mask.astype(np.bool)
    blobs = measure.label(binary_img, connectivity=1)

    # Measure area
    proportions = measure.regionprops(blobs)

    if not proportions:
        print('No mask detected! Returning original mask')
        return input_mask

    area = [ele.area for ele in proportions]
    largest_blob_ind = np.argmax(area)
    largest_blob_label = proportions[largest_blob_ind].label

    output_mask[blobs == largest_blob_label] = 255

    return output_mask


if __name__ == "__main__":
    start = time()

    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset_root', type=Path, default='/media/dios/dios2/RabbitSegmentation/µCT/Should_be_resegmented_with_neural_network')
    #parser.add_argument('--dataset_root', type=Path, default='/media/santeri/Transcend1/Full samples/')
    #parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/µCT/Should_be_resegmented_with_neural_network')
    parser.add_argument('--subdir', type=Path, choices=['NN_prediction', ''], default='')
    parser.add_argument('--dataset_root', type=Path, default='../../../Data/µCT/images')
    parser.add_argument('--save_dir', type=Path, default='../../../Data/µCT/predictions')
    #parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/µCT/predictions_databank_12samples/')
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--weight', type=str, choices=['pyramid', 'mean'], default='mean')
    parser.add_argument('--completed', type=int, default=0)
    parser.add_argument('--avg_planes', type=bool, default=False)
    parser.add_argument('--snapshot', type=Path,
                        # default='../../../workdir/snapshots/dios-erc-gpu_2020_02_17_14_08_35_no_XY/')
                        default='../../../workdir/snapshots/dios-erc-gpu_2020_03_19_19_53_42_resampled_corr')
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    args = parser.parse_args()
    subdir = 'NN_prediction'  # 'NN_prediction'
    threshold = 0.8


    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    with open(args.snapshot / 'split_config.dill', 'rb') as f:
        split_config = dill.load(f)
    args.save_dir.mkdir(exist_ok=True)
    (args.save_dir / 'visualizations').mkdir(exist_ok=True)

    # Load models
    models = glob(str(args.snapshot) + '/*fold_[0-9]_*.pth')
    #models = glob(str(args.snapshot) + '/*fold_3_*.pth')
    models.sort()
    #device = auto_detect_device()
    device = 'cuda'  # Use the second GPU for inference

    crop = config['training']['crop_size']
    mean_std_path = args.snapshot.parent / f"mean_std_{crop[0]}x{crop[1]}.pth"
    tmp = torch.load(mean_std_path)
    mean, std = tmp['mean'], tmp['std']

    # List the models
    model_list = []
    for fold in range(len(models)):
        if args_experiment.model_unet and args_experiment.gpus > 1:
            model = nn.DataParallel(smp.Unet(config['model']['backbone'], encoder_weights="imagenet", activation='sigmoid'))
        elif args_experiment.model_unet:
            model = smp.Unet(config['model']['backbone'], encoder_weights="imagenet", activation='sigmoid')
        elif args_experiment.gpus > 1:
            model = nn.DataParallel(EncoderDecoder(**config['model']))
        else:
            model = EncoderDecoder(**config['model'])
        model.load_state_dict(torch.load(models[fold]))
        model_list.append(model)

    model = InferenceModel(model_list).to(device)
    #if torch.cuda.device_count() > 1:  # Multi-GPU
    #    model = nn.DataParallel(model).to(device)
    model.eval()

    threshold = 0.5 if config['training']['log_jaccard'] is False else threshold  # Set probability threshold
    print(f'Found {len(model_list)} models.')

    # Load samples
    # samples = [os.path.basename(x) for x in glob(str(args.dataset_root / '*XZ'))]  # Load with specific name
    samples = os.listdir(args.dataset_root)
    samples.sort()
    # samples = [samples[id] for id in [7, 11]]  # Get intended samples from list

    # Skip the completed samples
    if args.completed > 0:
        samples = samples[args.completed:]
    for idx, sample in enumerate(samples):
        try:
            print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

            # Load image stacks
            data_xz, files = load(str(args.dataset_root / sample), rgb=True)
            data_xz = np.transpose(data_xz, (1, 0, 2, 3))  # X-Z-Y-Ch
            data_yz = np.transpose(data_xz, (0, 2, 1, 3))  # Y-Z-X-Ch
            mask_xz = np.zeros(data_xz.shape)[:, :, :, 0]  # Remove channel dimension
            mask_yz = np.zeros(data_yz.shape)[:, :, :, 0]

            # Loop for image slices
            # 1st orientation
            with torch.no_grad():  # Do not update gradients
                for slice_idx in tqdm(range(data_yz.shape[2]), desc='Running inference, YZ'):
                    mask_yz[:, :, slice_idx] = inference(model, data_yz[:, :, slice_idx, :])
                # 2nd orientation
                if args.avg_planes:
                    for slice_idx in tqdm(range(data_xz.shape[2]), desc='Running inference, XZ'):
                        mask_xz[:, :, slice_idx] = inference(model, data_xz[:, :, slice_idx, :])

            # Average probability maps
            if args.avg_planes:
                #mask_avg = ((mask_xz + np.transpose(mask_yz, (0, 2, 1))) / 2)
                mask_avg = ((mask_yz + np.transpose(mask_xz, (0, 2, 1))) / 2)
                mask_final = mask_avg >= threshold
            else:
                mask_avg = mask_yz
                mask_final = mask_yz >= threshold
            mask_xz = list()
            mask_yz = list()

            # Convert to original orientation
            #mask_final = np.transpose(mask_final, (0, 2, 1)).astype('uint8') * 255
            mask_final = mask_final.astype('uint8') * 255
            mask_final = largest_object(mask_final)

            # Save predicted full mask
            if str(args.subdir) != '.':  # Save in original location
                save(str(args.dataset_root / sample / subdir), files, mask_final, dtype=args.dtype)
            else:  # Save in new location
                save(str(args.save_dir / sample), files, mask_final, dtype=args.dtype)
                (args.save_dir.parent / 'probability').mkdir(exist_ok=True)
                save(str(args.save_dir.parent / 'probability' / sample), files, mask_avg * 255, dtype=args.dtype)
            """
            render_volume(data_yz[:, :, :, 0] * mask_final,
                          savepath=str(args.save_dir / 'visualizations' / (sample + '_render' + args.dtype)),
                          white=True, use_outline=False)
            """
            print_orthogonal(data_yz[:, :, :, 0], invert=True, res=3.2, title=None, cbar=True,
                             savepath=str(args.save_dir / 'visualizations' / (sample + '_input.png')),
                             scale_factor=1000)

            print_orthogonal(data_yz[:, :, :, 0], mask=mask_final, invert=True, res=3.2, title=None, cbar=True,
                             savepath=str(args.save_dir / 'visualizations' / (sample + '_prediction.png')),
                             scale_factor=1000)
        except Exception as e:
            print(f'Sample {sample} failed due to error:\n\n {e}\n\n.')
            continue
    dur = time() - start
    print(f'Inference completed in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
