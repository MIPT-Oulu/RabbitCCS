import os
import numpy as np
from pathlib import Path
from rabbitccs.training.session import init_experiment
from rabbitccs.data.utilities import load, save, print_orthogonal
from scipy.ndimage import zoom

if __name__ == "__main__":
    # Initialize experiment
    args, config, device, snapshots_dir, snapshot_name = init_experiment()
    #base_path = args.data_location / 'µCT'
    base_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Curated training data')
    #base_path = Path('/media/dios/dios2/RabbitSegmentation/Data_from_Lingwei/New manual CC segmentation of 4 knees')
    #base_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Full dataset')
    masks_loc = base_path / 'masks_full'
    images_loc = base_path / 'CC_window_resampled'
    #images_loc = base_path / 'CC_window_OA'
    #images_loc = base_path / 'CC_window_rec'
    #images_loc = base_path / 'CC_window_OA_resampled'

    images_save = base_path / 'images_resampled'
    #images_save = base_path / 'images'
    #images_save = base_path / 'CC_window_resampled'
    #images_save = base_path / 'CC_window_OA_resampled'
    masks_save = base_path / 'masks_resampled'
    #masks_save = base_path / 'masks'

    subdir = 'Manual segmentation'
    images_save.mkdir(exist_ok=True)
    masks_save.mkdir(exist_ok=True)

    resample = False
    factor = 55
    n_slices = 10

    # Resample large number of slices
    samples = os.listdir(images_loc)
    samples = [name for name in samples if os.path.isdir(os.path.join(images_loc, name))]
    samples.sort()
    for sample in samples:
        try:
            if resample:  # Resample slices
                im_path = images_loc / sample
                mask_path = masks_loc / sample

                data, _ = load(im_path, axis=(1, 2, 0))
                mask, _ = load(mask_path, axis=(1, 2, 0))

                data_resampled = zoom(data, (1, 1, 1 / factor), order=0)  # nearest interpolation
                mask_resampled = zoom(mask, (1, 1, 1 / factor), order=0)  # nearest interpolation
                #print_orthogonal(data_resampled)

                save(str(images_save / sample), sample, data_resampled[:, :, :n_slices], dtype='.bmp')
                save(str(masks_save / sample), sample, mask_resampled[:, :, :n_slices], dtype='.bmp')
            else:  # Move segmented samples to training data
                im_path = str(images_loc / sample)
                mask_path = str(images_loc / sample / subdir)
                files = os.listdir(im_path)
                if subdir in files:
                    data, _ = load(im_path, axis=(1, 2, 0))
                    mask, _ = load(mask_path, axis=(1, 2, 0))

                    save(str(images_save / sample), sample, data, dtype='.bmp')
                    save(str(masks_save / sample), sample, mask, dtype='.bmp')




        except ValueError:
            print(f'Error in sample {sample}')
            continue

