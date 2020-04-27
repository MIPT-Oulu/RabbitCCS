import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
import argparse
import pandas as pd
from time import time, strftime

from rabbitccs.data.utilities import load, print_orthogonal

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()
    pred_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Used in method manuscript for CC segmentation/')
    subdir = 'Prediction_12samples'
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mask_path', type=Path, default='../../../Data/µCT/masks')  # Manual segmentation
    parser.add_argument('--mask_path', type=Path, default=pred_path)
    parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/results')
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    parser.add_argument('--eval_name', type=str, default='µCT_prediction')
    args = parser.parse_args()

    # Initialize results
    results = {'Sample': [], 'Average thickness': []}

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    #samples = os.listdir(str(args.mask_path))
    samples = [os.path.basename(x) for x in glob(str(args.mask_path / '*'))]
    samples.sort()
    for sample in tqdm(samples, 'Analysing thickness'):
        # New sample
        #sample_name = sample.rsplit('_', 1)[0]
        thickness_list = []

        # Load image stacks
        if 'subdir' in locals():
            mask, files_mask = load(str(args.mask_path / sample / subdir), rgb=False, n_jobs=args.n_threads)
        else:
            mask, files_mask = load(str(args.mask_path / sample), rgb=False, n_jobs=args.n_threads)

        for slice in range(mask.shape[0]):
            img = mask[slice, :, :]

            if np.max(img) == 0:
                continue

            # Threshold >= 125
            img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]

            # Calculate thickness
            thickness_list.append(np.sum(img.flatten() / 255) / img.shape[1])

        results['Average thickness'].append(np.mean(thickness_list))
        results['Sample'].append(sample)





        #except Exception as e:
        #    print(f'Sample {sample} failing due to error:\n\n{e}\n!')
        #    continue

    # Write to excel
    writer = pd.ExcelWriter(
        str(args.save_dir / ('thickness_' + strftime(f'%m_%d_%H_%M_%S') + '_' + args.eval_name)) + '.xlsx')
    df1 = pd.DataFrame(results)
    df1.to_excel(writer, sheet_name='Thickness')
    writer.save()

    print(f'Thickness analysed in {(time() - start) // 60} minutes, {(time() - start) % 60} seconds.')
