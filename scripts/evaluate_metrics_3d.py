import cv2
import numpy as np
import os
from pathlib import Path
from glob import glob
import argparse
import dill
import yaml
import pandas as pd
from time import sleep, time

from rabbitccs.data.utilities import load, print_orthogonal
from rabbitccs.data.visualizations import render_volume

from deeppipeline.segmentation.evaluation.metrics import calculate_iou, calculate_dice, \
    calculate_volumetric_similarity, calculate_confusion_matrix_from_arrays as calculate_conf


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()
    base_path = Path('../../../Data/µCT/')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=Path, default=base_path / 'masks')
    parser.add_argument('--image_path', type=Path, default=base_path / 'images')
    parser.add_argument('--prediction_path', type=Path, default=base_path / 'predictions')
    parser.add_argument('--save_dir', type=Path, default=base_path / 'evaluation')
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--n_labels', type=int, default=2)
    args = parser.parse_args()

    # Snapshots to be evaluated
    snaps = os.listdir(str(args.prediction_path))
    snaps.sort()

    # Iterate through snapshots
    args.save_dir.mkdir(exist_ok=True)
    for snap in snaps:

        # Initialize results
        results = {'Sample': [], 'Dice': [], 'IoU': [], 'Similarity': []}

        # Loop for samples
        (args.save_dir / ('visualizations_' + snap)).mkdir(exist_ok=True)
        samples = os.listdir(str(args.mask_path))
        samples.sort()
        for idx, sample in enumerate(samples):

            sleep(0.5)
            print(f'==> Processing sample {idx + 1} of {len(samples)}: {sample}')

            # Load image stacks
            if 'subdir_mask' in locals():
                mask, files_mask = load(str(args.mask_path / sample / subdir_mask), rgb=False, n_jobs=args.n_threads)
            else:
                mask, files_mask = load(str(args.mask_path / sample), axis=(0, 2, 1), rgb=False, n_jobs=args.n_threads)
            if 'subdir' in locals():
                pred, files_pred = load(str(args.prediction_path / sample / subdir), rgb=False, n_jobs=args.n_threads)
            else:
                pred, files_pred = load(str(args.prediction_path / sample), axis=(0, 2, 1), rgb=False, n_jobs=args.n_threads)
            data, files_data = load(str(args.image_path / sample), axis=(0, 2, 1), rgb=False, n_jobs=args.n_threads)

            # Crop in case of inconsistency
            crop = min(pred.shape, mask.shape)
            mask = mask[:crop[0], :crop[1], :crop[2]]
            pred = pred[:crop[0], :crop[1], :crop[2]]

            # Evaluate metrics
            conf_matrix = calculate_conf(pred.astype(np.bool), mask.astype(np.bool), args.n_labels)
            dice = calculate_dice(conf_matrix)[1]
            iou = calculate_iou(conf_matrix)[1]
            sim = calculate_volumetric_similarity(conf_matrix)[1]

            print(f'Sample {sample}: dice = {dice}, IoU = {iou}, similarity = {sim}')

            # Save predicted full mask
            print_orthogonal(data, invert=False, res=3.2, cbar=True,
                             savepath=str(args.save_dir / ('visualizations_' + snap) / (sample + '_input.png')),
                             scale_factor=1500)
            print_orthogonal(data, mask=mask, invert=False, res=3.2, cbar=True,
                             savepath=str(args.save_dir / ('visualizations_' + snap) / (sample + '_reference.png')),
                             scale_factor=1500)
            print_orthogonal(data, mask=pred, invert=False, res=3.2, cbar=True,
                             savepath=str(args.save_dir / ('visualizations_' + snap) / (sample + '_prediction.png')),
                             scale_factor=1500)

            # Update results
            results['Sample'].append(sample)
            results['Dice'].append(dice)
            results['IoU'].append(iou)
            results['Similarity'].append(sim)

        # Add average value to
        results['Sample'].append('Average values')
        results['Dice'].append(np.average(results['Dice']))
        results['IoU'].append(np.average(results['IoU']))
        results['Similarity'].append(np.average(results['Similarity']))

        # Write to excel
        writer = pd.ExcelWriter(str(args.save_dir / ('metrics_' + str(snap)) + '.xlsx'))
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name='Metrics')
        writer.save()

        print(f'Metrics evaluated in {(time() - start) // 60} minutes, {(time() - start) % 60} seconds.')
