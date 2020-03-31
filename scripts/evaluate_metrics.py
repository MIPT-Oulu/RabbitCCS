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
    pred_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Used in method manuscript for CC segmentation/')
    snapshot = Path('dios-erc-gpu_2019_11_19_15_45_01_hist_validation')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=Path, default='../../../Data/masks')
    parser.add_argument('--image_path', type=Path, default='../../../Data/images')
    parser.add_argument('--prediction_path', type=Path, default='/media/dios/dios2/RabbitSegmentation/Histology/Rabbits/Predictions_CTRL_crop_outoffold/Largest')
    parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/results/evaluation_outoffold')
    parser.add_argument('--eval_name', type=str, default='histo_outoffold')
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--n_labels', type=int, default=2)
    parser.add_argument('--experiment', default='./experiment_config.yml')
    parser.add_argument('--snapshot', type=Path,
                        default=Path('../../../workdir/snapshots/') / snapshot)
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    args = parser.parse_args()

    # Load snapshot configuration
    with open(args.snapshot / 'config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    with open(args.snapshot / 'args.dill', 'rb') as f:
        args_experiment = dill.load(f)

    with open(args.snapshot / 'split_config.dill', 'rb') as f:
        split_config = dill.load(f)

    # Initialize results
    results = {'Sample': [], 'Dice': [], 'IoU': [], 'Similarity': []}

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    #samples = [os.path.basename(x) for x in glob(str(args.mask_path / '*XZ'))]
    samples_mask = os.listdir(str(args.mask_path))
    samples_pred = os.listdir(str(args.prediction_path))
    samples_mask.sort()
    samples_pred.sort()
    for idx, sample in enumerate(samples_mask):
        try:
            sleep(0.5); print(f'==> Processing sample {idx + 1} of {len(samples_mask)}: {sample}')

            # Load image stacks
            data = cv2.imread(str(args.image_path / sample))
            mask = cv2.imread(str(args.mask_path / sample), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(args.prediction_path / samples_pred[idx]), cv2.IMREAD_GRAYSCALE)

            # Crop in case of inconsistency
            crop = min(pred.shape, mask.shape)
            mask = mask[:crop[0], :crop[1]]
            pred = pred[:crop[0], :crop[1]]

            # Evaluate metrics
            conf_matrix = calculate_conf(pred.astype(np.bool), mask.astype(np.bool), args.n_labels)
            dice = calculate_dice(conf_matrix)[1]
            iou = calculate_iou(conf_matrix)[1]
            sim = calculate_volumetric_similarity(conf_matrix)[1]

            print(f'Sample {sample}: dice = {dice}, IoU = {iou}, similarity = {sim}')

            # Update results
            results['Sample'].append(sample)
            results['Dice'].append(dice)
            results['IoU'].append(iou)
            results['Similarity'].append(sim)

        except Exception as e:
            print(f'Sample {sample} failing due to error:\n\n{e}\n!')
            continue

    # Add average value to
    results['Sample'].append('Average values')
    results['Dice'].append(np.average(results['Dice']))
    results['IoU'].append(np.average(results['IoU']))
    results['Similarity'].append(np.average(results['Similarity']))

    # Write to excel
    writer = pd.ExcelWriter(str(args.save_dir / ('metrics_' + str(snapshot) + '_' + args.eval_name)) + '.xlsx')
    df1 = pd.DataFrame(results)
    df1.to_excel(writer, sheet_name='Metrics')
    writer.save()

    print(f'Metrics evaluated in {(time() - start) // 60} minutes, {(time() - start) % 60} seconds.')
