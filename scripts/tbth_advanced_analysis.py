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

from components.data.utilities import load, print_orthogonal

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()
    base_path = Path('/media/santeri/Transcend/CC_window_rec')
    subdir = 'TBTH'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=Path, default=base_path)
    parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/results')
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    parser.add_argument('--eval_name', type=str, default='µCT_prediction')
    args = parser.parse_args()

    # Initialize results
    results = {'Sample': [], 'Most frequent thickness value': []}

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    (args.save_dir / 'Visualizations').mkdir(exist_ok=True)
    samples = os.listdir(str(args.mask_path))
    #samples = [os.path.basename(x) for x in glob(str(args.mask_path / '*'))]
    samples.sort()
    for sample in tqdm(samples, 'Analysing thickness'):
        # Sample path
        thickness_path = base_path / sample / subdir

        # Load image stacks
        if thickness_path.exists():
            data, files = load(str(thickness_path), rgb=False, n_jobs=args.n_threads)
        else:
            continue

        # Visualize thickness map
        print_orthogonal(data, savepath=str(args.save_dir / 'Visualizations' / ('CCTh_' + sample + '.png')))

        # Histogram
        data = data.flatten()
        #data = data[data != 0]  # Remove zeros
        max_value = np.max(data)
        hist, bin_edges = np.histogram(data * 3.2, bins=254, range=[1, max_value])  # Exclude background

        plt.hist(data, bins=range(1, max_value))
        plt.title(sample)
        plt.savefig(str(args.save_dir / 'Visualizations' / ('histogram_' + sample + '.png')))

        # Save most the frequent value
        results['Most frequent thickness value'].append(bin_edges[np.argmax(hist)])
        results['Sample'].append(sample)

    # Write to excel
    writer = pd.ExcelWriter(
        str(args.save_dir / ('thickness_histogram_' + strftime(f'%m_%d_%H_%M_%S') + '_' + args.eval_name)) + '.xlsx')
    df1 = pd.DataFrame(results)
    df1.to_excel(writer, sheet_name='Thickness')
    writer.save()

    print(f'Thickness analysed in {(time() - start) // 60} minutes, {(time() - start) % 60} seconds.')
