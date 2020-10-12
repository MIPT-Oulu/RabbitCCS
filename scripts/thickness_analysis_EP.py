import os
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from time import time, strftime
import pandas as pd

import numpy as np
import scipy.ndimage as ndi
import argparse

from components.data.utilities import load, save, print_orthogonal
from components.inference.thickness_analysis import _local_thickness


if __name__ == '__main__':
    # ******************************** 3D case ************************************
    start = time()
    # base_path = Path('../../../Data/µCT')
    base_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Full dataset')
    filter_size = 12
    parser = argparse.ArgumentParser()
    parser.add_argument('--masks', type=Path, default=base_path / 'Predictions_FPN_Resnet18_OA')
    parser.add_argument('--th_maps', type=Path, default=base_path / f'thickness_median{filter_size}_missing')
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--save_h5', type=bool, default=True)
    parser.add_argument('--batch_id', type=int, default=None)
    parser.add_argument('--resolution', type=tuple, default=(3.2, 3.2, 3.2))  # in µm
    #parser.add_argument('--resolution', type=tuple, default=(12.8, 12.8, 12.8))  # in µm
    parser.add_argument('--mode', type=str,
                        choices=['med2d_dist3d_lth3d', 'stacked_2d', 'med2d_dist2d_lth3d'],
                        default='med2d_dist3d_lth3d')
    parser.add_argument('--max_th', type=float, default=None)  # in µm
    parser.add_argument('--median', type=int, default=filter_size)
    parser.add_argument('--completed', type=int, default=0)

    args = parser.parse_args()

    # Sample list
    samples = os.listdir(args.masks)
    samples.sort()
    if args.batch_id is not None:
        samples = [samples[args.batch_id]]
    elif args.completed > 0:
        samples = samples[args.completed:]

    # Save paths
    args.th_maps.mkdir(exist_ok=True)
    (args.th_maps / 'visualization').mkdir(exist_ok=True)
    (args.th_maps / 'h5').mkdir(exist_ok=True)

    results = {'Sample': [], 'Mean thickness': [], 'Median thickness': [], 'Thickness STD': [],'Maximum thickness': []}
    t = strftime(f'%Y_%m_%d_%H_%M')

    # Loop for samples
    for sample in samples:
        time_sample = time()
        print(f'Processing sample {sample}')

        # Load prediction
        pred, files = load(str(args.masks / sample), axis=(1, 2, 0,))

        # Downscale
        #pred = (ndi.zoom(pred, 0.25) > 126).astype(np.bool)

        if args.plot:
            print_orthogonal(pred, savepath=str(args.th_maps / 'visualization' / (sample + '_pred.png')))

        # Median filter
        pred = ndi.median_filter(pred, size=args.median)
        if args.plot:
            print_orthogonal(pred, savepath=str(args.th_maps / 'visualization' / (sample + '_median.png')))

        # Thickness analysis
        # Create array of correct size
        th_map = _local_thickness(pred, mode=args.mode, spacing_mm=args.resolution, stack_axis=1,
                                  thickness_max_mm=args.max_th)
        if args.plot:
            print_orthogonal(th_map, savepath=str(args.th_maps / 'visualization' / (sample + '_th_map.png')),
                             cmap='hot')

        plt.hist(x=th_map[np.nonzero(th_map)].flatten(), bins='auto')
        plt.show()


        # Save resulting thickness map with bmp and h5py
        save(str(args.th_maps / sample), sample, th_map, dtype='.bmp')

        # H5PY save
        if args.save_h5:
            savepath = args.th_maps / 'h5' / (sample + '.h5')
            h5 = h5py.File(str(savepath), 'w')
            h5.create_dataset('data', data=th_map)
            h5.close()

        # Update results
        th_map = th_map[np.nonzero(th_map)].flatten()
        results['Sample'].append(sample)
        results['Mean thickness'].append(np.mean(th_map))
        results['Median thickness'].append(np.median(th_map))
        results['Maximum thickness'].append(np.max(th_map))
        results['Thickness STD'].append(np.std(th_map))

        # Save results to excel
        writer = pd.ExcelWriter(str(args.th_maps / ('Results_' + t)) + '.xlsx')
        df1 = pd.DataFrame(results)
        df1.to_excel(writer, sheet_name='Thickness analysis')
        writer.save()

        dur_sample = time() - time_sample
        print(f'Sample processed in {(dur_sample % 3600) // 60} minutes, {dur_sample % 60} seconds.')

    dur = time() - start
    completed = strftime(f'%Y_%m_%d_%H_%M')
    print(f'Analysis completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds at time {completed}.')