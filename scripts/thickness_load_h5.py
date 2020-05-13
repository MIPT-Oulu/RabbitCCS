import os
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from time import time, strftime
from glob import glob
import pandas as pd

import numpy as np
import scipy.ndimage as ndi
import argparse

from rabbitccs.data.utilities import load, save, print_orthogonal
from rabbitccs.inference.thickness_analysis import _local_thickness


if __name__ == '__main__':
    # ******************************** 3D case ************************************
    start = time()
    base_path = Path('../../../Data/µCT')
    #base_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Full dataset')
    filter_size = 12
    parser = argparse.ArgumentParser()
    parser.add_argument('--th_maps', type=Path, default=base_path / f'Thickness_cluster/h5')
    parser.add_argument('--visual', type=Path, default=base_path / f'Thickness_cluster/visualization')
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
    samples = glob(str(args.th_maps) + '/*.h5')
    samples.sort()

    # Save paths
    results = {'Sample': [], 'Mean thickness': [], 'Median thickness': [], 'Thickness STD': [],'Maximum thickness': []}
    t = strftime(f'%Y_%m_%d_%H_%M')

    # Loop for samples
    for sample in samples:
        time_sample = time()
        print(f'Processing sample {sample}')
        try:
            # Load prediction
            h5 = h5py.File(sample, 'r')
            th_map = h5['data'][:]
            h5.close()
            sample = os.path.basename(sample)[:-3]

            th_map = th_map[np.nonzero(th_map)].flatten()

            # Plot histogram
            plt.hist(x=th_map, bins=30, density=True)
            plt.xlabel('Thickness (µm)')
            plt.savefig(str(args.visual / (sample + '_histogram.png')), dpi=600)
            plt.show()

            # Update results
            results['Sample'].append(sample)
            results['Mean thickness'].append(np.mean(th_map))
            results['Median thickness'].append(np.median(th_map))
            results['Maximum thickness'].append(np.max(th_map))
            results['Thickness STD'].append(np.std(th_map))
        except (OSError, KeyError):
            print(f'Sample {sample} failing. Skipping to next one.')
            continue
        dur_sample = time() - time_sample
        print(f'Sample processed in {(dur_sample % 3600) // 60} minutes, {dur_sample % 60} seconds.')


    # Create dataframe
    results = pd.DataFrame(results)
    samples_cut = results['Sample'].values

    # Sample group by age, anatomical location or OA
    age = np.array([i[0] == '8' for i in samples_cut], dtype=np.uint8)
    results['Age'] = age
    oa = np.array(['_CL_' in i for i in samples_cut], dtype=np.uint8)
    oa += np.array([('2C' in i or '8C' in i) for i in samples_cut], dtype=np.uint8) * 2
    results['OA'] = oa
    locations = ['lateral_femur', 'lateral_groove', 'lateral_plateau',
                 'medial_femur', 'medial_plateau', 'patella']
    loc = np.zeros(len(samples_cut))
    for loc_idx in range(len(locations)):
        loc += np.array([locations[loc_idx] in i for i in samples_cut], dtype=np.uint8) * (loc_idx + 1)
    results['Location'] = loc.astype(np.uint8)
    # Sample ID
    id = np.array([i.rsplit('_M', 1)[1][0] for i in samples_cut]).astype(np.uint8)
    max_id = np.max(id)
    oa[np.where(oa == 1)] = 0
    oa[np.where(oa == 2)] = 1
    id += oa * max_id
    id += age * max_id * 2
    results['Sample_ID'] = id

    # Save results to excel
    writer = pd.ExcelWriter(str(args.th_maps / ('Results_' + t)) + '.xlsx')
    results.to_excel(writer, sheet_name='Thickness analysis')
    writer.save()

    dur = time() - start
    completed = strftime(f'%Y_%m_%d_%H_%M')
    print(f'Analysis completed in {(dur % 3600) // 60} minutes, {dur % 60} seconds at time {completed}.')