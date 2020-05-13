import numpy as np
import pandas as pd
import os
from pathlib import Path
from glob import glob
from copy import deepcopy

if __name__ == '__main__':
    # Set up file paths
    #path = Path('/media/dios/dios2/RabbitSegmentation/Histology/Rabbits/Manual vs auto/thickness_median12_predicted')
    path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Manual vs auto/thickness_median12_manual')
    average = True

    result_path = glob(str(path) + '/*Results*.xlsx')
    if len(result_path) == 1:
        # Load data
        results = pd.read_excel(result_path[0], index_col=0)
    else:
        result_list = []
        for i in range(len(result_path)):
            result_list.append(pd.read_excel(result_path[i], index_col=0))
        results = pd.concat(result_list)
        results.sort_values('Sample')

    result_path = result_path[0]

    samples = results['Sample'].values

    if average:
        # Get unique sample names
        idx = 0
        unique_samples = [i.rsplit('_', 1)[0] for i in samples]
        unique_list = np.zeros(len(unique_samples))
        for i in range(len(unique_samples)):
            if i == 0 or unique_samples[i] == unique_samples[i - 1]:
                unique_list[i] = idx
            else:
                idx += 1
                unique_list[i] = idx

        # New dataframe for the average values
        samples_cut = np.unique(unique_samples)
        results_avg = deepcopy(results.iloc[:len(samples_cut), :])
        results_avg.iloc[:, 0] = samples_cut
        results_avg.iloc[:, 1:] = 0

        # Average values
        for sample in range(len(samples_cut)):
            # Rows corresponding to the sample
            rows = np.where(unique_list == sample)
            # Relevant values
            values = results.iloc[rows].values[:, 1:]
            # Mean
            results_avg.iloc[sample, 1:] = np.mean(values, axis=0)
    else:
        samples_cut = samples
        results_avg = results

    # Sample group by age, anatomical location or OA
    age = np.array([i[0] == '8' for i in samples_cut], dtype=np.uint8)
    results_avg['Age'] = age
    oa = np.array(['_CL_' in i for i in samples_cut], dtype=np.uint8)
    oa += np.array([('2C' in i or '8C' in i) for i in samples_cut], dtype=np.uint8) * 2
    results_avg['OA'] = oa
    locations = ['lateral_femur', 'lateral_groove', 'lateral_plateau',
                 'medial_femur', 'medial_plateau', 'patella']
    loc = np.zeros(len(samples_cut))
    for loc_idx in range(len(locations)):
        loc += np.array([locations[loc_idx] in i for i in samples_cut], dtype=np.uint8) * (loc_idx + 1)
    results_avg['Location'] = loc.astype(np.uint8)
    # Sample ID
    id = np.array([i.rsplit('_M', 1)[1][0] for i in samples_cut]).astype(np.uint8)
    max_id = np.max(id)
    oa[np.where(oa == 1)] = 0
    oa[np.where(oa == 2)] = 1
    id += oa * max_id
    id += age * max_id * 2
    results_avg['Sample_ID'] = id

    # Save results
    savepath = str(path / (os.path.basename(result_path)[:-5] + '_average.xlsx'))
    results_avg.to_excel(savepath)
