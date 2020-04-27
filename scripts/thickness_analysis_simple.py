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


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    start = time()


    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=Path, default='../../../Data/masks')
    #parser.add_argument('--mask_path', type=Path, default='/media/dios/databank/Lingwei_Huang/Used histology images and CC masks')
    parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/RabbitSegmentation/results')

    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--save_images', type=bool, default=False)
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.png')
    parser.add_argument('--eval_name', type=str, default='Histology_manual')
    args = parser.parse_args()

    # Initialize results
    results = {'Sample': [], 'Average thickness': []}

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    #samples = os.listdir(str(args.mask_path))
    samples = [os.path.basename(x) for x in glob(str(args.mask_path / '*.png'))]
    samples.sort()
    sample_name = ''; thickness_list = []
    for sample in tqdm(samples, 'Analysing thickness'):
        # New sample in list
        if sample_name != sample.rsplit('_', 1)[0]:
            # Add previous result to list
            if len(thickness_list) != 0:
                results['Average thickness'].append(np.mean(thickness_list))
                results['Sample'].append(sample_name)
            # New sample
            sample_name = sample.rsplit('_', 1)[0]
            thickness_list = []

        # Load image
        img = cv2.imread(str(args.mask_path / sample), cv2.IMREAD_GRAYSCALE)

        if args.save_images:
            img = img[2:-2, 2:-2]
            cv2.imwrite(str(args.save_dir / sample)[:-4] + args.dtype, img)
        if args.plot:
            plt.imshow(img); plt.title('Loaded image'); plt.show()
        # Threshold >= 125
        img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]

        # Calculate thickness
        thickness_list.append(np.sum(img.flatten() / 255) / img.shape[1])

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
