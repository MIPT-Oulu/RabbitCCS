import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
import argparse
import pandas as pd
from time import time
from rabbitccs.data.utilities import load_images as load, save_images as save, bounding_box, print_orthogonal

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    """
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    """
    # Check for errors
    if not(0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError('"lower_bound" must be in the range [0, 65535]')
    elif not(0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError('"upper_bound" must be in the range [0, 65535]')
    elif lower_bound >= upper_bound:
        raise ValueError('"lower_bound" must be smaller than "upper_bound"')

    # Automatic scaling if not given
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)

    # Create lookup that maps 16-bit to 8-bit values
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


if __name__ == "__main__":
    start = time()


    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=Path, default='../../../Data/masks')
    parser.add_argument('--data_path', type=Path, default='/media/dios/dios2/3DHistoData/HMDS_data')
    parser.add_argument('--save_dir', type=Path, default='/media/dios/dios2/3DHistoData/HMDS_scaled')
    parser.add_argument('--k_closing', type=tuple, default=(13, 13))
    parser.add_argument('--k_gauss', type=tuple, default=(9, 9))
    parser.add_argument('--k_median', type=int, default=7)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--dtype', type=str, choices=['.bmp', '.png', '.tif'], default='.bmp')
    args = parser.parse_args()

    # Loop for samples
    args.save_dir.mkdir(exist_ok=True)
    samples = os.listdir(str(args.data_path))
    #samples = [os.path.basename(x) for x in glob(str(args.mask_path / '*.png'))]
    samples.sort()
    for sample in tqdm(samples, 'Smoothing'):
        #try:
        # Load image
        _, data = load(str(args.data_path / sample), uCT=True)
        data_scaled = map_uint16_to_uint8(data, lower_bound=0, upper_bound=40000)
        print_orthogonal(data_scaled)
        img = cv2.imread(str(args.mask_path / sample), cv2.IMREAD_GRAYSCALE)
        if args.plot:
            plt.imshow(img); plt.title('Loaded image'); plt.show()
        # Opening
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=args.k_closing)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel)
        #plt.imshow(img); plt.title('Closing'); plt.show()

        # Gaussian blur
        img = cv2.GaussianBlur(img, ksize=args.k_gauss, sigmaX=0, sigmaY=0)
        #plt.imshow(img); plt.title('Gaussian blur'); plt.show()
        # Median filter (round kernel 7)
        img = cv2.medianBlur(img, ksize=args.k_median)
        #plt.imshow(img); plt.title('Median filter'); plt.show()
        # Threshold >= 125
        img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)[1]
        if args.plot:
            plt.imshow(img); plt.title('Threshold'); plt.show()
        # Save image
        cv2.imwrite(str(args.save_dir / sample), img)
        #except Exception as e:
        #    print(f'Sample {sample} failing due to error:\n\n{e}\n!')
        #    continue

    print(f'Metrics evaluated in {(time() - start) // 60} minutes, {(time() - start) % 60} seconds.')
