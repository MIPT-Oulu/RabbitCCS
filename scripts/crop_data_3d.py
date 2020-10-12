import numpy as np
import matplotlib.pyplot as plt
import os
from components.data.utilities import load_images as load, save_images as save, bounding_box
from tqdm import tqdm
import cv2
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=pathlib.Path,
                    default='/media/dios/dios2/RabbitSegmentation/µCT/images_test')
parser.add_argument('--mask_dir', type=pathlib.Path,
                    default='/media/dios/dios2/RabbitSegmentation/µCT/predictions_best_fold/')
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--saved', type=bool, default=True)
parser.add_argument('--plot', type=bool, default=True)
parser.add_argument('--largest', type=bool, default=False)
args = parser.parse_args()

im_paths = os.listdir(args.dataset_root)
mask_paths = os.listdir(args.mask_dir)

for dataset in range(len(im_paths)):

    # Load and set paths
    i_path = args.dataset_root / im_paths[dataset]
    m_path = args.mask_dir / mask_paths[dataset] / 'Largest'
    im_files, data = load(str(i_path), rgb=True, uCT=True)
    mask_files, mask = load(str(m_path), rgb=False, uCT=True)

    # Get bounding box for masks and crop data + mask
    contours = []
    removed = 0
    for sample in tqdm(range(len(mask)), 'Getting bounding boxes'):
        try:
            bbox, contour = bounding_box(mask[sample - removed], largest=args.largest)
        except ValueError:  # Empty mask
            data.pop(sample)
            mask.pop(sample)
            removed += 1
            continue
        # Add contour to list
        contours.append(contour)
        if args.crop:
            data[sample - removed] = data[sample - removed][:, bbox[0]:bbox[0] + bbox[2] - 1, :]
            mask[sample - removed] = mask[sample - removed][:, bbox[0]:bbox[0] + bbox[2] - 1]

    # Show example image

    if args.plot:
        data_ref = data.copy()
        for sample_id in tqdm(range(len(data)), 'Plotting'):
            img = data_ref[sample_id]
            if args.largest:
                cv2.drawContours(img, [contours[sample_id]], 0, (0, 0, 255), 2)
            else:
                cv2.drawContours(img, contours[sample_id], -1, (0, 0, 255), 2)


    # Save images
    if args.saved:
        save_im = '/media/dios/dios2/HistologySegmentation/Images_cropped2'
        save_im_ref = args.mask_dir / mask_paths[dataset] / 'Contour'
        save_mask = '/media/dios/dios2/HistologySegmentation/Masks_cropped2'

        if args.plot:
            save(str(save_im_ref), im_files, data_ref, crop=args.crop)
        if args.crop:
            save(save_mask, mask_files, mask, crop=args.crop)
            save(save_im, im_files, data, crop=args.crop)