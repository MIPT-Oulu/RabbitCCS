import numpy as np
import matplotlib.pyplot as plt
from rabbitccs.data.utilities import load_images as load, save_images as save, bounding_box
from tqdm import tqdm
import cv2
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=pathlib.Path,
                    #default='/media/dios/dios2/RabbitSegmentation/µCT/images_test/8C_M1_lateral_condyle_XZ')
                    default='/media/dios/dios2/RabbitSegmentation/Histology/Rabbits/Images_CTRL')
parser.add_argument('--mask_dir', type=pathlib.Path,
                    #default='/media/dios/dios2/RabbitSegmentation/µCT/predictions_4fold/8C_M1_lateral_condyle_XZ/Largest')
                    default='/media/dios/dios2/RabbitSegmentation/Histology/Rabbits/Predictions_resnet34_UNet_4fold_CTRL')
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--crop_method', type=str, default='bbox')
parser.add_argument('--saved', type=bool, default=True)
parser.add_argument('--plot', type=bool, default=True)
parser.add_argument('--largest', type=bool, default=False)
args = parser.parse_args()

# Load and set paths
im_files, data = load(args.dataset_root, rgb=True, uCT=False)
mask_files, mask = load(args.mask_dir, rgb=False, uCT=False)

# Expand mask to 3 channels
# mask_large = np.zeros((mask.shape[0], mask.shape[1], 3, mask.shape[2]))
# for i in range(mask_large.shape[2]):
#     mask_large[:, :, i, :] = mask

if args.crop_method == 'bbox':
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
elif args.crop_method == 'value':
    for sample in range(len(mask)):
        w, h = mask[sample].shape
        mask[sample] = mask[sample][w // 2:, :]
        data[sample] = data[sample][w // 2:, :]



# Save images
if args.saved:
    #save_im = '/media/dios/dios2/HistologySegmentation/Images_cropped2'
    save_im_ref = '/media/dios/dios2/RabbitSegmentation/Histology/Rabbits/Predictions_resnet34_UNet_4fold_contour'
    #save_mask = '/media/dios/dios2/HistologySegmentation/Masks_cropped2'
    save_im = str(args.dataset_root / 'crop')
    save_mask = str(args.mask_dir / 'crop')

    if args.plot:
        save(save_im_ref, im_files, data_ref, crop=args.crop)
    if args.crop:
        save(save_mask, mask_files, mask, crop=False)
        save(save_im, im_files, data, crop=False)