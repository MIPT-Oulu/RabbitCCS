import cv2
import pandas as pd
import pathlib
import torch
import dill
from sklearn import model_selection

from rabbitccs.data.transforms import estimate_mean_std


def build_meta_from_files(base_path, phase='train'):
    if phase == 'train':
        masks_loc = base_path / 'masks'
        images_loc = base_path / 'images'
    else:
        masks_loc = base_path / 'predictions_test'
        images_loc = base_path / 'images_test'

    # List files
    images = set(map(lambda x: x.stem, images_loc.glob('**/*[0-9].[pb][nm][gp]')))
    masks = set(map(lambda x: x.stem, masks_loc.glob('**/*[0-9].[pb][nm][gp]')))
    res = masks.intersection(images)

    #masks = list(map(lambda x: pathlib.Path(x).with_suffix('.png'), masks))
    images = list(map(lambda x: pathlib.Path(x.name), images_loc.glob('**/*[0-9].[pb][nm][gp]')))
    masks = list(map(lambda x: pathlib.Path(x.name), masks_loc.glob('**/*[0-9].[pb][nm][gp]')))

    assert len(res), len(masks)

    d_frame = {'fname': [], 'mask_fname': []}

    # Making dataframe
    if str(base_path)[-3:] == 'µCT':
        [d_frame['fname'].append((images_loc / str(img_name).rsplit('_', 1)[0] / img_name)) for img_name in images]
        [d_frame['mask_fname'].append(masks_loc / str(img_name).rsplit('_', 1)[0] / img_name) for img_name in masks]
    else:
        [d_frame['fname'].append((images_loc / img_name)) for img_name in images]
        [d_frame['mask_fname'].append(masks_loc / img_name) for img_name in masks]

    metadata = pd.DataFrame(data=d_frame)

    return metadata


def build_splits(data_dir, args, config, parser, snapshots_dir, snapshot_name):
    # Metadata
    metadata = build_meta_from_files(data_dir)
    metadata['subj_id'] = metadata.fname.apply(lambda x: x.stem[:-9], 0)  # Group_ID

    # Mean and std
    crop = config['training']['crop_size']
    mean_std_path = snapshots_dir / f"mean_std_{crop[0]}x{crop[1]}.pth"
    if mean_std_path.is_file():
        print('==> Loading mean and std from cache')
        tmp = torch.load(mean_std_path)
        mean, std = tmp['mean'], tmp['std']
    else:
        print('==> Estimating mean and std')
        mean, std = estimate_mean_std(config, metadata, parser, args.num_threads, args.bs)
        torch.save({'mean': mean, 'std': std}, mean_std_path)

    print('==> Mean:', mean)
    print('==> STD:', std)

    # Group K-Fold by rabbit ID
    gkf = model_selection.GroupKFold(n_splits=config['training']['n_folds'])

    # Create splits for all folds
    splits_metadata = dict()
    iterator = gkf.split(metadata, groups=metadata.subj_id)
    for fold in range(config['training']['n_folds']):
        train_idx, val_idx = next(iterator)
        splits_metadata[f'fold_{fold}'] = {'train': metadata.iloc[train_idx],
                                           'val': metadata.iloc[val_idx]}

    # Add mean and std to metadata
    splits_metadata['mean'] = mean
    splits_metadata['std'] = std

    with open(snapshots_dir / snapshot_name / 'split_config.dill', 'wb') as f:
        dill.dump(splits_metadata, f)

    return splits_metadata


