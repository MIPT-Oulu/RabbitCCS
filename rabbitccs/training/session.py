import pathlib
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import torch
import torch.nn as nn
import dill
import json
import cv2
import os
import solt.data as sld
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial

from collagen.data import DataProvider, ItemLoader
from collagen.core.utils import auto_detect_device
from collagen.callbacks.meters import RunningAverageMeter, ItemWiseBinaryJaccardDiceMeter
from collagen.callbacks.logging import ScalarMeterLogger
from collagen.callbacks import ModelSaver, ImageMaskVisualizer, SimpleLRScheduler
from collagen.losses.segmentation import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss

from rabbitccs.data.transforms import train_test_transforms


def init_experiment():
    """
    Setup the model training experiments.
    Lists all configuration files in the args.experiment directory and runs experiments with the given parameters.

    :return: General arguments, experiment parameters, computation device (CPU/GPU)
    """

    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', type=pathlib.Path, default='../../../Data',
                        help='Path to input and target images')
    parser.add_argument('--workdir', type=pathlib.Path, default='../../../workdir/',
                        help='Path for saving the experiment logs and segmentation models')
    parser.add_argument('--experiment', type=pathlib.Path, default='../experiments/run',
                        help='Path to experiment files for training (all experiments are conducted)')
    parser.add_argument('--ID_char', type=str, default='_',
                        help='Separator for the subject ID and image name')
    parser.add_argument('--ID_split', type=int, default=4,
                        help='Count of the ID_char to split the subject ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to allow consistent experiments (e.g. for random augmentations)')
    parser.add_argument('--num_threads', type=int, default=16,
                        help='Number of CPUs for parallel processing')
    parser.add_argument('--gpus', type=int, default=2,
                        help='Number of GPUs for model training')
    args = parser.parse_args()

    # Initialize working directories
    args.snapshots_dir = args.workdir / 'snapshots'
    args.snapshots_dir.mkdir(exist_ok=True)

    # List configuration files
    config_paths = os.listdir(str(args.experiment))
    config_paths.sort()

    # Open configuration files and add to list
    config_list = []
    for config_path in config_paths:
        if config_path[-4:] == '.yml':
            with open(args.experiment / config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                config_list.append(config)

        # Snapshot directory
        encoder = config['model']['backbone']
        decoder = config['model']['decoder']
        experiment = config['training']['experiment']
        snapshot_name = time.strftime(f'{socket.gethostname()}_%Y_%m_%d_%H_%M_%S_{experiment}_{encoder}_{decoder}')
        (args.snapshots_dir / snapshot_name).mkdir(exist_ok=True, parents=True)
        config['training']['snapshot'] = snapshot_name

        # Save the experiment parameters
        with open(args.snapshots_dir / snapshot_name / 'config.yml', 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=False)
        # Save args
        with open(args.snapshots_dir / snapshot_name / 'args.dill', 'wb') as f:
            dill.dump(args, f)

    # Seeding
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Calculation resource
    device = auto_detect_device()

    return args, config_list, device


def init_callbacks(fold_id, config, snapshots_dir, snapshot_name, model, optimizer, data_provider, mean, std):
    """
    Initializes the Collagen callbacks for logging the training and saving models.

    :param fold_id: Number of train/validation fold
    :param config: Experiment configuration
    :param snapshots_dir: Path for saved experiment results (model and logs)
    :param snapshot_name: Name for the experiment
    :param model: Model that is trained
    :param optimizer: Parameter optimizer (e.g. Adam, SGD)
    :param data_provider: Collagen dataloader
    :param mean: Dataset mean
    :param std: Dataset std
    :return: List of training and validation callbacks
    """

    # Snapshot directory
    current_snapshot_dir = snapshots_dir / snapshot_name
    crop = config['training']['crop_size']
    log_dir = current_snapshot_dir / f"fold_{fold_id}_log"
    device = next(model.parameters()).device

    # Tensorboard writer
    writer = SummaryWriter(comment='RabbitCCS', log_dir=log_dir, flush_secs=15, max_queue=1)
    prefix = f"{crop[0]}x{crop[1]}_fold_{fold_id}"

    # Set threshold
    if 'threshold' in config['training']:  # Threshold in config file
        threshold = config['training']['threshold']
    else:  # Not given
        threshold = 0.3 if config['training']['log_jaccard'] else 0.5

    # Callbacks for training and validation phase
    train_cbs = (RunningAverageMeter(prefix="train", name="loss"),
                 ScalarMeterLogger(writer, comment='training', log_dir=str(log_dir))
                 )

    val_cbs = (RunningAverageMeter(prefix="eval", name="loss"),
               ImageMaskVisualizer(writer, log_dir=str(log_dir), comment='visualize', mean=mean, std=std),
               ModelSaver(metric_names='eval/loss',
                          prefix=prefix,
                          save_dir=str(current_snapshot_dir),
                          conditions='min', model=model),
               ItemWiseBinaryJaccardDiceMeter(prefix="eval", name='jaccard',
                                              parse_output=partial(parse_binary_label, threshold=threshold),
                                              parse_target=lambda x: x.squeeze().to(device)),
               ItemWiseBinaryJaccardDiceMeter(prefix="eval", name='dice',
                                              parse_output=partial(parse_binary_label, threshold=threshold),
                                              parse_target=lambda x: x.squeeze().to(device)),
               # Reduce LR on plateau
               SimpleLRScheduler('eval/loss', ReduceLROnPlateau(optimizer,
                                                                patience=int(config['training']['patience']),
                                                                factor=float(config['training']['factor']),
                                                                eps=float(config['training']['eps']))),
               ScalarMeterLogger(writer=writer, comment='validation', log_dir=log_dir))

    return train_cbs, val_cbs


def init_loss(config, device='cuda'):
    """
    Set up the loss function.
    :param config: Experiment congiguration
    :param device: Computation device (CPU/GPU)
    :return: Selected loss function.
    """
    loss = config['training']['loss']
    if loss == 'bce':
        return BCEWithLogitsLoss2d().to(device)
    elif loss == 'jaccard':
        return SoftJaccardLoss(use_log=config['training']['log_jaccard']).to(device)
    elif loss == 'mse':
        return nn.MSELoss().to(device)
    elif loss == 'combined':
        return CombinedLoss([BCEWithLogitsLoss2d(),
                            SoftJaccardLoss(use_log=config['training']['log_jaccard'])]).to(device)
    else:
        raise Exception('No compatible loss selected in experiment_config.yml! Set training->loss accordingly.')


def create_data_provider(args, config, parser, metadata, mean, std):
    """
    Setup dataloader and augmentations
    :param args: General arguments
    :param config: Experiment parameters
    :param parser: Function for loading images
    :param metadata: Image paths and subject IDs
    :param mean: Dataset mean
    :param std: Dataset std
    :return: The compiled dataloader
    """
    # Compile ItemLoaders
    item_loaders = dict()
    for stage in ['train', 'val']:
        item_loaders[f'bfpn_{stage}'] = ItemLoader(meta_data=metadata[stage],
                                                   transform=train_test_transforms(config, mean, std,
                                                   crop_size=tuple(config['training']['crop_size']))[stage],
                                                   parse_item_cb=parser,
                                                   batch_size=config['training']['bs'], num_workers=args.num_threads,
                                                   shuffle=True if stage == "train" else False)

    return DataProvider(item_loaders)


def parse_grayscale(root, entry, transform, data_key, target_key, debug=False):
    """
    Loader function for grayscale images.
    """
    # Image and mask generation
    img = cv2.imread(str(entry.fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:, :, 1] = img[:, :, 0]
    img[:, :, 2] = img[:, :, 0]
    mask = cv2.imread(str(entry.mask_fname), 0) / 255.

    if img.shape[0] != mask.shape[0]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    elif img.shape[1] != mask.shape[1]:
        mask = mask[:, :img.shape[1]]

    img, mask = transform((img, mask))
    img = img.permute(2, 0, 1)  # img.shape[0] is the color channel after permute

    # Debugging
    if debug and np.random.uniform(0, 1) > 0.98:
        plt.imshow(img.numpy().transpose((1, 2, 0)) / 255., cmap='gray')
        plt.colorbar()
        plt.imshow(np.ma.masked_array(mask * 255, mask == 0).squeeze(), cmap='autumn', alpha=0.3)
        plt.show()

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    return {data_key: img, target_key: mask}


def parse_color(root, entry, transform, data_key, target_key, debug=False):
    """
    Loader function for color images.
    """
    # Image and mask generation
    img = cv2.imread(str(entry.fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(entry.mask_fname), 0) / 255.

    if img.shape[0] != mask.shape[0]:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    elif img.shape[1] != mask.shape[1]:
        mask = mask[:, :img.shape[1]]

    img, mask = transform((img, mask))
    img = img.permute(2, 0, 1) / 255.  # img.shape[0] is the color channel after permute

    # Debugging
    if debug:
        plt.imshow(np.asarray(img).transpose((1, 2, 0)))
        plt.imshow(np.asarray(mask).squeeze(), alpha=0.3)
        plt.show()

    # Images are in the format 3xHxW
    # and scaled to 0-1 range
    return {data_key: img, target_key: mask}


def parse_binary_label(x, threshold=0.5):
    out = x.gt(threshold)
    return out.squeeze().float()


def save_config(path, config, args):
    """
    Alternate way to save model parameters.
    """
    with open(path + '/experiment_config.txt', 'w') as f:
        f.write(f'\nArguments file:\n')
        f.write(f'Seed: {args.seed}\n')
        f.write(f'Batch size: {args.bs}\n')
        f.write(f'N_epochs: {args.n_epochs}\n')

        f.write('Configuration file:\n\n')
        for key, val in config.items():
            f.write(f'{key}\n')
            for key2 in config[key].items():
                f.write(f'\t{key2}\n')


def save_transforms(path, config, args, mean, std):
    """
    Function to save the used augmentations.
    """
    transforms = train_test_transforms(config, mean, std, crop_size=tuple(config['training']['crop_size']))
    # Save the experiment parameters
    with open(path / 'transforms.json', 'w') as f:
        f.writelines(json.dumps(transforms['train_list'][1].serialize(), indent=4))
