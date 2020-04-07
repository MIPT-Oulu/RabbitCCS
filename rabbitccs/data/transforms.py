import torch
import numpy as np
from functools import partial
from tqdm import tqdm

import solt.transforms as slt
import solt.data as sld
import solt.core as slc

from collagen.data.utils import ApplyTransform, Compose
from collagen.data import ItemLoader


def normalize_channel_wise(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Normalizes given tensor channel-wise
    Parameters
    ----------
    tensor: torch.Tensor
        Tensor to be normalized
    mean: torch.tensor
        Mean to be subtracted
    std: torch.Tensor
        Std to be divided by
    Returns
    -------
    result: torch.Tensor
    """
    if len(tensor.size()) != 3:
        raise ValueError
    # Original version
    """
    for channel in range(tensor.size(0)):
        tensor[channel, :, :] -= mean[channel]
        tensor[channel, :, :] /= std[channel]

    return tensor
    """
    # Modified shape
    for channel in range(tensor.size(2)):
        tensor[:, :, channel] -= mean[channel]
        tensor[:, :, channel] /= std[channel]

    return tensor


def numpy2tens(x: np.ndarray, dtype='f') -> torch.Tensor:
    """Converts a numpy array into torch.Tensor
    Parameters
    ----------
    x: np.ndarray
        Array to be converted
    dtype: str
        Target data type of the tensor. Can be f - float and l - long
    Returns
    -------
    result: torch.Tensor
    """
    x = x.squeeze()
    x = torch.from_numpy(x)
    if x.dim() == 2:  # CxHxW format
        x = x.unsqueeze(0)

    if dtype == 'f':
        return x.float()
    elif dtype == 'l':
        return x.long()
    else:
        raise NotImplementedError


def wrap_solt(entry):
    return sld.DataContainer(entry, 'IM', transform_settings={0: {'interpolation': 'bilinear'},
                                                              1: {'interpolation': 'nearest'}})


def unwrap_solt(dc):
    return dc.data


def train_test_transforms(conf, mean=None, std=None, crop_size=(512, 1024)):
    trf = conf['training']
    prob = trf['transform_probability']
    # Training transforms
    if trf['experiment'] == '3D':
        train_transforms = [
            slt.RandomProjection(
                slc.Stream([
                    slt.RandomRotate(rotation_range=tuple(trf['rotation_range']), p=prob),
                    slt.RandomScale(range_x=tuple(trf['scale_range']),
                                    range_y=tuple(trf['scale_range']), same=False, p=prob),
                    slt.RandomShear(range_x=tuple(trf['shear_range']),
                                    range_y=tuple(trf['shear_range']), p=prob),
                    slt.RandomTranslate(range_x=trf['translation_range'], range_y=trf['translation_range'], p=prob)
                ]),
                v_range=tuple(trf['v_range'])),
            # Spatial
            slt.RandomFlip(p=prob),
            slt.PadTransform(pad_to=crop_size),
            slt.CropTransform(crop_mode='r', crop_size=crop_size),
            # Intensity

            #slt.ImageGammaCorrection(gamma_range=tuple(trf['gamma_range']), p=prob),
            #slt.ImageRandomHSV(h_range=tuple(trf['hsv_range']),
            #                   s_range=tuple(trf['hsv_range']),
            #                   v_range=tuple(trf['hsv_range']), p=prob),
            # Brightness/contrast
            slc.SelectiveStream([
                slt.ImageRandomBrightness(brightness_range=tuple(trf['brightness_range']), p=prob),
                slt.ImageRandomContrast(contrast_range=trf['contrast_range'], p=prob)]),
            # Noise
            slc.SelectiveStream([
                slt.ImageSaltAndPepper(p=prob, gain_range=trf['gain_range_sp']),
                slt.ImageAdditiveGaussianNoise(p=prob, gain_range=trf['gain_range_gn']),
                slc.SelectiveStream([
                    slt.ImageBlur(p=prob, blur_type='g', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma'])),
                    slt.ImageBlur(p=prob, blur_type='m', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma']))])])
        ]
    else:
        train_transforms = [
            # Projection
            slt.RandomProjection(
                slc.Stream([
                    slt.RandomRotate(rotation_range=tuple(trf['rotation_range']), p=prob),
                    slt.RandomScale(range_x=tuple(trf['scale_range']),
                                    range_y=tuple(trf['scale_range']), same=False, p=prob),
                    #slt.RandomShear(range_x=tuple(trf['shear_range']),
                    #                range_y=tuple(trf['shear_range']), p=prob),
                    #slt.RandomTranslate(range_x=trf['translation_range'], range_y=trf['translation_range'], p=prob)
                ]),
                v_range=tuple(trf['v_range'])),
            # Spatial
            slt.RandomFlip(p=prob),
            slt.PadTransform(pad_to=crop_size[1]),
            slt.CropTransform(crop_mode='r', crop_size=crop_size),
            # Intensity
            # Add an empty stream
            #slc.SelectiveStream([]),
            slc.SelectiveStream([
                slt.ImageGammaCorrection(gamma_range=tuple(trf['gamma_range']), p=prob),
                slt.ImageRandomHSV(h_range=tuple(trf['hsv_range']),
                                   s_range=tuple(trf['hsv_range']),
                                   v_range=tuple(trf['hsv_range']), p=prob)]),
            slc.SelectiveStream([
                slt.ImageRandomBrightness(brightness_range=tuple(trf['brightness_range']), p=prob),
                slt.ImageRandomContrast(contrast_range=trf['contrast_range'], p=prob)]),
            slc.SelectiveStream([
                slt.ImageSaltAndPepper(p=prob, gain_range=trf['gain_range_sp']),
                slt.ImageAdditiveGaussianNoise(p=prob, gain_range=trf['gain_range_gn']),
                slc.SelectiveStream([
                    slt.ImageBlur(p=prob, blur_type='g', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma'])),
                    slt.ImageBlur(p=prob, blur_type='m', k_size=(3, 7, 11), gaussian_sigma=tuple(trf['sigma']))])])
        ]

    train_trf = [
        wrap_solt,
        slc.Stream(train_transforms),
        unwrap_solt,
        ApplyTransform(numpy2tens, (0, 1, 2))
    ]
    # Validation transforms
    val_trf = [
        wrap_solt,
        slc.Stream([
            slt.PadTransform(pad_to=crop_size[1]),
            slt.CropTransform(crop_mode='r', crop_size=crop_size)
        ]),
        unwrap_solt,
        ApplyTransform(numpy2tens, idx=(0, 1, 2))
    ]
    # Test transforms
    test_trf = [
        unwrap_solt,
        ApplyTransform(numpy2tens, idx=(0, 1, 2))
    ]

    # Use normalize_channel_wise if mean and std not calculated
    if mean is not None and std is not None:
        train_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    if mean is not None and std is not None:
        val_trf.append(ApplyTransform(partial(normalize_channel_wise, mean=mean, std=std)))

    # Compose transforms
    train_trf_cmp = Compose(train_trf)
    val_trf_cmp = Compose(val_trf)
    test_trf_cmp = Compose(test_trf)

    return {'train': train_trf_cmp, 'val': val_trf_cmp, 'test': test_trf_cmp,
            'train_list': train_trf, 'val_list': val_trf, 'test_list': test_trf}


def estimate_mean_std(config, metadata, parse_item_cb, num_threads=8, bs=16):
    mean_std_loader = ItemLoader(meta_data=metadata,
                                 transform=train_test_transforms(config)['train'],
                                 parse_item_cb=parse_item_cb,
                                 batch_size=bs, num_workers=num_threads,
                                 shuffle=False)

    mean = None
    std = None
    for i in tqdm(range(len(mean_std_loader)), desc='Calculating mean and standard deviation'):
        for batch in mean_std_loader.sample():
            if mean is None:
                mean = torch.zeros(batch['data'].size(1))
                std = torch.zeros(batch['data'].size(1))
            # for channel in range(batch['data'].size(1)):
            #     mean[channel] += batch['data'][:, channel, :, :].mean().item()
            #     std[channel] += batch['data'][:, channel, :, :].std().item()
            mean += batch['data'].mean().item()
            std += batch['data'].std().item()

    mean /= len(mean_std_loader)
    std /= len(mean_std_loader)

    return mean, std
