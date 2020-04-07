import numpy as np
from torch import optim, cuda, nn
from time import time
from copy import deepcopy
import gc
import cv2
import segmentation_models_pytorch as smp

from collagen.modelzoo.segmentation import EncoderDecoder
from collagen.losses.segmentation import CombinedLoss, BCEWithLogitsLoss2d, SoftJaccardLoss
from collagen.strategies import Strategy


from rabbitccs.training.session import create_data_provider, init_experiment, init_callbacks, save_transforms,\
    init_loss, parse_grayscale, parse_color_im

from rabbitccs.data.splits import build_splits

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


if __name__ == "__main__":
    # Timing
    start = time()

    # Initialize experiment
    args_base, config_list, device = init_experiment()

    for experiment in range(len(config_list)):
        # Current experiment
        args = deepcopy(args_base)
        config = config_list[experiment]

        # Update arguments according to the configuration file
        if config['training']['experiment'] == '3D':
            args.data_location = args.data_location / 'µCT'
        elif config['training']['experiment'] == '2D_large':
            args.data_location = args.data_location / 'human'

        # Loss
        loss_criterion = init_loss(config, device=device)

        # Split training folds
        splits_metadata = build_splits(args.data_location, args, config, parse_grayscale,
                                       args.snapshots_dir, config['training']['snapshot'])
        mean, std = splits_metadata['mean'], splits_metadata['std']

        # Save transforms list
        save_transforms(args.snapshots_dir / config['training']['snapshot'], config, args, mean, std)

        # Training for separate folds
        for fold in range(config['training']['n_folds']):
            print(f'\nTraining fold {fold}')
            # Initialize data provider
            data_provider = create_data_provider(args, config, parse_grayscale, metadata=splits_metadata[f'fold_{fold}'],
                                                 mean=mean, std=std)
            # Initialize model model
            backbone = config['model']['backbone']
            decoder = config['model']['decoder']
            if decoder.lower() == 'unet':
                model = smp.Unet(backbone, encoder_weights="imagenet", activation='sigmoid')
            else:
                model = EncoderDecoder(**config['model'])
            if args.gpus > 1:
                model = nn.DataParallel(model).to(device)
            else:
                model = model.to(device)

            print(f'Encoder: {backbone}, decoder: {decoder}')

            # Optimizer
            optimizer = optim.Adam(model.parameters(),
                                   lr=config['training']['lr'],
                                   weight_decay=config['training']['wd'])
            # Callbacks
            train_cbs, val_cbs = init_callbacks(fold, config, args.snapshots_dir, config['training']['snapshot'], model,
                                                optimizer, data_provider, mean, std)
            # Run training
            strategy = Strategy(data_provider=data_provider,
                                train_loader_names=tuple(config['data_sampling']['train']['data_provider'].keys()),
                                val_loader_names=tuple(config['data_sampling']['eval']['data_provider'].keys()),
                                data_sampling_config=config['data_sampling'],
                                loss=loss_criterion,
                                model=model,
                                n_epochs=config['training']['epochs'],
                                optimizer=optimizer,
                                train_callbacks=train_cbs,
                                val_callbacks=val_cbs,
                                device=device)
            strategy.run()

            # Manage memory
            del strategy
            del model
            cuda.empty_cache()
            gc.collect()

        dur = time() - start
        print(f'Model {experiment + 1} trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')

    dur = time() - start
    print(f'Models trained in {dur // 3600} hours, {(dur % 3600) // 60} minutes, {dur % 60} seconds.')
