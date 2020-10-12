import os
from pathlib import Path
from components.training.session import init_experiment

if __name__ == "__main__":
    # Initialize experiment
    args, config, device, snapshots_dir, snapshot_name = init_experiment()
    base_path = args.data_location / 'µCT'
    #base_path = Path('/media/dios/dios2/RabbitSegmentation/µCT/Should_be_resegmented_with_neural_network/New Manual CC segmentation')
    masks_loc = base_path / 'masks'
    images_loc = base_path / 'images'

    # List files
    samples = os.listdir(images_loc)
    for sample in samples:
        im_path = images_loc / Path(sample)
        mask_path = masks_loc / Path(sample)
        images = list(map(lambda x: x, im_path.glob('**/*[0-9].[pb][nm][gp]')))
        masks = list(map(lambda x: x, mask_path.glob('**/*[0-9].[pb][nm][gp]')))
        images.sort(); masks.sort()
        for slice in range(len(images)):
            # Image
            new_name = images_loc / sample / Path(sample + f'_{str(slice).zfill(8)}{str(images[slice])[-4:]}')
            os.rename(images[slice], new_name)
            new_name = masks_loc / sample / Path(sample + f'_{str(slice).zfill(8)}{str(masks[slice])[-4:]}')
            os.rename(masks[slice], new_name)
