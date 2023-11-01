hp = {
    'data_path': "../lung_segmentation/Lung Disease Dataset/train/Viral Pneumonia",
    'output_dir': "./",

    # model settings
    'net_type': 'unet',
    'in_channels': 1,
    'out_channels': 1,
    'features': [64, 128, 256, 512],

    # train settings
    "num_epochs": 1,
    "learning_rate": 1e-4,
    "batch_size": 1,
    "num_workers": 4,
}