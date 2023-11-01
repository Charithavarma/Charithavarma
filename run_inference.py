import numpy as np
import torch
import torch.nn as nn
from model import UNET
from utils import get_data_loaders, calculate_accuracy
from config import hp


def run_inference(model_path):
    # model
    model = UNET().to(device).double()
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)

    test_loader = get_data_loaders(hp=hp, data_type='test', batch_size=hp['batch_size'], shuffle=False, drop_last=False,
                                   num_workers=hp['num_workers'], pin_memory=True)

    model.eval()
    test_accuracy = 0.0

    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image = image.to(device, dtype=torch.double)
            mask = mask.to(device, dtype=torch.double)

            preds = model(image)
            test_accuracy += calculate_accuracy(preds=preds, targets=mask)

    # accuracy = 100 * test_accuracy
    print('Dice Score on test set: %.2f %%' % test_accuracy)


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_dir = " "  # example: "checkpoints/ResNet_20201108_222637/"
    model_path = "best_model.pth"  # example: "pretrained_model.pth"

    run_inference(model_path)
