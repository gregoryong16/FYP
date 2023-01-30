import asyncio
from functools import partial

import cv2
import torch
import torch.backends.cudnn as cudnn
from joblib import cpu_count
from torch.utils.data import DataLoader

from config import get_config
from data import detection_collate, get_dataset
# from model_training.detection.retinanet_v2 import build_retinanet
from retinanet import build_retinanet
from ssd import build_ssd
from trainer import Trainer

cudnn.benchmark = True
cv2.setNumThreads(0)


def _get_model(config):
    model_config = config['model']
    if model_config['name'] == 'ssd':
        model = build_ssd(model_config)
    elif model_config['name'] == 'retina_net':
        model = build_retinanet(model_config)
    else:
        raise ValueError("Model [%s] not recognized." % model_config['name'])
    return model


if __name__ == '__main__':
    config = get_config("config/train.yaml")

    batch_size = config.pop('batch_size')
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=cpu_count(),
                             shuffle=True, drop_last=True,
                             collate_fn=detection_collate, pin_memory=True)
    
    datasets = map(config.pop, ('train', 'val'))
    datasets = map(get_dataset, datasets)
    train, val = map(get_dataloader, datasets)
    
    trainer = Trainer(_get_model(config).cuda(), config, train=train, val=val)
    trainer.train()