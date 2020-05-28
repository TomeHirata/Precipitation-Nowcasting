import argparse
import os
import random
import logging
import numpy as np
import mxnet as mx
import torch
from mxnet import nd
from nowcasting.config import cfg, cfg_from_file, save_cfg
from nowcasting.helpers.gifmaker import save_gif
from nowcasting.movingmnist_iterator import MovingMNISTAdvancedIterator
from nowcasting.helpers.ordered_easydict import OrderedEasyDict as edict
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.models.loss import Weighted_mse_mae
from nowcasting.train_and_test import train_mnist
from experiments.net_params import convlstm_encoder_params_mnist, convlstm_forecaster_params_mnist, encoder_params_mnist, forecaster_params_mnist

random.seed(123)
mx.random.seed(9302)
np.random.seed(9212)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the MovingMNIST++ dataset')
    parser.add_argument('--batch_size', dest='batch_size', help="batchsize of the training process",
                        default=None, type=int)
    parser.add_argument('--cfg', dest='cfg_file', help='Training configuration file', default=None, type=str)
    parser.add_argument('--resume', help='Continue to train the previous model', action='store_true',
                        default=False)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--ctx', dest='ctx', help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`',
                        type=str, default='gpu')
    parser.add_argument('--lr', dest='lr', help='learning rate', default=None, type=float)
    parser.add_argument('--wd', dest='wd', help='weight decay', default=None, type=float)
    parser.add_argument('--grad_clip', dest='grad_clip', help='gradient clipping threshold',
                        default=None, type=float)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file, target=cfg)
    if args.batch_size is not None:
        cfg.MODEL.TRAIN.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.MODEL.TRAIN.LR = args.lr
    if args.wd is not None:
        cfg.MODEL.TRAIN.WD = args.wd
    if args.grad_clip is not None:
        cfg.MODEL.TRAIN.GRAD_CLIP = args.grad_clip
    if args.wd is not None:
        cfg.MODEL.TRAIN.WD = args.wd
    cfg.MODEL.SAVE_DIR = args.save_dir
    logging.info(args)
    return args


def save_movingmnist_cfg(dir_path):
    tmp_cfg = edict()
    tmp_cfg.MOVINGMNIST = cfg.MOVINGMNIST
    tmp_cfg.MODEL = cfg.MODEL
    save_cfg(dir_path=dir_path, source=tmp_cfg)


def train(args):
    base_dir = args.save_dir
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    # logging_config(folder=base_dir, name="training")
    save_movingmnist_cfg(base_dir)

    batch_size = cfg.GLOBAL.BATCH_SZIE
    max_iterations = 200000
    test_iteration_interval = 2000
    test_and_save_checkpoint_iterations = 2000
    LR_step_size = 20000
    gamma = 0.7

    LR = 1e-4

    criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)
    if cfg.MODEL.TYPE == 'TrajGRU':
        encoder_subnet, encoder_net = encoder_params_mnist()
        forecaster_subnet, forecaster_net = forecaster_params_mnist()
    elif cfg.MODEL.TYPE == 'ConvLSTM':
        encoder_subnet, encoder_net = convlstm_encoder_params_mnist()
        forecaster_subnet, forecaster_net = convlstm_forecaster_params_mnist()
    else:
        raise NotImplementedError('Model is not found.')

    encoder = Encoder(encoder_subnet, encoder_net).to(cfg.GLOBAL.DEVICE)
    forecaster = Forecaster(forecaster_subnet, forecaster_net).to(cfg.GLOBAL.DEVICE)

    encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

    optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)
    folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]

    train_mnist(encoder_forecaster, optimizer, criterion, exp_lr_scheduler, batch_size, max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations, folder_name, base_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
