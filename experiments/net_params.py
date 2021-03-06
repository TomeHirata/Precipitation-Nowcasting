import sys
sys.path.insert(0, '..')
from nowcasting.hko.dataloader import HKOIterator
from nowcasting.config import cfg
import torch
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from collections import OrderedDict
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.models.loss import Weighted_mse_mae
from nowcasting.models.trajGRU import TrajGRU
from nowcasting.train_and_test import train_and_test
import numpy as np
from nowcasting.hko.evaluation import *
from nowcasting.models.convLSTM import ConvLSTM
from nowcasting.models.model import activation

batch_size = cfg.GLOBAL.BATCH_SZIE

IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN

# build model
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]


# build model
conv2d_params = OrderedDict({
    'conv1_relu_1': [5, 64, 7, 5, 1],
    'conv2_relu_1': [64, 192, 5, 3, 1],
    'conv3_relu_1': [192, 192, 3, 2, 1],
    'deconv1_relu_1': [192, 192, 4, 2, 1],
    'deconv2_relu_1': [192, 64, 5, 3, 1],
    'deconv3_relu_1': [64, 64, 7, 5, 1],
    'conv3_relu_2': [64, 20, 3, 1, 1],
    'conv3_3': [20, 20, 1, 1, 0]
})


# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1),
    ]
]
# rnn_block_num = len(cfg.RNN_BLOCKS.NUM_FILTER)
# build model
def convlstm_encoder_params_mnist():
    return [
        [
            OrderedDict({
                layer[0]: [layer[1], layer[2], layer[3], layer[4], layer[5]]
                for layer in sub
            })
            for sub in cfg.MODEL.ENCODER.DOWNSAMPLE
        ],
    # in_channels=v[0], out_channels=v[1],kernel_size=v[2], stride=v[3],padding=v[4]
        [
            ConvLSTM(input_channel=layer[1], num_filter=layer[2], b_h_w=(batch_size, layer[3], layer[4]),
                    kernel_size=layer[5], stride=layer[6], padding=layer[7])
            for layer in cfg.MODEL.ENCODER.RNN_BLOCKS.LAYERS
        ]
    ]

def convlstm_forecaster_params_mnist():
    return [
        [
            OrderedDict({
                layer[0]: [layer[1], layer[2], layer[3], layer[4], layer[5]]
                for layer in sub
            })
            for sub in cfg.MODEL.FORECASTER.UPSAMPLE
        ],
        [
            ConvLSTM(input_channel=layer[1], num_filter=layer[2], b_h_w=(batch_size, layer[3], layer[4]),
                    kernel_size=layer[5], stride=layer[6], padding=layer[7])
            for layer in cfg.MODEL.FORECASTER.RNN_BLOCKS.LAYERS
        ]
    ]

def encoder_params_mnist():
    return [
        [
            OrderedDict({
                layer[0]: [layer[1], layer[2], layer[3], layer[4], layer[5]]
                for layer in sub
            })
            for sub in cfg.MODEL.ENCODER.DOWNSAMPLE
        ],

        [
            TrajGRU(input_channel=cfg.MODEL.ENCODER.RNN_BLOCKS.NUM_INPUT[i], num_filter=cfg.MODEL.ENCODER.RNN_BLOCKS.NUM_FILTER[i],
                b_h_w=(batch_size, cfg.MODEL.ENCODER.RNN_BLOCKS.HW[i][0], cfg.MODEL.ENCODER.RNN_BLOCKS.HW[i][1]), zoneout=0.0, L=cfg.MODEL.ENCODER.RNN_BLOCKS.L[i],
                i2h_kernel=cfg.MODEL.ENCODER.RNN_BLOCKS.I2H_KERNEL[i], i2h_stride=cfg.MODEL.ENCODER.RNN_BLOCKS.I2H_STRIDE[i], i2h_pad=cfg.MODEL.ENCODER.RNN_BLOCKS.I2H_PAD[i],
                h2h_kernel=cfg.MODEL.ENCODER.RNN_BLOCKS.H2H_KERNEL[i], h2h_dilate=cfg.MODEL.ENCODER.RNN_BLOCKS.H2H_DILATE[i],
                act_type=activation(cfg.MODEL.RNN_ACT_TYPE, negative_slope=0.2, inplace=True))
            for i in range(len(cfg.MODEL.ENCODER.RNN_BLOCKS.NUM_FILTER))
        ]
    ]

def forecaster_params_mnist():
    return [
        [
            OrderedDict({
                layer[0]: [layer[1], layer[2], layer[3], layer[4], layer[5]]
                for layer in sub
            })
            for sub in cfg.MODEL.FORECASTER.UPSAMPLE
        ],
        [
            TrajGRU(input_channel=cfg.MODEL.FORECASTER.RNN_BLOCKS.NUM_INPUT[i], num_filter=cfg.MODEL.FORECASTER.RNN_BLOCKS.NUM_FILTER[i],
                b_h_w=(batch_size, cfg.MODEL.FORECASTER.RNN_BLOCKS.HW[i][0], cfg.MODEL.FORECASTER.RNN_BLOCKS.HW[i][1]),
                zoneout=0.0, L=cfg.MODEL.FORECASTER.RNN_BLOCKS.L[i],
                i2h_kernel=cfg.MODEL.FORECASTER.RNN_BLOCKS.I2H_KERNEL[i], i2h_stride=cfg.MODEL.FORECASTER.RNN_BLOCKS.I2H_STRIDE[i], i2h_pad=cfg.MODEL.FORECASTER.RNN_BLOCKS.I2H_PAD[i],
                h2h_kernel=cfg.MODEL.FORECASTER.RNN_BLOCKS.H2H_KERNEL[i], h2h_dilate=cfg.MODEL.FORECASTER.RNN_BLOCKS.H2H_DILATE[i],
                act_type=activation(cfg.MODEL.RNN_ACT_TYPE, negative_slope=0.2, inplace=True))
            for i in range(len(cfg.MODEL.FORECASTER.RNN_BLOCKS.NUM_FILTER))
        ]
    ]