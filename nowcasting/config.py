from nowcasting.helpers.ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch
from collections import OrderedDict
import yaml
import logging
import subprocess
import json

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __C.GLOBAL.DEVICE.type == 'cuda':
    gpu_info = get_gpu_info()
    memory_use = np.array(list(map(int, [info['utilization.gpu'] for info in gpu_info])))
    __C.GLOBAL.DEVICE = torch.device("cuda:%i" % (memory_use.argmin()) if torch.cuda.is_available() else "cpu")
__C.GLOBAL.BATCH_SZIE = 2
# for dirs in ['/home/hzzone/save', '/Users/hzzone/Downloads']:
#     if os.path.exists(dirs):
#         __C.GLOBAL.MODEL_SAVE_DIR = dirs
# assert __C.GLOBAL.MODEL_SAVE_DIR is not None


__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
__C.HKO_DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'hko_data')

for dirs in ['/Users/hzzone/Downloads/HKO-7_data/radarPNG', '/home/hzzone/HKO-7/radarPNG']:
    if os.path.exists(dirs):
        __C.HKO_PNG_PATH = dirs
for dirs in ['/Users/hzzone/Downloads/HKO-7_data/radarPNG_mask', '/home/hzzone/HKO-7/radarPNG_mask']:
    if os.path.exists(dirs):
        __C.HKO_MASK_PATH = dirs

__C.MNIST_PATH = os.path.join(__C.ROOT_DIR, 'mnist_data')
if not os.path.exists(__C.MNIST_PATH):
    os.makedirs(__C.MNIST_PATH)
__C.HKO_DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'hko_data')

__C.HKO = edict()


__C.HKO.EVALUATION = edict()
__C.HKO.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
__C.HKO.EVALUATION.CENTRAL_REGION = (120, 120, 360, 360)
__C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)

__C.HKO.EVALUATION.VALID_DATA_USE_UP = True
# __C.HKO.EVALUATION.VALID_TIME = 100
__C.HKO.EVALUATION.VALID_TIME = 20


__C.HKO.BENCHMARK = edict()

__C.HKO.BENCHMARK.STAT_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'benchmark_stat')
if not os.path.exists(__C.HKO.BENCHMARK.STAT_PATH):
    os.makedirs(__C.HKO.BENCHMARK.STAT_PATH)

__C.HKO.BENCHMARK.VISUALIZE_SEQ_NUM = 10  # Number of sequences that will be plotted and saved to the benchmark directory
__C.HKO.BENCHMARK.IN_LEN = 5   # The maximum input length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.OUT_LEN = 20  # The maximum output length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.STRIDE = 5   # The stride


# pandas data
__C.HKO_PD_BASE_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'pd')
if not os.path.exists(__C.HKO_PD_BASE_PATH):
    os.makedirs(__C.HKO_PD_BASE_PATH)
__C.HKO_VALID_DATETIME_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'valid_datetime.pkl')
__C.HKO_SORTED_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'sorted_day.pkl')
__C.HKO_RAINY_TRAIN_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_train_days.txt')
__C.HKO_RAINY_VALID_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_valid_days.txt')
__C.HKO_RAINY_TEST_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_test_days.txt')

__C.HKO_PD = edict()
__C.HKO_PD.ALL = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all.pkl')
__C.HKO_PD.ALL_09_14 = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all_09_14.pkl')
__C.HKO_PD.ALL_15 = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all_15.pkl')
__C.HKO_PD.RAINY_TRAIN = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_train.pkl')
__C.HKO_PD.RAINY_VALID = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_valid.pkl')
__C.HKO_PD.RAINY_TEST = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_test.pkl')


__C.HKO.ITERATOR = edict()
__C.HKO.ITERATOR.WIDTH = 480
__C.HKO.ITERATOR.HEIGHT = 480
__C.HKO.ITERATOR.FILTER_RAINFALL = True           # Whether to discard part of the rainfall, has a denoising effect
__C.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD = 0.28 # All the pixel values that are smaller than round(threshold * 255) will be discarded

__C.MOVINGMNIST = edict()
__C.MOVINGMNIST.DISTRACTOR_NUM = 0
__C.MOVINGMNIST.VELOCITY_LOWER = 0.0
__C.MOVINGMNIST.VELOCITY_UPPER = 3.6
__C.MOVINGMNIST.SCALE_VARIATION_LOWER = 1/1.1
__C.MOVINGMNIST.SCALE_VARIATION_UPPER = 1.1
__C.MOVINGMNIST.ROTATION_LOWER = -30
__C.MOVINGMNIST.ROTATION_UPPER = 30
__C.MOVINGMNIST.ILLUMINATION_LOWER = 0.6
__C.MOVINGMNIST.ILLUMINATION_UPPER = 1.0
__C.MOVINGMNIST.DIGIT_NUM = 3
__C.MOVINGMNIST.IN_LEN = 10
__C.MOVINGMNIST.OUT_LEN = 10
__C.MOVINGMNIST.TESTING_LEN = 20
__C.MOVINGMNIST.IMG_SIZE = 64
__C.MOVINGMNIST.TEST_FILE = os.path.join(__C.MNIST_PATH, "movingmnist_10000_nodistr.npz")

__C.MODEL = edict()
__C.MODEL.TYPE = 'TrajGRU'
__C.MODEL.RESUME = False  # If True, load LOAD_ITER parameters from LOAD_DIR
__C.MODEL.TESTING = False # If True, run in Testing mode
__C.MODEL.LOAD_DIR = "" # The directory to load the pre-trained parameters
                        # Could be like `D:\\HKUST\\3-2\\NIPS2017\\hko_0502\\bal_loss_direct`
__C.MODEL.LOAD_ITER = 79999           # Only applicable when LOAD_DIR is non-empty
__C.MODEL.SAVE_DIR = ""
__C.MODEL.CNN_ACT_TYPE = "leaky"
__C.MODEL.RNN_ACT_TYPE = "leaky"
__C.MODEL.FRAME_STACK = 1          # Stack multiple frames as the input
__C.MODEL.FRAME_SKIP = 1           # The frame skip size
__C.MODEL.IN_LEN = 5               # Size of the input
__C.MODEL.OUT_LEN = 20             # Size of the output
__C.MODEL.OUT_TYPE = "direct"      # Can be "direct", or "DFN"
__C.MODEL.NORMAL_LOSS_GLOBAL_SCALE = 0.00005
__C.MODEL.USE_BALANCED_LOSS = True
__C.MODEL.TEMPORAL_WEIGHT_TYPE = "same"  # Can be "same", "linear" or "exponential"
__C.MODEL.TEMPORAL_WEIGHT_UPPER = 5      # Only applicable when temporal_weights_type is "linear" or "exponential"
                                         # If linear
                                         #   the weights will be increased following (1 + i * (upper - 1) / (T - 1))
                                         # If exponential
                                         #   the weights will be increased following exp^{i * \ln(upper) / (T-1)}
__C.MODEL.L1_LAMBDA = 1.0
__C.MODEL.L2_LAMBDA = 1.0
__C.MODEL.GDL_LAMBDA = 0.0
__C.MODEL.USE_SEASONALITY = False          # Whether to use seasonality

__C.MODEL.TRAJRNN = edict()
__C.MODEL.TRAJRNN.INIT_GRID = True
__C.MODEL.TRAJRNN.FLOW_LR_MULT = 1.0
__C.MODEL.TRAJRNN.SAVE_MID_RESULTS = False

__C.MODEL.ENCODER = edict()
__C.MODEL.ENCODER.HAS_MASK = True
__C.MODEL.ENCODER.FEATMAP_SIZE = [96, 32, 16]
__C.MODEL.ENCODER.FIRST_CONV = (8, 7, 5, 1)  # Num filter, kernel, stride, pad
__C.MODEL.ENCODER.LAST_DECONV = (8, 7, 5, 1)  # Num filter, kernel, stride, pad
__C.MODEL.ENCODER.DOWNSAMPLE = [
    [
        ['conv1_leaky_1', 1, 8, 3, 1, 1]
    ],
    [
        ['conv2_leaky_1', 64, 64, 3, 2, 1]
    ],
    [
        ['conv3_leaky_1', 96, 96, 3, 2, 1]
    ]
]
__C.MODEL.ENCODER.UPSAMPLE = [(5, 3, 1),
                                         (4, 2, 1)]  # (kernel, stride, pad) for deconv2d

__C.MODEL.ENCODER.RNN_BLOCKS = edict()    # Define the RNN blocks for the encoder RNN
                                                     # In our network, the forecaster RNN will always have the reverse structure of encoder RNN
__C.MODEL.ENCODER.RNN_BLOCKS.RES_CONNECTION = True
__C.MODEL.ENCODER.RNN_BLOCKS.LAYER_TYPE = ["ConvGRU", "ConvGRU", "ConvGRU"]
__C.MODEL.ENCODER.RNN_BLOCKS.STACK_NUM = [2, 3, 3]
# These features are used for both ConvGRU
__C.MODEL.ENCODER.RNN_BLOCKS.NUM_FILTER = [32, 64, 64]
__C.MODEL.ENCODER.RNN_BLOCKS.H2H_KERNEL = [(5, 5), (5, 5), (3, 3)]
__C.MODEL.ENCODER.RNN_BLOCKS.H2H_DILATE = [(1, 1), (1, 1), (1, 1)]
__C.MODEL.ENCODER.RNN_BLOCKS.I2H_KERNEL = [(3, 3), (3, 3), (3, 3)]
__C.MODEL.ENCODER.RNN_BLOCKS.I2H_PAD = [(1, 1), (1, 1), (1, 1)]
# These features are only used in TrajGRU
__C.MODEL.ENCODER.RNN_BLOCKS.L = [5, 5, 5]
__C.MODEL.ENCODER.RNN_BLOCKS.LAYERS = [
    ['ConvLSTM', 8, 64, 64, 64, 3, 1, 1],
    ['ConvLSTM', 64, 96, 32, 32, 3, 1, 1],
    ['ConvLSTM', 96, 96, 16, 16, 3, 1, 1]
]
__C.MODEL.ENCODER.RNN_BLOCKS.NUM_INPUT = [1, 64, 96]
__C.MODEL.ENCODER.RNN_BLOCKS.HW = None
__C.MODEL.ENCODER.RNN_BLOCKS.I2H_STRIDE = None

__C.MODEL.FORECASTER = edict()
__C.MODEL.FORECASTER.HAS_MASK = True
__C.MODEL.FORECASTER.FEATMAP_SIZE = [96, 32, 16]
__C.MODEL.FORECASTER.FIRST_CONV = (8, 7, 5, 1)  # Num filter, kernel, stride, pad
__C.MODEL.FORECASTER.LAST_DECONV = (8, 7, 5, 1)  # Num filter, kernel, stride, pad
__C.MODEL.FORECASTER.DOWNSAMPLE = [(5, 3, 1),
                                           (3, 2, 1)]  # (kernel, stride, pad) for conv2d
__C.MODEL.FORECASTER.UPSAMPLE = [
    [
        ['deconv1_leaky_1', 96, 96, 4, 2, 1]
    ],
    [
        ['deconv2_leaky_1', 96, 64, 4, 2, 1]
    ], [
        ['deconv1_leaky_1', 1, 8, 3, 1, 1],
        ['conv3_leaky_2', 8, 8, 3, 1, 1],
        ['conv3_3', 8, 1, 1, 1, 0]
    ]
]

__C.MODEL.FORECASTER.RNN_BLOCKS = edict()    # Define the RNN blocks for the encoder RNN
                                                     # In our network, the forecaster RNN will always have the reverse structure of encoder RNN
__C.MODEL.FORECASTER.RNN_BLOCKS.RES_CONNECTION = True
__C.MODEL.FORECASTER.RNN_BLOCKS.LAYER_TYPE = ["ConvGRU", "ConvGRU", "ConvGRU"]
__C.MODEL.FORECASTER.RNN_BLOCKS.STACK_NUM = [2, 3, 3]
# These features are used for both ConvGRU
__C.MODEL.FORECASTER.RNN_BLOCKS.NUM_FILTER = [32, 64, 64]
__C.MODEL.FORECASTER.RNN_BLOCKS.H2H_KERNEL = [(5, 5), (5, 5), (3, 3)]
__C.MODEL.FORECASTER.RNN_BLOCKS.H2H_DILATE = [(1, 1), (1, 1), (1, 1)]
__C.MODEL.FORECASTER.RNN_BLOCKS.I2H_KERNEL = [(3, 3), (3, 3), (3, 3)]
__C.MODEL.FORECASTER.RNN_BLOCKS.I2H_PAD = [(1, 1), (1, 1), (1, 1)]
# These features are only used in TrajGRU
__C.MODEL.FORECASTER.RNN_BLOCKS.L = [5, 5, 5]
__C.MODEL.FORECASTER.RNN_BLOCKS.LAYERS = [['ConvLSTM', 96, 96, 16, 16, 3, 1, 1],['ConvLSTM', 96, 96, 32, 32, 3, 1, 1],['ConvLSTM', 64, 64, 64, 64, 3, 1, 1]]
__C.MODEL.FORECASTER.RNN_BLOCKS.NUM_INPUT = [96, 96, 64]
__C.MODEL.FORECASTER.RNN_BLOCKS.HW = None
__C.MODEL.FORECASTER.RNN_BLOCKS.I2H_STRIDE = None

__C.MODEL.DECONVBASELINE = edict()
__C.MODEL.DECONVBASELINE.BASE_NUM_FILTER = 16
__C.MODEL.DECONVBASELINE.USE_3D = True
__C.MODEL.DECONVBASELINE.ENCODER = "separate"
__C.MODEL.DECONVBASELINE.BN = True
__C.MODEL.DECONVBASELINE.BN_GLOBAL_STATS = False
__C.MODEL.DECONVBASELINE.COMPAT = edict() # Compatibility flags to recover behavior of previous versions
__C.MODEL.DECONVBASELINE.COMPAT.CONV_INSTEADOF_FC_IN_ENCODER = False # Until 6th May 2017
__C.MODEL.DECONVBASELINE.FC_BETWEEN_ENCDEC = 0

__C.MODEL.TRAIN = edict()
__C.MODEL.TRAIN.BATCH_SIZE = 3
__C.MODEL.TRAIN.TBPTT = False
__C.MODEL.TRAIN.OPTIMIZER = "adam"
__C.MODEL.TRAIN.LR = 1E-4
__C.MODEL.TRAIN.GAMMA1 = 0.9   # Used in RMSProp
__C.MODEL.TRAIN.BETA1 = 0.5    # When using ADAM, momentum is called beta1
__C.MODEL.TRAIN.EPS = 1E-8
__C.MODEL.TRAIN.MIN_LR = 1E-6
__C.MODEL.TRAIN.GRAD_CLIP = 50.0
__C.MODEL.TRAIN.WD = 0
__C.MODEL.TRAIN.MAX_ITER = 180000
__C.MODEL.VALID_ITER = 5000
__C.MODEL.SAVE_ITER = 15000
__C.MODEL.TRAIN.LR_DECAY_ITER = 10000
__C.MODEL.TRAIN.LR_DECAY_FACTOR = 0.7
__C.MODEL.TRAIN.LR_STEP = 10000

__C.MODEL.TEST = edict()
__C.MODEL.TEST.FINETUNE = True
__C.MODEL.TEST.MAX_ITER = 1      # Number of samples to generate in testing mode
__C.MODEL.TEST.MODE = "online"    # Can be `online` or `fixed`
__C.MODEL.TEST.DISABLE_TBPTT = True
__C.MODEL.TEST.ONLINE = edict()
__C.MODEL.TEST.ONLINE.OPTIMIZER = "adagrad"
__C.MODEL.TEST.ONLINE.LR = 1E-4
__C.MODEL.TEST.ONLINE.FINETUNE_MIN_MSE = 0.0
__C.MODEL.TEST.ONLINE.GAMMA1 = 0.9    # Used in RMSProp
__C.MODEL.TEST.ONLINE.BETA1 = 0.5     # Used in ADAM!
__C.MODEL.TEST.ONLINE.EPS = 1E-6
__C.MODEL.TEST.ONLINE.GRAD_CLIP = 50.0
__C.MODEL.TEST.ONLINE.WD = 0

def _merge_two_config(user_cfg, default_cfg):
    """ Merge user's config into default config dictionary, clobbering the
        options in b whenever they are also specified in a.
        Need to ensure the type of two val under same key are the same
        Do recursive merge when encounter hierarchical dictionary
    """
    if type(user_cfg) is not edict:
        return
    for key, val in user_cfg.items():
        # Since user_cfg is a sub-file of default_cfg
        if not key in default_cfg:
            raise KeyError('{} is not a valid config key'.format(key))

        if (type(default_cfg[key]) is not type(val) and
                default_cfg[key] is not None):
            if isinstance(default_cfg[key], np.ndarray):
                val = np.array(val, dtype=default_cfg[key].dtype)
            else:
                raise ValueError(
                     'Type mismatch ({} vs. {}) '
                     'for config key: {}'.format(type(default_cfg[key]),
                                                 type(val), key))
        # Recursive merge config
        if type(val) is edict:
            try:
                _merge_two_config(user_cfg[key], default_cfg[key])
            except:
                print('Error under config key: {}'.format(key))
                raise
        else:
            default_cfg[key] = val


def cfg_from_file(file_name, target=__C):
    """ Load a config file and merge it into the default options.
    """
    import yaml
    with open(file_name, 'r') as f:
        print('Loading YAML config file from %s' %f)
        yaml_cfg = edict(yaml.load(f))

    _merge_two_config(yaml_cfg, target)


def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items(), flow_style=False)

    def _ndarray_representer(dumper, data):
        return dumper.represent_list(data.tolist())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    OrderedDumper.add_representer(edict, _dict_representer)
    OrderedDumper.add_representer(np.ndarray, _ndarray_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def save_cfg(dir_path, source=__C):
    cfg_count = 0
    file_path = os.path.join(dir_path, 'cfg%d.yml' %cfg_count)
    while os.path.exists(file_path):
        cfg_count += 1
        file_path = os.path.join(dir_path, 'cfg%d.yml' % cfg_count)
    with open(file_path, 'w') as f:
        logging.info("Save YAML config file to %s" %file_path)
        ordered_dump(source, f, yaml.SafeDumper, default_flow_style=None)


def load_latest_cfg(dir_path, target=__C):
    import re
    cfg_count = None
    source_cfg_path = None
    for fname in os.listdir(dir_path):
        ret = re.search('cfg(\d+)\.yml', fname)
        if ret != None:
            if cfg_count is None or (int(re.group(1)) > cfg_count):
                cfg_count = int(re.group(1))
                source_cfg_path = os.path.join(dir_path, ret.group(0))
    cfg_from_file(file_name=source_cfg_path, target=target)