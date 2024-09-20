import os
from os.path import dirname, abspath
import logging
# Defaults
ROOT_DIR   = dirname(dirname(abspath(__file__))) # Project Root
HOME_DIR   = os.getenv("HOME") # User home dir
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'out')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
from .utils import launch_logger

import numpy as np

def initialize_seed(seed):
    np.random.seed(seed)

initialize_seed(42)