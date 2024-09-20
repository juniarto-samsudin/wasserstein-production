import sys, os.path

from .samples_loss import SamplesLoss
from .wasserstein_barycenter_images import ImagesBarycenter
from .sinkhorn_images import sinkhorn_divergence
from .__version__ import __version__

__all__ = sorted(["SamplesLoss, ImagesBarycenter"])

import numpy as np

def initialize_seed(seed):
    np.random.seed(seed)

initialize_seed(42)