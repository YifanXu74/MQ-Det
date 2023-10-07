import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino_new.datasets.transforms as T
from groundingdino_new.models import build_model
from groundingdino_new.util import box_ops
from groundingdino_new.util.slconfig import SLConfig
from groundingdino_new.util.utils import clean_state_dict, get_phrases_from_posmap

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    return model