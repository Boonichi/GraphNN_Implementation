# System Packages
import argparse
import logging
from pathlib import Path
import time
import datetime
import os
import pickle
import json

from numba.core.errors import NumbaWarning
import warnings

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd

from configs import get_args_parser

from models.GAT import GAT

def main(args):
    print(args)
    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load Dataset
    if args.model == "GAT":
        model = GAT()
    
    model.to(device)
    
    

         
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Graph Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)