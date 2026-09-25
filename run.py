print("initiating")
import sys

a = ["--gpu", "1", "--epoch", "100", "--batch_size", "4", "--lr", "5e-5"]
# a = ["--test", "--weight_path", "best.pth","--gpu", "1"]
# a.extend(["--debug"])
sys.argv.extend(a)

import os
import torch

from run.Args import args
from run.train import run
from model.model import Multi_view_Knee
from data.dataloader import test_ds_dict
from val_with_save import val_with_save, show_CAM
import random
import numpy as np

seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

print("running")
if args.test:
    assert args.weight_path != "", "Please specify the weight path"
    model = Multi_view_Knee()
    model_file = args.weight_path
    model.load_state_dict(torch.load(model_file), strict=False)
    model = model.float()
    test_dataset = test_ds_dict['center3']
    # show_CAM(net=model, dataset=test_dataset)
    val_with_save(model, test_dataset)
else:
    run()