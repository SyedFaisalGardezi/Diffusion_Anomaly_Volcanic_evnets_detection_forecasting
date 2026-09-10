#!/usr/bin/env python
# coding: utf-8

import sys

import pkgutil
if pkgutil.find_loader("cfg_ddim"):
    print("✅ Package 'cfg_ddim' found")
else:
    print("❌ Package 'cfg_ddim' NOT found! Check `sys.path.append()` or `__init__.py`")


import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import labml

from labml import experiment
from labml.configs import option
from cfg_ddim import experiment_cfg_ddim
from cfg_ddim.experiment_cfg_ddim import Configs
# 
# from cfg.src.cfg_ddim import experiment_cfg_ddim
# from cfg.src.cfg_ddim.experiment_cfg_ddim import Configs




# In[9]:

try:
    print("LabML Location:", labml.__file__)
    print("LabML Experiment Location:", experiment.__file__)
    print("LabML Configs Location:", labml.configs.__file__)
    print("cfg ddim Experiment Location:", experiment_cfg_ddim.__file__)
except AttributeError:
    print("Module path issue detected! Ensure 'cfg_ddim' is a valid package.")


# # ### Create an experiment


experiment.create(name="diffuse", writers={'screen'})


# ### Configurations


configs = Configs()

# ✅ Store original configurations before updating
original_configs = vars(Configs()).copy()


# Debugging: Print configs to ensure initialization is successful
# print(f"Configs Initialized: {configs}")


# Set experiment configurations and assign a configurations dictionary to override configurations

# In[18]:

exp = "whakaari"   # checkpoint name; e.g. whakaari / ruapehu / pavlof / swat / wadi / synthetic
dataset ='Tremor'  #  'SWAT'  #        'MNIST', 'Synthetic' , 'Yahoo' , 'WADI' # 'SWAT'   #  #
experiment.configs(configs, {
    # 'dataset': dataset,
    # 'image_channels': 5, #5, # 123,  # 3 channels for rsam, mf, hf,123
    'epochs': 30000,
    "n_samples": 10,## its batch size
    "chunk_size": 50,  #100,
    "guidance_scale" : 3.0,
    "lambda_max" : 500,
    "T_MIN" : 10,
    "ALPHA" : 1.0,
    "S" : 2.0,
    "exp" : exp,
    "save_model_path":f"./cfg/src/cfg_ddim/model_weights/{exp}.pth",
})


if dataset == 'WADI':
    print("the dataset selected is WADI")
    experiment.configs(configs, {
        'dataset': 'WADI',
        'image_channels': 123,
        "dataset_path":"./data_external/time_series_datasets/WaDi/WADI.A1_9 Oct 2017/df_normal.csv"
    })

elif dataset == 'SWAT':
    print("the dataset selected is 'SWAT'")
    experiment.configs(configs, {
        'dataset': 'SWAT',
        'image_channels': 51,
        "dataset_path":"./cfg/data/processed/SWat/robust_1_1_train.npy"
    })

elif dataset == 'Tremor':
    print("the dataset selected is Tremor")
    experiment.configs(configs, {
        'dataset': 'Tremor',
        'image_channels': 5,
        # "dataset_path": "./cfg/data/raw/Whakaari_clean_tremor_data_mydata_ssam_norm_0_1.csv",
        # "dataset_path": "./cfg/data/processed/previous_paper_data/Whakaari_eruption_data_z_percen_m11_clean.csv",
        "dataset_path": "./cfg/data/raw/Ruapehu_eruption_data_z_percen_m11_clean.csv",
        # "dataset_path": "./cfg/data/raw/outside_NZ/PVV_clean_only_seismic_data_with_ssam_norm_z_percen_m11.csv"
    })
    
elif dataset == 'Yahoo':
    print("the dataset selected is Yahoo")
    experiment.configs(configs, {
        'dataset': 'Yahoo',
        'image_channels': 5,
        "dataset_path":"./cfg/data/processed/yahoo/learningData_yahoo_train.csv"
    })
elif dataset == 'Synthetic':
    print("the dataset selected is Synthetic")
    experiment.configs(configs, {
        'dataset': 'Synthetic',
        'image_channels': 5,
        "dataset_path": "./cfg/data/processed/synth_data/pattern_seasonal/train.csv"
    })




# print(f"configs.dataset in training.py {configs.dataset} ")
# ### Initializ

configs.init()
print(f"configs.dataset in training.py {configs.dataset} at location {configs.dataset_path}")

# ✅ Save final configurations to a text file
config_path = f"./cfg/src/cfg_ddim/logs/final_configs_{configs.exp}.txt"
os.makedirs(os.path.dirname(config_path), exist_ok=True)

with open(config_path, "w") as f:
    f.write("Final Configurations for Training\n")
    f.write("=" * 50 + "\n")
    for key, value in vars(configs).items():
        f.write(f"{key}: {value}\n")

print(f"✅ Final configs saved to {config_path}")

print(f"training for dataset {configs.dataset}")
# # ✅ Compare original vs modified configurations and print only the changes
# print("✅ Modified Configurations:")
# for key, value in vars(configs).items():
#     if key not in original_configs or original_configs[key] != value:
#         print(f"🔹 {key}: {value}")


# Set PyTorch models for loading and saving
experiment.add_pytorch_models({'eps_model': configs.eps_model})


# Start the experiment
with experiment.start():
    configs.run()
    
    
    
    
    
