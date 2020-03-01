# import basic packages
import time, os, sys, torch, argparse
import os.path as osp
import numpy as np
from pprint import PrettyPrinter

# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

# import train and test from apis
from detection_tutorial.Apis.train import train
from detection_tutorial.Apis.test import test
from detection_tutorial.Apis.config import *

# BEFORE_MODE = "| Finish Setting Configs For Mode : {} ..."
# AFTER_MODE = "| Finish {} Mode ! "

def get_args() :
	# define parser and some variables
	parser = argparse.ArgumentParser(description=\
		'Single Shot MultiBox Detector Training for Custom Dataset With Pytorch')
	parser.add_argument('--config_file', default="detection_tutorial/Config/ssd_config.py", help="Specify Your Model's Config file !")
	
	args = parser.parse_args()

	# get all config file's dict
	model_config, training_config, testing_config, dataset_config = get_config_dict(args.config_file)

	return model_config, training_config, testing_config, dataset_config
	
# =================================================================

if __name__ == "__main__" :

	model_config, training_config, testing_config, dataset_config = get_args()

	# print("| Ready For Training ... ")
	# # train stage, 包括跑 numbers of epoch，跟驗證
	# current_mode = training_config["status"]
	# if current_mode:

	# 	# declare config's variables
	# 	checkpoint_result, optimizer_dict, detection, multiboxloss, validation, logger, data_loader, epoch, device, map_overlap = training_config_declare(model_config,training_config,dataset_config, "TRAIN")
		
	# 	# before train or test mode
	# 	print(BEFORE_MODE.format(training_config["mode"]))
		
	# 	# train
	# 	train(checkpoint_result, optimizer_dict, detection, multiboxloss, validation, logger, data_loader, epoch, device, map_overlap)

	# 	# after train or test mode
	# 	print(AFTER_MODE.format(training_config["mode"]))

	# 	# release sources
	# 	del checkpoint_result, optimizer_dict, detection, multiboxloss, validation, logger, data_loader, epoch

	# else :
	# 	print("| Skip Training / Validation ... ")

	# 跑完訓練以及驗證後，來觀察測試結果
	print("| Ready For Testing ... ")
	current_mode = testing_config["status"]
	if current_mode :
		
		# declare config's variables
		checkpoint_result, data_loader, outputs, label_file, detection, device = testing_config_declare(model_config,testing_config,dataset_config, "TEST")

		# before train or test mode
		print(BEFORE_MODE.format(testing_config["mode"]))
		
		test(checkpoint_result, data_loader["test"], outputs, label_file, detection, device)

		# after train or test mode
		print(AFTER_MODE.format(testing_config["mode"]))

		# release sources
		del checkpoint_result, data_loader, outputs, label_file, detection, device
	
	else :
		print("| Skip Testing ... ")

	print("| Mission Complete !!!")
	
	
