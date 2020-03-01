# import basic packages
import time, os, sys, functools
from importlib import import_module
import os.path as osp

# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader

# import model
from detection_tutorial.Model.Detector.single_stage_detector import single_stage_detector as ssd_detector

# import checkpointer
from detection_tutorial.Utils.checkpoint import checkpointer
from detection_tutorial.Datasets.custom import CustomDataset
from detection_tutorial.Datasets.augmentation import Augmentation
from detection_tutorial.Utils.detect import Detection
from detection_tutorial.Utils.loss import MultiBoxLoss

#  declare global strings
BEFORE_MODE = "| Finish Setting Configs For Mode : {} ..."
AFTER_MODE = "| Finish {} Mode ! "

def get_dataloader(current_dataset_name, data_root, label_file, batch_size, number_workers, current_mode ,current_dataloader_mode, augmentation_config, input_size) :

	print("| Preparing Your Dataset : {} {} dataset ...".format(current_dataset_name, current_dataloader_mode))

	if augmentation_config["status"] :
		
		augmentation = Augmentation(augmentation_config, current_dataloader_mode)

	else :

		augmentation = None

	# create dataset
	dataset = CustomDataset(current_mode, data_root, label_file, augmentation, input_size)
	data_collater = functools.partial(dataset.collate_fn, mode=current_mode)

	# make a dataset
	dataloader = DataLoader(dataset, batch_size, num_workers=number_workers, shuffle=True, \
		collate_fn=data_collater, pin_memory=True)

	return dataloader

def get_config_dict(filename) :

	filename = osp.abspath(osp.expanduser(filename))
	print("| Loading All Configs in {}...".format(filename))
	
	if not osp.isfile(filename):
		raise FileNotFoundError('"{}" does not exist'.format(filename))
	if filename.endswith('.py'):
		module_name = osp.basename(filename)[:-3]
		if '.' in module_name:
			raise ValueError('Dots are not allowed in config file path.')
		config_dir = osp.dirname(filename)
		sys.path.insert(0, config_dir)
		mod = import_module(module_name)
		sys.path.pop(0)
		cfg_dict = {
			name: value
			for name, value in mod.__dict__.items()
			if not name.startswith('__')
		}

		return cfg_dict["ssd_model"], cfg_dict["training_config"], cfg_dict["testing_config"], cfg_dict["dataset"]

def get_checkpoint_result(model_config, training_config, current_mode, device) :

	# declare return result
	checkpoint_result = {}

	# declare model and initialize weight
	model = ssd_detector(model_config, training_config, current_mode, device) 

	# declare optimizers
	lr_rate = training_config["learning_rate"]
	weight_decay = training_config["weight_decay"]
	gamma = training_config["gamma"]
	gradient_clip = training_config["gradient_clip"]
	optimizer_kind = training_config["optimizer"]
	learning_steps = training_config["lr_steps"]
	momentum = training_config["momentum"]
	detection_endecode_variance = model_config["prior_box"]["variance"]

	# declare detection object for detection job
	detection = Detection(training_config["validation"]["config"], model.anchor_head.prior_box, model.anchor_head.num_classes, model.device, detection_endecode_variance)
	
	# declare loss object for computing loss job
	multiboxloss = MultiBoxLoss(model_config["loss_function"], model.anchor_head.center_variance, \
		model.anchor_head.size_variance, model.anchor_head.prior_box, model.anchor_head.num_classes, model.device)
	
	if optimizer_kind == "adam" :
		optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
	elif optimizer_kind == "SGD" :
		
		# Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
		biases = list()
		not_biases = list()
		for param_name, param in model.named_parameters():
			if param.requires_grad:
				if param_name.endswith('.bias'):
					biases.append(param)
				else:
					not_biases.append(param)
		optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr_rate}, {'params': not_biases}],
									lr=lr_rate, momentum=momentum, weight_decay=weight_decay)
	else :
		raise TypeError("You Should Specify A Kind Of Optimizer !")

	# declare checkpointer (logger)
	checkpoint = training_config["checkpoint"]
	logger = checkpointer(model, checkpoint)
	if checkpoint["model_name"] != None :

		print("| Resuming Training ...")
		checkpoint_result = logger.load_train(optimizer, checkpoint, device)

	else :
		print("| Starting Training From Beginning ...")
		checkpoint_result = {
			"model" : model,
			"optimizer" : optimizer,
			"best_loss" : 10000000,
			"start_epoch" : 0
		}

	validation = training_config["validation"]

	# combine to optimizer's dict
	optimizer_dict = {
		"learning_steps" : learning_steps,
		"gamma" : gamma,
		"lr_rate" : lr_rate,
		"gradient_clip" : gradient_clip,

	}

	return checkpoint_result, optimizer_dict, detection, multiboxloss, validation, logger

def training_config_declare(model_config,training_config,dataset_config,current_mode) :

	# set bench mark
	torch.backends.cudnn.benchmark = True
	# for validation initial
	map_overlap = 0

	print("| Ready For Training ... ")

	# declare device
	print("| Loading Device ...")
	if training_config["device"]["GPU"] >= 1 :
		if torch.cuda.is_available() :
			device = torch.device("cuda")
		else :
			device = torch.device("cpu")
			print("| Only Can Use CPU !!!")
	elif training_config["device"]["GPU"] == 0 :
		device = torch.device("cpu")
	else :
		raise TypeError("Should be either 'GPU' or 'CPU' !!!")

	# declare epoch and batch size and num-workers...
	input_size = model_config["input_size"]
	print("| Loading Training Configs for SSD-{}...".format(str(input_size)))
	epoch = training_config["epoch"]
	batch_size = training_config["batch_size"]
	number_workers = training_config["number_workers"]

	# declare model and initialize weight
	print("| Loading Model and Optimizer ...")
	checkpoint_result, optimizer_dict, detection, multiboxloss, validation, logger = get_checkpoint_result(model_config, training_config, current_mode, device)

	# declare dataloader
	print("| Loading Dataset ... ")
	current_dataset_name = dataset_config["dataset_name"]
	label_file = dataset_config["label_file"]
	train_data_file = dataset_config["training_data"]
	augmentation = dataset_config["dataloader"]["transformation"]
	dataloader_train = get_dataloader(current_dataset_name, train_data_file, label_file, batch_size, number_workers, "train", "TRAIN", augmentation, input_size)
	if training_config["validation"]["status"] :

		map_overlap = training_config["validation"]["mAP_overlap"]
		val_data_file = dataset_config["validation_data"]
		dataloader_val = get_dataloader(current_dataset_name, val_data_file, label_file, batch_size, number_workers, "validation", "TEST", augmentation, input_size)
	else :
		dataloader_val = None
	
	# combine all data_loader
	data_loader = {
		"train" : dataloader_train,
		"validation" : dataloader_val,
		"label_file" : label_file
	}

	return checkpoint_result, optimizer_dict, detection, multiboxloss, validation, logger, data_loader, epoch, device, map_overlap

def testing_config_declare(model_config,test_config,dataset_config,current_mode) :

	print("| Ready For Testing ... ")

	# declare device
	print("| Loading Device ...")
	if test_config["device"]["GPU"] >= 1 :
		if torch.cuda.is_available() :
			device = torch.device("cuda")
		else :
			device = torch.device("cpu")
			print("| Only Can Use CPU !!!")
	elif test_config["device"]["GPU"] == 0 :
		device = torch.device("cpu")
	else :
		raise TypeError("Should be either 'GPU' or 'CPU' !!!")

	# declare model and initialize weight
	model = ssd_detector(model_config, test_config, current_mode, device) 

	# declare other configs
	detection_endecode_variance = model_config["prior_box"]["variance"]
	detection = Detection(test_config["config"], model.anchor_head.prior_box, \
		model.anchor_head.num_classes, model.device, detection_endecode_variance)

	# declare checkpointer
	checkpoint = test_config["checkpoint"]
	logger = checkpointer(model, checkpoint)
	if checkpoint["model_name"] != None :
	
		print("| Resuming Testing ...")
		print("| Main Work is : {} ...".format(test_config["main_work"]))
		checkpoint_result = logger.load(checkpoint)

	else :
		raise TypeError("| It Should Contains Checkpoint Model in Testing Mode  ...")

	# declare dataloader
	print("| Loading Dataset ... ")
	current_dataset_name = dataset_config["dataset_name"]
	test_data_file = dataset_config["testing_data"]
	label_file = dataset_config["label_file"]
	augmentation = dataset_config["dataloader"]["transformation"]
	batch_size = test_config["batch_size"]
	number_workers = test_config["number_workers"]
	input_size = model_config["input_size"]
	dataloader_test = get_dataloader(current_dataset_name, test_data_file, label_file, batch_size, number_workers, "test", "TEST", augmentation, input_size)

	# combine all data_loader
	data_loader = {
		"test" : dataloader_test
	}

	# declare outputs directory
	outputs = test_config["outputs"]

	# create directory if not exists
	if not os.path.exists(outputs):
		os.makedirs(outputs)

	if not os.path.exists(outputs + "detection_original/"):
		os.makedirs(outputs + "detection_original/")

	if not os.path.exists(outputs + "detection_result/"):
		os.makedirs(outputs + "detection_result/")

	del logger

	return checkpoint_result, data_loader, outputs, label_file, detection, device