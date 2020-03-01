# import basic packages
import time, os, sys, torch, argparse
import os.path as osp
import numpy as np
from pprint import PrettyPrinter

class checkpointer() :

	# basic setup
	def __init__(self, model, checkpoint) :

		self.model = model
		self.checkpoint = checkpoint
		self.base_file = self.checkpoint["directory"]
		self.model_name = self.checkpoint["model_name"]
		self.log_file = self.base_file + "log.txt"

		# create directory
		if not os.path.exists(self.base_file):
			os.makedirs(self.base_file)

		# train
		self.time_train = 0
		self.val_loss_loc_train = 0
		self.avg_loss_loc_train = 0
		self.sum_loss_loc_train = 0
		self.val_loss_conf_train = 0
		self.avg_loss_conf_train = 0
		self.sum_loss_conf_train = 0

		# validation
		self.time_val = 0
		self.val_loss_loc_val = 0
		self.avg_loss_loc_val = 0
		self.sum_loss_loc_val = 0
		self.val_loss_conf_val = 0
		self.avg_loss_conf_val = 0
		self.sum_loss_conf_val = 0
		self.count = 0

	# load model if using pre-trained model	
	def load_train(self, optimizer, train_checkpoint, device) :

		# load from model
		self.base_file = train_checkpoint["directory"]
		self.model_name = train_checkpoint["model_name"]

		other, ext = os.path.splitext(self.model_name)
		model_name = os.path.join(self.base_file,self.model_name)
		check_point_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
		
		# load model
		if ext == '.pkl' or '.pth' or '.pt':
			print('\n| Loading weights into state dict...')
			self.model.load_state_dict(check_point_dict["model"])
			self.model = self.model.to(device)
			print('| Finished!')
		else:
			print('| Error => Sorry only .pth, .pkl and .pt files supported.')

		# load optimizer
		self.optimizer = optimizer
		self.optimizer.load_state_dict(check_point_dict['optimizer'])

		# load others
		best_loss = check_point_dict["best_loss"]
		start_epoch = check_point_dict["start_epoch"]

		checkpoint_result = {
			"model" : self.model,
			"optimizer" : self.optimizer,
			"best_loss" : best_loss,
			"start_epoch" : start_epoch + 1
		}

		print("| Model Will Training From Epoch {} ; Current Loss : {} ...".format(start_epoch + 1,best_loss))

		return checkpoint_result

	# load model if using for testing mode
	def load(self, test_checkpoint) :

		# load from model
		self.base_file = test_checkpoint["directory"]
		self.model_name = test_checkpoint["model_name"]

		# check format
		other, ext = os.path.splitext(self.model_name)
		model_name = os.path.join(self.base_file,self.model_name)
		check_point_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
		
		# load model
		if ext == '.pkl' or '.pth' or '.pt':
			print('\n| Loading weights into state dict...')
			self.model.load_state_dict(check_point_dict["model"])
			print('| Finished!')
		else:
			print('| Error => Sorry only .pth, .pkl and .pt files supported.')

		# load others
		best_loss = check_point_dict["best_loss"]
		start_epoch = check_point_dict["start_epoch"]

		checkpoint_result = {
			"model" : self.model,
			"best_loss" : best_loss,
			"start_epoch" : start_epoch
		}

		return checkpoint_result

	# update information
	def update(self, val, kind, batch_size=1) :

		# count
		self.count += batch_size

		if kind == "time_train" :
			# time update
			self.time_train = val
		elif kind == "loc_train" :

			# location loss update
			self.val_loss_loc_train = val
			self.sum_loss_loc_train += val
			self.avg_loss_loc_train = self.sum_loss_loc_train / self.count

		elif kind == "conf_train" :

			# confidence loss update
			self.val_loss_conf_train = val
			self.sum_loss_conf_train += val
			self.avg_loss_conf_train = self.sum_loss_conf_train / self.count

		elif kind == "time_val" :
			# time update
			self.time_val = val
		elif kind == "loc_val" :

			# location loss update
			self.val_loss_loc_val = val
			self.sum_loss_loc_val += val
			self.avg_loss_loc_val = self.sum_loss_loc_val / self.count

		elif kind == "conf_val" :

			# confidence loss update
			self.val_loss_conf_val = val
			self.sum_loss_conf_val += val
			self.avg_loss_conf_val = self.sum_loss_conf_val / self.count

		else :

			raise TypeError("It Should Be Confidence / Location or Time !")

	# update model
	def update_model(self, model, optimizer, best_loss, total_loss, update_status, each_epoch) :

		print("| Updating Model ... ")
		state = {
			'start_epoch': each_epoch,
			'loss': total_loss,
			'best_loss': best_loss,
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict()
			}
		filename = 'SSD.pt'
		torch.save(state, self.base_file + filename)

		# If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
		if update_status:
			torch.save(state, self.base_file + "BEST_" + filename)

	# upadate log
	def update_log(self, txt_content) :
		
		with  open(self.log_file, 'a') as f :
			f.write(txt_content)