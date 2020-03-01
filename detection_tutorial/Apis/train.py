# import basic packages
import time, os, sys, torch, argparse
import numpy as np
from pprint import PrettyPrinter

# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

# import box_utils
from detection_tutorial.Utils.box_utils import *

# declare global strings and PrettyPrinter()
BEFORE_MODE = "| Finish Setting Configs For Mode : {} ..."
AFTER_MODE = "| Finish {} Mode ! "
pp = PrettyPrinter()

def clip_gradient(optimizer, grad_clip):
	"""
	Clips gradients computed during backpropagation to avoid explosion of gradients.

	:param optimizer: optimizer with the gradients to be clipped
	:param grad_clip: clip value
	"""
	for group in optimizer.param_groups:
		for param in group['params']:
			if param.grad is not None:
				param.grad.data.clamp_(-grad_clip, grad_clip)

# adjust learning rate
def adjust_learning_rate(optimizer, scale):

	"""Sets the learning rate to the initial LR decayed by 10 at every
		specified step
	# Adapted from PyTorch Imagenet example:
	# https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * scale

	print("| DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


	return optimizer

def train_iter(dataloader, optimizer, model, train_logger,\
	 gradient_clip, each_epoch, final_epoch, multiboxloss, device) :

	# train mode
	model.train()

	# Batches
	for each_iter, (images, ground_truth_boxes, ground_truth_labels, _) in enumerate(dataloader) :

		# if each_iter == 2 :
		# 	 break

		# record beginning time
		start_time = time.time()

		# Move to default device
		images = images.to(device)  # (batch_size (N), 3, 300, 300)
		ground_truth_boxes = [b.to(device) for b in ground_truth_boxes]
		ground_truth_labels = [l.to(device) for l in ground_truth_labels]
		
		# Forward prop.
		predicted_boxes, predicted_scores = model(images)
		
		# 因為計算 loss，在論文提到：
		# we regress to offsets for the center (cx; cy) of the default bounding box (d)
		# and for its width (w) and height (h).
		# 因此在計算 loss 時，會先針對真實資料跟我們所定義的 prior box 找出 offset
		# 而我們要訓練的就是這個 offset，比直接訓練位置來好
		confidence_loss,  location_loss = multiboxloss(predicted_boxes, predicted_scores,\
				ground_truth_boxes, ground_truth_labels)
		total_loss = confidence_loss + location_loss

		# Backward prop.
		optimizer.zero_grad()
		total_loss.backward()

		# Clip gradients, if necessary
		# if gradient_clip :
		# 	clip_gradient(optimizer, True)

		# Update model
		optimizer.step()

		# update variables
		train_logger.update(time.time() - start_time, "time_train")
		train_logger.update(confidence_loss, "conf_train", batch_size=images.size(0))
		train_logger.update(location_loss, "loc_train", batch_size=images.size(0))

		print('| Training Epoch: [%d / %d][%d / %d]  /  '
				  'Data Time %.3f \n'
				  '| Confidence Loss %.4f /  '
				  'Coordinates Loss %.4f /  '
				  'Total Loss %.4f '
				  '\n===================================================================\n' %(each_epoch+1, final_epoch, each_iter, len(dataloader),\
										train_logger.time_train, train_logger.val_loss_conf_train, train_logger.val_loss_loc_train, \
										train_logger.val_loss_conf_train + train_logger.val_loss_loc_train), end="")

	return model, optimizer, multiboxloss

def validation_iter(dataloader, model, optimizer, best_loss, label_file, multiboxloss, \
	detection, device, train_logger, each_epoch, final_epoch, validation_iou_max) :

	# set for caculating mAP
	det_boxes = list()
	det_labels = list()
	det_scores = list()
	true_boxes = list()
	true_labels = list()
	true_difficulties = list()
	validation_loss = 0
	
	# eval mode
	model.eval()
	
	# with no gradient
	with torch.no_grad() :

		# Batches
		for each_iter, (images, ground_truth_boxes, ground_truth_labels, difficulties) in enumerate(dataloader) :

			# record beginning time
			start_time = time.time()

			# Move to default device
			images = images.to(device)  # (N, 3, 300, 300)
			ground_truth_boxes = [b.to(device) for b in ground_truth_boxes]
			ground_truth_labels = [l.to(device) for l in ground_truth_labels]
			difficulties = [d.to(device) for d in difficulties]

			# Forward prop.
			predicted_boxes, predicted_scores = model(images)

			# 因為計算 loss，在論文提到：
			# we regress to offsets for the center (cx; cy) of the default bounding box (d)
			# and for its width (w) and height (h).
			# 因此在計算 loss 時，會先針對真實資料跟我們所定義的 prior box 找出 offset
			# 而我們要訓練的就是這個 offset，比直接訓練位置來好
			confidence_loss,  location_loss = multiboxloss(predicted_boxes, predicted_scores,\
				ground_truth_boxes, ground_truth_labels)
			total_loss = confidence_loss + location_loss

			# update variables
			train_logger.update(time.time() - start_time, "time_val")
			train_logger.update(confidence_loss, "conf_val", batch_size=images.size(0))
			train_logger.update(location_loss, "loc_val", batch_size=images.size(0))

			# detection result
			all_images_boxes, all_images_labels, all_images_scores = detection(predicted_boxes, predicted_scores)

			det_boxes.extend(all_images_boxes)
			det_labels.extend(all_images_labels)
			det_scores.extend(all_images_scores)
			true_boxes.extend(ground_truth_boxes)
			true_labels.extend(ground_truth_labels)
			true_difficulties.extend(difficulties)

			print('| Validation Epoch: [%d / %d][%d / %d]  /  '
					'Data Time %.3f \n'
					'| Confidence Loss %.4f /  '
					'Coordinates Loss %.4f /  '
					'Total Loss %.4f '
					'\n===================================================================\n' %(each_epoch+1, final_epoch, each_iter+1, len(dataloader),\
											train_logger.time_val, train_logger.val_loss_conf_val, train_logger.val_loss_loc_val, \
											train_logger.val_loss_conf_val + train_logger.val_loss_loc_val), end="")
			
			txt_content = '| Validation Epoch: [{} / {}][{} / {}]  /  Data Time {} \n| Confidence Loss {} / Coordinates Loss {} /  Total Loss {} \n===================================================================\n'.format(each_epoch+1, final_epoch, each_iter+1, len(dataloader),\
											train_logger.time_val, train_logger.val_loss_conf_val, train_logger.val_loss_loc_val, \
											train_logger.val_loss_conf_val + train_logger.val_loss_loc_val)

			train_logger.update_log(txt_content)
			validation_loss += total_loss.item()

			del all_images_boxes, all_images_labels, all_images_scores
			del ground_truth_boxes, ground_truth_labels, difficulties
			del predicted_boxes, predicted_scores
			del total_loss, confidence_loss, location_loss
		
	# 檢查要不要 update model
	if validation_loss < best_loss : 
		best_loss = validation_loss
		update_status = True
	else :
		update_status = False
	
	train_logger.update_model(model, optimizer, best_loss, validation_loss, update_status, each_epoch)

	torch.cuda.empty_cache()

	# Calculate mAP
	APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, label_file, device)

	# Print AP for each class
	print("==============================")
	print("APs : =>")
	pp.pprint(APs)
	print("\nMean Average Precision : => {:.3f}".format(mAP))
	print("==============================")

	del APs, mAP
	torch.cuda.empty_cache()

	return model, optimizer, best_loss, multiboxloss

# train / validate the model
def train(checkpoint, optimizer_dict, detection, multiboxloss, validation, train_logger, \
	data_loader, total_epoch, device, map_overlap) :

	# declare some variables
	start_epoch = checkpoint["start_epoch"]
	final_epoch = start_epoch + total_epoch
	model = checkpoint["model"]
	optimizer = checkpoint["optimizer"]
	best_loss = checkpoint["best_loss"]
	lr_steps = optimizer_dict["learning_steps"]
	gamma = optimizer_dict["gamma"]
	lr_rate = optimizer_dict["lr_rate"]
	gradient_clip = optimizer_dict["gradient_clip"]
	multiboxloss = multiboxloss.to(device)
	model = model.to(device)

	# declare step index
	step_index = 0

	# Start Training
	print("| Start Training ... ")
	for each_epoch in range(start_epoch, final_epoch) :

		# if each_epoch in lr_steps:
		# 	print("| Now We Update Learning Rate...")
		# 	step_index += 1
		# 	optimizer = adjust_learning_rate(optimizer, 0.1)

		print("| Training Iter ... ")
		model, optimizer, multiboxloss = train_iter(data_loader["train"], optimizer, model, train_logger,\
			 gradient_clip, each_epoch, final_epoch, multiboxloss, device)
		print("| Finish Training Iter ... ")

		if validation["status"] :
			
			# validation iter
			if (each_epoch+1) % validation["number_epoch"] == 0 :

				print("| Validation Iter ... ")
				model, optimizer, best_loss, multiboxloss = validation_iter(data_loader["validation"], model, optimizer, best_loss,\
					 data_loader["label_file"], multiboxloss, detection, device, train_logger, each_epoch, final_epoch, map_overlap)
				print("| Finish Validation Iter... ")
			else :

				print("| Skip Validation ... ")
		else :
			print("| Skip Validation ... ")

		
	del model, optimizer, best_loss, checkpoint, multiboxloss
	del start_epoch, final_epoch
