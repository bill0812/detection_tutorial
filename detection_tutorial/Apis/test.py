# import basic packages
import time, os, sys, torch, argparse, cv2
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

BEFORE_MODE = "| Finish Setting Configs For Mode : {} ..."
AFTER_MODE = "| Finish {} Mode ! "

# save testing outputs
def output_save(images, all_images_boxes, all_images_labels, outputs, label_file, labe_map, label_color, count, dir_count, device) :

	for each_image in range(len(all_images_labels)) :

		if not os.path.exists(outputs + "detection_original/" + str(dir_count+1) + "/" ):
			os.makedirs(outputs + "detection_original/"+ str(dir_count+1))

		if not os.path.exists(outputs + "detection_result/" + str(dir_count+1) + "/" ):
			os.makedirs(outputs + "detection_result/"+ str(dir_count+1))

		if (count+1) % 101 == 0 :
			dir_count += 1
			count = 0
		
		# covert to numpy
		current_image = images[each_image].cpu().data.numpy()
		
		height, weight, channel = current_image.shape

		# save original image
		cv2.imwrite(outputs + "detection_original/" + str(dir_count+1) + "/" + str(count+1)+".jpg", current_image)

		# get label from detection result
		current_labels = [labe_map[l] for l in all_images_labels[each_image].to(device).tolist()]
		# set font 
		font = cv2.FONT_HERSHEY_SIMPLEX
	
		# If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
		if current_labels == ['__NOCLASS__']:

			# Just save original image
			cv2.imwrite(outputs + "detection_result/" + str(dir_count+1) + "/" + str(count+1)+".jpg", current_image)

		else :
			# Annotate
			# print(det_boxes)
			for i in range(all_images_boxes[each_image].size(0)):

				# Boxes
				box_location = all_images_boxes[each_image][i].tolist()
				this_box_label = all_images_labels[each_image][i].tolist()
				color = label_color[int(this_box_label)].split("-")
				xmin, ymin, xmax, ymax = box_location

				cv2.putText(current_image, current_labels[i],(int(xmin*weight), int(ymin*height-10)), font, 0.5, (0,255,0), 1)
				image = cv2.rectangle(current_image,(int(xmin*weight), int(ymin*height)),(int(xmax*weight), int(ymax*height)), (int(color[0]),int(color[1]),int(color[2])), 2)
			
			cv2.imwrite(outputs + "detection_result/" + str(dir_count+1) + "/" + str(count+1)+".jpg", current_image)

		# 累加數目
		count += 1

	return count, dir_count

# test iter
def test_iter(dataloader, model, outputs, label_file, detection, device) :

	# declare for stroing
	count = 0
	dir_count = 0

	label_map, label_color = get_label(label_file)

	# Batches
	for each_iter, (images, images_original) in enumerate(dataloader) :

		# print(images.size())
		# images = images_original

		# declare inputs and outputs
		images = Variable(images.to(device))

		predicted_boxes, predicted_scores = model(images)

		all_images_boxes, all_images_labels, all_images_scores = detection(predicted_boxes, predicted_scores)

		# Detect objects in SSD output
		count, dir_count = output_save(images_original, all_images_boxes, all_images_labels, outputs, label_file, label_map, label_color, count, dir_count, device)

		del all_images_boxes, all_images_labels, all_images_scores

		torch.cuda.empty_cache()

		print("| Finished {} Iter / {} Iter ...\n".format(each_iter+1,len(dataloader)), end="")

#  test model
def test(checkpoint_result, dataloader, outputs, label_file, detection, device) :
	
	# declare model
	model = checkpoint_result["model"]

	model = model.to(device)
	model.eval()

	print("| Testing Data And The Output Would Be '{}'...".format(outputs))

	print("| Start Testing ... ")

	test_iter(dataloader, model, outputs, label_file, detection, device)

	print("| Finish Testing ... ")

	del model




