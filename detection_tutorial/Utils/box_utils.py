import time, os, sys, torch, argparse, random
import os.path as osp
import numpy as np

# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# encode the variances from the priorbox layers into the ground truth boxes
def center_encode(matched_true_box, prior_box, center_variance, size_variance) :

	"""
	The conversion of "encode" and "decode":
		1. matched_true_box_encode_center * center_variance = (matched_true_box_center - prior_center)/prior_hw
		2. exp(matched_true_box_encode_hw * size_variance) = (matched_true_box_hw)/prior_hw
		=> We do it in the inverse direction here.

		3. variance :
			a. For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
			b. For the size coordinates, scale by the size of the prior box, and convert to the log-space.
		4. encode / decode 算的是 offsets，而我們訊練的就是這個 offsets，論文中 loss 部分有提到
	Args :
		1. matched_true_box (batch_size, num_priors, 4) : the regression output of SSD. It will contain the outputs as well.
		2. ✨ prior_box (num_priors, 4) or (batch_size or 1, num_priors, 4) : prior boxes ... ✨
		=> priors can have one dimension less than locations
		3. center_variance : a float used to change the scale of center.
		4. size_variance : a float used to change of scale of size.
	Returns :
		boxes:  priors: [[center_x, center_y, w, h]]. 
		=> All the values are relative to the image size.
	"""
	
	# priors can have one dimension less，上面註解有提到
	# so we need to unsqueeze the first dimension
	if prior_box.dim() + 1 == matched_true_box.dim():
		prior_box = prior_box.unsqueeze(0)

	# 計算 encode coordinates using variance
	matched_true_box_encode_center = \
		(matched_true_box[:, :2] - prior_box[:, :2]) / (prior_box[:, 2:] * center_variance)
	matched_true_box_encode_hw = \
		torch.log(matched_true_box[:, 2:] / prior_box[:, 2:]) / size_variance

	# combine center and height and weight
	encode_matched_location = \
		torch.cat([matched_true_box_encode_center, matched_true_box_encode_hw],\
			dim=matched_true_box.dim() - 1)

	return encode_matched_location

# decode the variances from the priorbox layers into the ground truth boxes
# 跟上面 "encode" 類似，是反過來
def center_decode(matched_true_box, prior_box, center_variance, size_variance) :

	# priors can have one dimension less，上面註解有提到
	# so we need to unsqueeze the first dimension
	if prior_box.dim() + 1 == matched_true_box.dim():
		prior_box = prior_box.unsqueeze(0)

	# 計算 encode coordinates using variance
	matched_true_box_decode_center = \
		matched_true_box[ :, :2] * center_variance * prior_box[ :, 2:] + prior_box[ :, :2]
	matched_true_box_decode_hw = \
		torch.exp(matched_true_box[ :, 2:] * size_variance) * prior_box[ :, 2:]

	# combine center and height and weight
	decode_matched_location = \
		torch.cat([matched_true_box_decode_center, matched_true_box_decode_hw],\
			dim=matched_true_box.dim() - 1)

	return decode_matched_location

# transfer from cxcy to xy coordinates
def cxcy_to_xy(boxes_cxcy) :

	'''
	Conversion : From center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max)
	'''

	xmin_ymin = boxes_cxcy[:,:2] - (boxes_cxcy[:,2:] / 2)
	xmax_ymax = boxes_cxcy[:,:2] + (boxes_cxcy[:,2:] / 2)

	# cat array along axis one
	# 因為預設是 0，但要把 x,c,w,h 在每個 row concat 在一起 [xmin_ymin],[xmax_ymax] => [xmin_ymin,xmax_ymax]
	final_xy = torch.cat([xmin_ymin,xmax_ymax], boxes_cxcy.dim()-1)

	return final_xy

# transfer from xy to cxcy coordinates
def xy_to_cxcy(boxes_xy) :

	'''
	Conversion : From boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h)
	'''

	cx_cy = (boxes_xy[:, 2:] + boxes_xy[:,:2]) / 2
	w_h = boxes_xy[:, 2:] - boxes_xy[:,:2]

	# cat array along axis one
	# 因為預設是 0，但要把 x,c,w,h 在每個 row concat 在一起 [x,y],[w,h] => [x,y,w,h]
	final_cxcy = torch.cat([cx_cy,w_h], boxes_xy.dim()-1)

	return final_cxcy

# find overlap (intersection) between set_1 and set_2
def find_intersection(set_1, set_2) :

	"""
	目的：
	=> Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
	參數：
	1. param set_1: set 1, a tensor of dimensions (n1, 4)
	2. param set_2: set 2, a tensor of dimensions (n2, 4)
	3. return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
	"""

	# PyTorch auto-broadcasts singleton dimensions
	lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
	upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
	intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
	return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)

# compute jaccard overlap
def jaccard_overlap(set_1, set_2) :

	'''
	目的：
	=> Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
	參數：
	1. param set_1: set 1, a tensor of dimensions (n1, 4)
	2. param set_2: set 2, a tensor of dimensions (n2, 4)
	3. return : Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
	'''

	# Find intersections
	intersection = find_intersection(set_1, set_2)  # 維度：(set_1, set_2)

	# Find areas of each box in both sets
	areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # 維度：(set_1)
	areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # 維度：(set_2)

	# Find the union
	union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # 維度：(n1, n2)

	return intersection / union  # (n1, n2)

# match function：看哪些 anchor box 可以跟 ground truth 匹配，
# 並且回傳 overlap 大於 threshold 的 anchor box
def match(threshold, predict_boxes, predict_score, gt_boxes, gt_labels, prior_boxes, variance, device) :

	# the number of current image's box
	number_objects = gt_boxes.size(0)

	# convert center-coordinates to corner-coordinates
	prior_boxes_xy = cxcy_to_xy(prior_boxes).to(device)
	prior_boxes = prior_boxes.to(device)

	overlap = jaccard_overlap(gt_boxes, prior_boxes_xy)

	# 我們這邊用 Bipartite Matching（有點複雜哦！）
	# 在 overlap 中，我們分別對兩個維度找出最大值跟其 index，原因在於：（有點類似雙重比對）
	# 1. 我們先找出對於每個 ground truth 中，哪些 prior boxes 的涵蓋面積最大 -- (number_objects, 1) best prior for each ground truth
	# 2. 接著再找出對於每個 prior boxes 中，哪些 ground truth 的涵蓋面積最大 -- (1 ,number_priors) best ground truth for each prior
	best_prior_overlap, best_prior_index = overlap.max(dim=1)  # (number_objects)
	best_truth_overlap, best_truth_index = overlap.max(dim=0)  # (number_priors)
	
	# 到這邊有兩個問題： We don't want a situation where an object is not represented in our positive (non-background) priors -
	# 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
	# 2. All priors with the object may be assigned as background based on the threshold (0.5).
	# => 情境劇：
	# 假想一種極端情況, 所有的 priorbox 與某個 GT 的 overlap = 1 , 而其他 groudtruth_box 也分别有一個 overlap 最高的priorbox
	# 但是這樣就會使得所有 priorbox 都和 GT 匹配, 為避免這樣, 
	# 我們反過來找出針對每個 groudtruth_box 具有最高 overlap 的 center_encode => best_truth_index[best_prior_index[j]] = j
	# 其餘（其他 priorbox) 則維持一樣，其目的為防止某個 groudtruth_box 没有匹配的 priorbox
	best_truth_index[best_prior_index] = torch.LongTensor(range(number_objects)).to(device)
	
	# To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
	best_truth_overlap[best_prior_index] = 1.
	
	# 根據匹配後的結果，找出 priorbox 對應 truth label
	best_truth_label = gt_labels[best_truth_index]
	best_truth_label[best_truth_overlap < threshold] = 0
	
	# Store
	true_classes = best_truth_label
	true_locs = center_encode(xy_to_cxcy(gt_boxes[best_truth_index]), prior_boxes, variance["center"], variance["size"])
	# print(torch.max(gt_boxes[best_truth_index]),torch.min(gt_boxes[best_truth_index]))
	return true_locs, true_classes

# get label name
def get_label(label_file) :

	label_map = {}
	label_color = {}
	labels = open(label_file, 'r')
	for line in labels:
		ids = line.split(',')
		label_map[int(ids[0])] = ids[1].split('\n')[0]
		label_color[int(ids[0])] = ids[2].split('\n')[0]
	
	return label_map, label_color

# Calculate mAP ：參考大大們
def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, label_file, device):
	"""
	Calculate the Mean Average Precision (mAP) of detected objects.

	See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

	:param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
	:param det_labels: list of tensors, one tensor for each image containing detected objects' labels
	:param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
	:param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
	:param true_labels: list of tensors, one tensor for each image containing actual objects' labels
	:param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
	:return: list of average precisions for all classes, mean average precision (mAP)
	"""
	assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
		true_labels) == len(
		true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
	label_map, label_color = get_label(label_file)
	n_classes = len(label_map)

	# Store all (true) objects in a single continuous tensor while keeping track of the image it is from
	true_images = list()
	for i in range(len(true_labels)):
		true_images.extend([i] * true_labels[i].size(0))
	true_images = torch.LongTensor(true_images).to(
		device)  # (n_objects), n_objects is the total no. of objects across all images
	true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
	true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
	true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

	assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

	# Store all detections in a single continuous tensor while keeping track of the image it is from
	det_images = list()
	for i in range(len(det_labels)):
		det_images.extend([i] * det_labels[i].size(0))
	det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
	det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
	det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
	det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

	assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

	# Calculate APs for each class (except background)
	average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
	for c in range(1, n_classes):
		# Extract only objects with this class
		true_class_images = true_images[true_labels == c]  # (n_class_objects)
		true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
		true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
		n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

		# Keep track of which true objects with this class have already been 'detected'
		# So far, none
		true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
			device)  # (n_class_objects)

		# Extract only detections with this class
		det_class_images = det_images[det_labels == c]  # (n_class_detections)
		det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
		det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
		n_class_detections = det_class_boxes.size(0)
		if n_class_detections == 0:
			continue

		# Sort detections in decreasing order of confidence/scores
		det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
		det_class_images = det_class_images[sort_ind]  # (n_class_detections)
		det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

		# In the order of decreasing scores, check if true or false positive
		true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
		false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
		for d in range(n_class_detections):
			this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
			this_image = det_class_images[d]  # (), scalar

			# Find objects in the same image with this class, their difficulties, and whether they have been detected before
			object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
			object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
			# If no such object in this image, then the detection is a false positive
			if object_boxes.size(0) == 0:
				false_positives[d] = 1
				continue

			# Find maximum overlap of this detection with objects in this image of this class
			overlaps = jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
			max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

			# 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
			# In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
			original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
			# We need 'original_ind' to update 'true_class_boxes_detected'

			# If the maximum overlap is greater than the threshold of 0.5, it's a match
			if max_overlap.item() > 0.5:
				# If the object it matched with is 'difficult', ignore it
				if object_difficulties[ind] == 0:
					# If this object has already not been detected, it's a true positive
					if true_class_boxes_detected[original_ind] == 0:
						true_positives[d] = 1
						true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
					# Otherwise, it's a false positive (since this object is already accounted for)
					else:
						false_positives[d] = 1
			# Otherwise, the detection occurs in a different location than the actual object, and is a false positive
			else:
				false_positives[d] = 1

		# Compute cumulative precision and recall at each detection in the order of decreasing scores
		cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
		cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
		cumul_precision = cumul_true_positives / (
				cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
		cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

		# Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
		recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
		precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
		for i, t in enumerate(recall_thresholds):
			recalls_above_t = cumul_recall >= t
			if recalls_above_t.any():
				precisions[i] = cumul_precision[recalls_above_t].max()
			else:
				precisions[i] = 0.
		average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

	# Calculate Mean Average Precision (mAP)
	mean_average_precision = average_precisions.mean().item()

	# Keep class-wise average precisions in a dictionary
	average_precisions = {label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

	return average_precisions, mean_average_precision
