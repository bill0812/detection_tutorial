# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops.boxes import nms

# import box_utils
from detection_tutorial.Utils.box_utils import *

class Detection(nn.Module) :

	# basic setup
	def __init__(self, detection_config, prior_boxes, num_classes, device, variance) :

		super(Detection,self).__init__()

		# define some variables
		self.min_score = detection_config["min_score_threshold"]
		self.max_overlap = detection_config["max_iou_threshold"]
		self.top_k = detection_config["top_k_result"]
		self.prior_boxes = prior_boxes
		self.num_classes = num_classes
		self.device = device
		self.variance = variance

	# compute the detection results
	def forward(self, predicted_boxes, predicted_scores) :
		
		# define batch size
		batch_size = predicted_boxes.size(0)
		
		self.prior_boxes = self.prior_boxes.to(self.device)

		# use softmax to get final scores
		predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, number_classes)

		# Lists to store final predicted boxes, labels, and scores for all images
		all_images_boxes = list()
		all_images_labels = list()
		all_images_scores = list()

		assert len(self.prior_boxes) == predicted_boxes.size(1) == predicted_scores.size(1)

		# for loop to deal with each images
		for batch_index in range(batch_size) :

			# Decode object coordinates from the form we regressed predicted boxes to
			decoded_locations = cxcy_to_xy(
				center_decode(predicted_boxes[batch_index], self.prior_boxes, self.variance["center"], self.variance["size"]))  # (8732, 4), these are fractional pt. coordinates
			
			# Lists to store boxes and scores for this image
			image_boxes = list()
			image_labels = list()
			image_scores = list()

			# Check for each class，不包括「背景」
			for each_class in range(1, self.num_classes) :

				# Keep only predicted boxes and scores where scores for this class are above the minimum score
				class_scores = predicted_scores[batch_index][:, each_class]  # (8732)
				
				score_above_min_score = class_scores > self.min_score  # torch.uint8 (byte) tensor, for indexing
				
				number_above_min_score = score_above_min_score.sum().item()
				
				if number_above_min_score == 0:
					continue

				# get the predicted result that the score is above min_score
				class_scores = class_scores[score_above_min_score] # 維度：(number_qualified), number_min_score <= 8732
				class_decoded_locations = decoded_locations[score_above_min_score] # 維度：(number_qualified, 4)
				
				# Sort predicted boxes and scores by scores
				class_scores, class_scores_index = class_scores.sort(dim=0, descending=True) # 維度：(number_qualified)
				class_decoded_locations = class_decoded_locations[class_scores_index] # 維度：(number_min_score, 4)

				# # Find the overlap between predicted boxes
				# Non-Maximum Suppression (NMS)
				nms_result = nms(class_decoded_locations,class_scores, 0)

				# to list
				nms_result = nms_result.tolist()

				image_boxes.append(class_decoded_locations[nms_result])
				image_labels.append(torch.LongTensor([each_class]*len(nms_result)).to(self.device))
				image_scores.append(class_scores[nms_result])

				# # Non-Maximum Suppression (NMS)

				# # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
				# # 1 implies suppress, 0 implies don't suppress
				# suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

				# # Consider each box in order of decreasing scores
				# for box in range(class_decoded_locs.size(0)):
				#	 # If this box is already marked for suppression
				#	 if suppress[box] == 1:
				#		 continue

				#	 # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
				#	 # Find such boxes and update suppress indices
				#	 suppress = torch.max(suppress, overlap[box] > max_overlap)
				#	 # The max operation retains previously suppressed boxes, like an 'OR' operation

				#	 # Don't suppress this box, even though it has an overlap of 1 with itself
				#	 suppress[box] = 0

				# # Store only unsuppressed boxes for this class
				# image_boxes.append(class_decoded_locs[1 - suppress])
				# image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
				# image_scores.append(class_scores[1 - suppress])

			# If no object in any class is found, store a placeholder for 'background'
			if len(image_boxes) == 0:
				image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
				image_labels.append(torch.LongTensor([0]).to(self.device))
				image_scores.append(torch.FloatTensor([0.]).to(self.device))

			# Concatenate into single tensors
			image_boxes = torch.cat(image_boxes, dim=0) # 維度： (number_objects, 4)
			image_labels = torch.cat(image_labels, dim=0) # 維度：(number_objects)
			image_scores = torch.cat(image_scores, dim=0) # 維度：(number_objects)
			number_objects = image_scores.size(0)

			# Keep only the top k objects
			if number_objects > self.top_k:
				image_scores, image_scores_index = image_scores.sort(dim=0, descending=True)
				image_scores = image_scores[:self.top_k] # 維度：(top_k)
				image_boxes = image_boxes[image_scores_index][:self.top_k] # 維度：(top_k, 4)
				image_labels = image_labels[image_scores_index][:self.top_k] # 維度：(top_k)
			
			# print(image_boxes, image_labels, image_scores)
			# Append to lists that store predicted boxes and scores for all images
			all_images_boxes.append(image_boxes)
			all_images_labels.append(image_labels)
			all_images_scores.append(image_scores)

			# print("===================================")

		return all_images_boxes, all_images_labels, all_images_scores


