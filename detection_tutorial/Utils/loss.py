# This file includes all loss that we'll use in different network
# Like confidence loss, location loss...Like
# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# import box_utils
from detection_tutorial.Utils.box_utils import *

class MultiBoxLoss(nn.Module):

	# basic setup 
	def __init__(self, loss_config, center_variance, size_variance, prior_boxes, number_classes, device):
		super(MultiBoxLoss, self).__init__()

		# define some variables
		self.loss_alpha = loss_config["regression_loss"]["regression_alpha"]
		self.regression_function = loss_config["regression_loss"]["function_type"]
		self.loss_negative_positive_ratio = loss_config["confidence_loss"]["negative_positive_ratio"]
		self.confidence_function = loss_config["confidence_loss"]["function_type"]
		self.threshold = loss_config["overlap_threshold"]
		self.variance = {
			"center" : center_variance,
			"size" : size_variance
		}
		self.prior_boxes = prior_boxes
		self.device = device

		# define batch size, number of prior boxes, number_classes, and combine variance
		self.number_classes = number_classes
		self.batch_size = 0
		self.number_priors = self.prior_boxes.size(0)
		
		# define loss for location loss and confidence loss by the config's function that we want to use
		if self.regression_function == "smooth_l1" :
			self.regression_function = nn.L1Loss()
		else :
			raise TypeError("The Regression Loss Function Doesn't Defined !")

		if self.confidence_function == "cross_entropy" :
			self.confidence_function = nn.CrossEntropyLoss(reduction='none')
		else :
			raise TypeError("The Confidence Loss Function Doesn't Defined !")

	# regression loss
	def regression_loss(self, predicted_boxes, true_locs, positive_priors) :

		# Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
		# So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)
		
		loss = self.regression_function(predicted_boxes[positive_priors], true_locs[positive_priors])
		return loss

	# confidence loss
	def confidence_loss(self, predicted_scores, true_classes, positive_priors) :
		
		"""
		解說：
		1. Hard Negative Mining : Concentrating on hardest negatives in each image, and also minimizes pos/neg imbalance
			- Details :
				✨ loss is computed over positive priors and the most difficult (hardest) negative priors in each image
					=> Taking the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
		"""

		# Number of positive and hard-negative priors per image
		number_positives = positive_priors.sum(dim=1) # 維度：(N)
		number_hard_negatives = self.loss_negative_positive_ratio * number_positives # 維度：(N)

		# First, find the loss for all priors
		confidence_loss_all = self.confidence_function(predicted_scores.view(-1, self.number_classes), true_classes.view(-1)) # 維度：(N * 8732)
		confidence_loss_all = confidence_loss_all.view(self.batch_size, self.number_priors) # 維度：(N, 8732)

		# We already know which priors are positive
		confidence_loss_positives = confidence_loss_all[positive_priors] # 維度：(sum(number_positives))

		# Next, find which priors are hard-negative
		# To do this, sort ONLY negative priors in each image in order of decreasing loss and take top number_hard_negatives
		confidence_loss_negatives = confidence_loss_all.clone() # 維度：(N, 8732)

		# positive priors are ignored (never in top n_hard_negatives)
		confidence_loss_negatives[positive_priors] = 0. # 維度：(N, 8732) 

		# sorted by decreasing hardness
		confidence_loss_negatives, confidence_loss_negatives_index = confidence_loss_negatives.sort(dim=1, descending=True) # 維度：(N, 8732)
		# create an empty hard-negative-mining tensor
		hardness_ranks = torch.LongTensor(range(self.number_priors)).unsqueeze(0).expand_as(confidence_loss_negatives).to(self.device) # 維度：(N, 8732)
		
		# get numbers of number_hard_negatives。因為上面用 range，等於是一個 index 的概念，並且用 number_hard_negatives 
		# 來篩選出比這個數小的 index，待會會用來找出前 number_hard_negatives 名的 confidence loss of negatives
		hard_negatives = hardness_ranks < number_hard_negatives.unsqueeze(1) # 維度：(N, 8732) 
		confidence_loss_hard_negatives = confidence_loss_negatives[hard_negatives] # 維度：(sum(n_hard_negatives))

		# As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
		confidence_all_loss = (confidence_loss_hard_negatives.sum() + confidence_loss_positives.sum()) /\
				number_positives.sum().float() # 維度：(), scalar

		return confidence_all_loss
		
	# compute all lose
	def forward(self, predicted_boxes, predicted_scores, ground_truth_boxes, ground_truth_labels) : 

		# define batch size
		self.batch_size = predicted_boxes.size(0)

		# make sure there are (8732) boxes ( 看我們怎麼定義 prior boxes，要和其數量一樣)
		assert self.number_priors == predicted_boxes.size(1) == predicted_scores.size(1)
		
		# create variables for true locations / classes，這個是待會會經過 match 後，然後填進去
		true_locs = torch.zeros((self.batch_size, self.number_priors, 4), dtype=torch.float).to(self.device)  # (N, 8732, 4)
		true_classes = torch.zeros((self.batch_size, self.number_priors), dtype=torch.long).to(self.device)  # (N, 8732)

		# for each image (numbers of batchsize)
		for batch_index in range(self.batch_size) :

			# defince current image's boxes, score, ground_truth
			predicted_boxes_current = predicted_boxes[batch_index]
			predicted_scores_current = predicted_scores[batch_index]
			ground_truth_boxes_current = ground_truth_boxes[batch_index]
			ground_truth_labels_current = ground_truth_labels[batch_index]

			# return matched result
			true_locs[batch_index], true_classes[batch_index] = match(self.threshold, predicted_boxes_current, \
				predicted_scores_current, ground_truth_boxes_current, ground_truth_labels_current, self.prior_boxes, self.variance, self.device)
			
		# Identify priors that are positive (object/non-background)
		positive_priors = true_classes != 0  # (N, 8732)

		# Regression Loss
		location_loss = self.regression_loss(predicted_boxes, true_locs, positive_priors)

		# Confidence Loss
		confidence_loss = self.confidence_loss(predicted_scores, true_classes, positive_priors)

		# Total Loss
		return confidence_loss , self.loss_alpha * location_loss
		