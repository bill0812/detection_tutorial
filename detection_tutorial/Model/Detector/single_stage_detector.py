# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# import some element from "Model"
from detection_tutorial.Model.Backbone.ssd_vgg import ssd_vgg as build_backbone
from detection_tutorial.Model.Anchor_Head.ssd_head import ssd_head as build_head

# import some element from "detection_tutorial."
from detection_tutorial.Utils import box_utils

# combine backbone and anchor_head, using config's variables
class single_stage_detector(nn.Module):
	
	# define backbone and anchor_head
	def __init__(self, model_config, other_config, mode, device) :
	
		# 繼承父類別
		super(single_stage_detector, self).__init__()
		
		# define variables, including backbone and anchor_head
		# config including model itself and other config, like testing or training config
		self.model_config = model_config
		self.other_config = other_config
		self.mode = other_config["mode"]
		self.device = device
		self.validation = False
		
		# model define
		self.backbone = build_backbone(model_config["backbone"])
		self.anchor_head = build_head(model_config["anchor_head"], model_config["prior_box"])

	# model forward
	def forward(self, images) :
		
		# vgg based model, return specific feature maps
		features_map = self.backbone(images)
		
		# return detections result and losses
		predicted_boxes, predicted_scores = self.anchor_head(features_map)

		return predicted_boxes, predicted_scores