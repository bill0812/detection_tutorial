# import other packages
from itertools import product
from math import sqrt

# import pytorch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class for ssd head, 包括 Prior box 的生成、location_layer、classification_layer，
# 以及算出結果、loss
class ssd_head(nn.Module) :
	
	# basic setup 
	def __init__(self, anchor_head_config, prior_box_config) :
		super(ssd_head, self).__init__()

		# define some config, 下面會用到
		self.anchor_head_config = anchor_head_config
		self.prior_box_config = prior_box_config
		self.num_classes = anchor_head_config["class_numbers"]

		# define anchor(bbox) head's module list : classification / location
		self.location_layer, self.classification_layer = self.predictor_layers()

		# define anchor box, 也叫做 Prior Box。根據 Config 裡的定義，像是 Stride 來生成
		# define some variables that we'll use here
		self.input_size = self.prior_box_config['input_size']
		self.aspect_ratios = self.prior_box_config['aspect_ratios'][str(self.input_size)]
		self.num_priors = len(self.aspect_ratios)
		self.center_variance = self.prior_box_config['variance']["center"]
		self.size_variance = self.prior_box_config['variance']["size"]
		self.feature_maps = self.prior_box_config['feature_maps'][str(self.input_size)]
		self.obj_scales = self.prior_box_config['obj_scales'][str(self.input_size)]
		self.clip = self.prior_box_config['clip']
		self.feature_maps_name = list(self.feature_maps.keys())
		self.prior_box = self.make_prior_box()

		# reset parameter
		self.reset_parameters()

	# define forward
	def forward(self, feature_map) :
		
		# forward predict locations and confidence
		location_predict, classification_confidence = self.box_predictor(feature_map)

		return location_predict, classification_confidence

	# reset parameters
	def reset_parameters(self) :

		# reset location layer
		for each_layer in self.location_layer.children():
			if isinstance(each_layer, nn.Conv2d):
				nn.init.xavier_uniform_(each_layer.weight)
				nn.init.zeros_(each_layer.bias)

		# reset classification layer
		for each_layer in self.classification_layer.children():
			if isinstance(each_layer, nn.Conv2d):
				nn.init.xavier_uniform_(each_layer.weight)
				nn.init.zeros_(each_layer.bias)

	# predictor layer, 包括 location / confidence layer
	def predictor_layers(self) :

		# define location layer and confidence layer
		location_layer = []
		confidence_layer = []

		# get some variables from config
		coordinates_number = self.anchor_head_config["coordinate_numbers"]
		number_classes = self.anchor_head_config["class_numbers"]
		input_size = self.anchor_head_config["input_size"]
		backbone_outchannel = self.anchor_head_config["backbone_outchannel"][str(input_size)]
		number_anchors = self.anchor_head_config["number_anchors"][str(input_size)]

		# 依照 config 設定的 backbone 特定層所輸出的 channel 數，來定義 location layer and confidence layer
		for index, out_channel in enumerate(backbone_outchannel):
			location_layer += [nn.Conv2d(out_channel, number_anchors[index] * coordinates_number, kernel_size=3, padding=1)]
			confidence_layer += [nn.Conv2d(out_channel, number_anchors[index] * number_classes, kernel_size=3, padding=1)]
		
		return nn.ModuleList(location_layer), nn.ModuleList(confidence_layer)

	# define anchor box (Prior Box)
	def make_prior_box(self) :
		
		# 找出對應的 anchor box，算 loss 會用到
		prior_box = []

		# use for loop to find prior box
		# find aspect ration for 1 and others
		# 1 for square
		# 同時依據各個大小的 feature maps，找出其對應的 stride 的每個格子的大小
		# 然後找出每個格子的中心點跟其對應的 anchor
		prior_boxes = []

		for k, feature_map in enumerate(self.feature_maps_name):
			for i in range(self.feature_maps[feature_map]):
				for j in range(self.feature_maps[feature_map]):
					cx = (j + 0.5) / self.feature_maps[feature_map]
					cy = (i + 0.5) / self.feature_maps[feature_map]

					for ratio in self.aspect_ratios[feature_map]:
						prior_boxes.append([cx, cy, self.obj_scales[feature_map] * sqrt(ratio), self.obj_scales[feature_map] / sqrt(ratio)])

						# For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
						# scale of the current feature map and the scale of the next feature map
						if ratio == 1.:
							try:
								
								additional_scale = sqrt(self.obj_scales[feature_map] * self.obj_scales[self.feature_maps_name[k + 1]])
							# For the last feature map, there is no "next" feature map
							except IndexError:
								additional_scale = 1.
							prior_boxes.append([cx, cy, additional_scale, additional_scale])

		prior_boxes = torch.FloatTensor(prior_boxes)  # (8732, 4)
		prior_boxes.clamp_(0, 1)  # (8732, 4)

		return prior_boxes

	# box predictor from feature maps
	def box_predictor(self, feature_map) :

		# define two return elements
		location_predict = []
		classification_confidence = []
		batch_size = feature_map[0].shape[0]

		# 解說這部分：
		# permute重新排列维度順序, PyTorch维度的默認順序為 (N, C, H, W)[ <batch, channels, width, height> ]
		# 所以在這我們調整卷積後的順序成 (N, H, W, C)，因為我們希望 channels 數在後面
		# 但 permute 會改變原本內存內部的儲存方法，加上因為待會要用 view（類似 numpy reshape）
		# 我們要用 contiguous 使內存連續化，變成同一區塊
		# 變成：loc => [batch, num_boxes*4] / conf => [batch, num_boxes*num_classes]
		for each_map, each_loc_layer, each_conf_layer \
			in zip(feature_map, self.location_layer, self.classification_layer) :

			location_predict.append(each_loc_layer(each_map).permute(0, 2, 3, 1).contiguous())
			classification_confidence.append(each_conf_layer(each_map).permute(0, 2, 3, 1).contiguous())
		
		# 將除batch以外的其他维度合併。因此，對於
		# 邊框座標來說最终的 shape 為：[batch, num_boxes*4]；
		# 標籤信心 shape 為：[batch, num_boxes*num_classes]
		# P.S. : 1 means => column，第二個維度
		location_predict = torch.cat([each_batch_loc.view(each_batch_loc.size(0),-1) \
			for each_batch_loc in location_predict], 1)
		classification_confidence = torch.cat([each_batch_conf.view(each_batch_conf.size(0), -1) \
			for each_batch_conf in classification_confidence], 1)

		# view these to : location : (batch, n , 4) / confidence : (batch, n , num_classes)
		location_predict = location_predict.view(batch_size, -1, 4)
		classification_confidence = classification_confidence.view(batch_size, -1, self.num_classes)

		return location_predict, classification_confidence
