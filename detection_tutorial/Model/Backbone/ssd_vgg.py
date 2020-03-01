# import other packages
import os

# import pytorch packages
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# load model dict from URL
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url

# import some packages from "Model" or "Layer"
from detection_tutorial.Utils.l2norm import L2Norm as l2norm
# ================================================================

# define vgg based net for ssd
class ssd_vgg(nn.Module) :
	
	# basic setup
	def __init__(self, backbone_config) :
		super(ssd_vgg, self).__init__()
		
		# backbone config, including vgg_based and extras
		self.base_config = backbone_config["basic_setting"]
		self.extra_config = backbone_config["extra_setting"]
		self.mode_input_size = backbone_config["input_size"]
		self.batchnorm = backbone_config["basic_setting"]["batch_norm"]

		# vgg network using cfg["basic_setting"]
		self.based = nn.ModuleList(self.vgg_based(self.base_config, self.mode_input_size))
		
		# Layer learns to scale the l2 normalized features from conv4_3 during vgg_based, 
		# and scale is set 20 originally, 論文有提到
		self.l2norm = l2norm(512, scale=20)
		
		# extra layers for ssd network using cfg["extra_setting"]
		self.extras = nn.ModuleList(self.vgg_extra(self.extra_config, self.mode_input_size))

		# reset parameter depends on the variable "pretrain_vgg"
		self.reset_parameters()

	# reset model's parameters
	def reset_parameters(self) :

		# set vgg based model from torchvision and convert fc to conv
		# Current state of base
		state_dict = self.based.state_dict()
		param_names = list(state_dict.keys())

		# Pretrained VGG base
		pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
		pretrained_param_names = list(pretrained_state_dict.keys())

		# Transfer conv. parameters from pretrained model to current model
		for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
			state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

		# Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
		# fc6
		conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
		conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
		state_dict['31.weight'] = self.decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
		state_dict['31.bias'] = self.decimate(conv_fc6_bias, m=[4])  # (1024)
		# fc7
		conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
		conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
		state_dict['33.weight'] = self.decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
		state_dict['33.bias'] = self.decimate(conv_fc7_bias, m=[4])  # (1024)

		# Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
		# ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
		# ...operating on the 2D image of size (C, H, W) without padding

		self.based.load_state_dict(state_dict)

		# =============================================================================================================

		# init for extra layers
		for each_extra in self.extras.children():
			if isinstance(each_extra, nn.Conv2d):
				nn.init.xavier_uniform_(each_extra.weight)
				nn.init.constant_(each_extra.bias, 0.)

	# define forward
	def forward(self, x) :

		# define feature maps that we'll return
		feature_maps = []
		
		if self.batchnorm :
			vgg_con4v = 32
		else :
			vgg_con4v = 23

		# print(vgg_con4v)

		# 1. 跑 vgg-based 網路
		# ========================================
		# go throught self.vgg_based base net
		for layer_index in range(vgg_con4v):
			x = self.based[layer_index](x)
			
		# apply l2 norm to conv4_3
		conv4_3 = self.l2norm(x)
		feature_maps.append(conv4_3)

		# 把剩下 self.vgg_based 跑完
		for layer_index in range(vgg_con4v, len(self.based)) :
			x = self.based[layer_index](x)
		
		# add conv7 to feature map
		conv7 = x
		feature_maps.append(conv7)

		# ========================================

		# 2. 跑 vgg-extra 網路
		# ========================================
		for layer_index, extra_layer in enumerate(self.extras):
			x = F.relu(extra_layer(x))
			if layer_index % 2 == 1:
				feature_maps.append(x)

		# ========================================

		# return anchor head 所需要的 feature maps
		return feature_maps

	@staticmethod
	# define vgg_extra layers
	def vgg_extra(config, input_size) :
		
		# some variables, 包括 input channels, 各個層數channel
		in_channel = config["input_channel"]
		extra_setting = config[str(input_size)]
		flag = False

		# define extra layer for extra feature maps
		extra_layer = []

		# 注釋 kernel_size=(1,3)[flag] =>
		# if flag==True : kernel_size=3 
		# else: kernel_size=1

		# for loop to 卷積額外的 feature maps
		for index, out_channel in enumerate(extra_setting):
			if in_channel != 'S':
				if out_channel == 'S':
					extra_layer += [nn.Conv2d(in_channel, extra_setting[index + 1],
							kernel_size=(1, 3)[flag], stride=2, padding=1)]
				else:
					extra_layer += [nn.Conv2d(in_channel, out_channel, kernel_size=(1, 3)[flag])]
				flag = not flag

			in_channel = out_channel

		# set for input_size = 512
		if input_size == 512:
			extra_layer.append(nn.Conv2d(in_channel, 128, kernel_size=1, stride=1))
			extra_layer.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))

		return extra_layer

	@staticmethod
	# define vgg_based network
	def vgg_based(config, input_size) :
		
		# R G B three channels, and 其他會用到的變數
		in_channel = config["input_channel"]
		basic_setting = config[str(input_size)]
		batch_norm = config["batch_norm"]
		
		# define module list
		based_layer = []
		
		# 卷積直到原本 vgg FC 之前
		for out_channel in basic_setting :
			if out_channel == 'M':
				based_layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
			elif out_channel == 'C':
				based_layer += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
			else :
				conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
				if batch_norm:
					based_layer += [conv2d, nn.BatchNorm2d(out_channel), nn.ReLU()]
				else:
					based_layer += [conv2d, nn.ReLU()]
				in_channel = out_channel 
			
		pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

		# 取代 vgg 原本的 fully connected layers
		conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
		conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
		based_layer += [pool5, conv6, nn.ReLU(), conv7, nn.ReLU()]
		return based_layer

	@staticmethod
	def decimate(tensor, m):
		"""
		Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

		This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

		:param tensor: tensor to be decimated
		:param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
		:return: decimated tensor
		"""
		assert tensor.dim() == len(m)
		for d in range(tensor.dim()):
			if m[d] is not None:
				tensor = tensor.index_select(dim=d,
											index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

		return tensor