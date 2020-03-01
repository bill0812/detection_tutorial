import torch
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Function
from torch.autograd import Variable

# 可以參考：https://www.ycc.idv.tw/ml-course-foundations_4.html
# define L2 Norm for conv4_3 layer
# Reference : https://github.com/weiliu89/caffe/issues/241#issuecomment-256096891
class L2Norm(nn.Module):
	def __init__(self,n_channels, scale):
		super(L2Norm,self).__init__()
		self.n_channels = n_channels
		self.gamma = scale or None
		self.eps = 1e-10
		self.weight = nn.Parameter(torch.Tensor(self.n_channels))
		self.reset_parameters()

	def reset_parameters(self):
		
		# conv4 3 has a different feature scale compared to the other layers
		# we use the L2 normalization technique introduced to scale the feature norm at each location in the feature map
		# learn the scale during back propagation
		init.constant_(self.weight,self.gamma)

	def forward(self, x):
		
		# 用 L2 正規化，限制權重W的大小以控制高次的影響
		# 用平方的方式，最後用來 divide
		norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
		x = torch.div(x,norm)
		
		# original dimension of weight is like above, 要去 unsqueeze 並且 expand 成 input 的樣子
		# and scale it
		out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
		return out