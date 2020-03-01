# import basic packages
import cv2, torch, types
import numpy as np
from numpy import random

# import pytorch packages
from torchvision import transforms
import torchvision.transforms.functional as FT

# for normalize / resize / to_tensor
class ToTensor(object) :

	def __call__(self, image, boxes=None, labels=None, difficulties=None):

		# Convert nd_array to Torch tensor
		new_image = FT.to_tensor(image)

		return new_image, boxes, labels, difficulties

# normalize
class Normalize(object) :

	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, image, boxes=None, labels=None, difficulties=None):

		new_image = FT.normalize(image, mean = self.mean, std = self.std)

		return new_image, boxes, labels, difficulties

# resize
class Resize(object):
	def __init__(self, size=300):
		self.size = size

	def __call__(self, image, boxes=None, labels=None, difficulties=None):

		# transfer from PIL to nd_array
		image = np.array(image)
		new_image = cv2.resize(image, (self.size,
								 self.size))

		height, width, _ = image.shape
		boxes[:,0] = boxes[:,0]/width
		boxes[:,1] = boxes[:,1]/height
		boxes[:,2] = boxes[:,2]/width
		boxes[:,3] = boxes[:,3]/height

		boxes = boxes.type(torch.float32)
		
		return new_image, boxes, labels, difficulties

# ===============================
# Photometri cDistort
class PhotometricDistort(object):
	def __init__(self, photometricdistort_arg) :
		self.pd = [
			RandomContrast(photometricdistort_arg["contrast_range"][0],photometricdistort_arg["contrast_range"][1]),
			RandomSaturation(photometricdistort_arg["saturation_range"][0],photometricdistort_arg["saturation_range"][1]),
			RandomBrightness(photometricdistort_arg["brightness_range"][0],photometricdistort_arg["brightness_range"][1]),
			RandomHue(photometricdistort_arg["hue_delta"]),
		]
		
	def __call__(self, image, box=None, labels=None, difficulties=None):

		new_image = image

		random.shuffle(self.pd)

		for d in self.pd :
			new_image = d(new_image)

		return new_image, box, labels, difficulties

class RandomSaturation(object):
	def __init__(self, lower, upper):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be >= lower."
		assert self.lower >= 0, "contrast lower must be non-negative."

	def __call__(self, image):
		if random.random() < 0.5:
			adjust_factor = random.uniform(self.lower, self.upper)

			# Transformation
			image = FT.adjust_saturation(image, adjust_factor)
		return image

class RandomHue(object):
	def __init__(self, delta):
		assert delta >= 0.0 and delta <= 360.0
		self.delta = delta

	def __call__(self, image):
		if random.random() < 0.5:
			adjust_factor = random.uniform( -self.delta/255. , self.delta/255. )
		
			# Transformation
			image = FT.adjust_hue(image, adjust_factor)
		return image

class RandomContrast(object):
	def __init__(self, lower, upper):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be >= lower."
		assert self.lower >= 0, "contrast lower must be non-negative."

	# expects float image
	def __call__(self, image):
		if random.random() < 0.5:
			adjuct_factor = random.uniform(self.lower, self.upper)
		
			image = FT.adjust_contrast(image, adjuct_factor)
		return image

class RandomBrightness(object):
	def __init__(self, lower, upper):
		self.lower = lower
		self.upper = upper
		assert self.upper >= self.lower, "contrast upper must be >= lower."
		assert self.lower >= 0, "contrast lower must be non-negative."

	def __call__(self, image):
		if random.random() < 0.5:
			adjust_factor = random.uniform(self.lower, self.upper)
		
			image = FT.adjust_brightness(image, adjust_factor)
		return image

#=================================

# Expand
class Expand(object):
	def __init__(self, mean):
		self.mean = mean

	def __call__(self, image, boxes=None, labels=None, difficulties=None):

		# convert PIL to numpy array 
		image = np.array(image)
		boxes = np.array(boxes)

		if random.random() < 0.5:
			return image, boxes, labels, difficulties
		else :
			height, width, depth = image.shape
			ratio = random.uniform(1, 4)
			left = random.uniform(0, width*ratio - width)
			top = random.uniform(0, height*ratio - height)

			expand_image = np.zeros(
				(int(height*ratio), int(width*ratio), depth),
				dtype=image.dtype)
			expand_image[:, :, :] = self.mean
			expand_image[int(top):int(top + height),
						int(left):int(left + width)] = image
			image = expand_image

			boxes = boxes.copy()
			boxes[:, :2] += (int(left), int(top))
			boxes[:, 2:] += (int(left), int(top))

			return image, boxes, labels, difficulties

# Flip / (also called random mirror)
class RandomMirror(object):
	def __call__(self, image, boxes=None, classes=None, difficulties=None):

		# retrieve width
		image = np.array(image)
		_, width, _ = image.shape

		new_image = transforms.ToPILImage()(image)
		new_boxes = boxes
		if random.randint(2):

			# Flip images
			new_image = FT.hflip(new_image)

			# Flip boxes
			new_boxes = boxes
			new_boxes[:, 0] = width - boxes[:, 0] - 1
			new_boxes[:, 2] = width - boxes[:, 2] - 1
			new_boxes = new_boxes[:, [2, 1, 0, 3]]

		return new_image, new_boxes, classes, difficulties

# random crop
class RandomSampleCrop(object):
	"""Crop
	Arguments:
		img (Image): the image being input during training
		boxes (Tensor): the original bounding boxes in pt form
		labels (Tensor): the class labels for each bbox
		mode (float tuple): the min and max jaccard overlaps
	Return:
		(img, boxes, classes)
			img (Image): the cropped image
			boxes (Tensor): the adjusted bounding boxes in pt form
			labels (Tensor): the class labels for each bbox
	"""
	
	def __init__(self, randomsamplecrop_arg):
		self.sample_options = []

		self.min_iou = randomsamplecrop_arg["min_iou"]
		for each in (self.min_iou) :
			if each == 1 :
				self.sample_options.append(None)
			else :
				self.sample_options.append((each,None))
		self.sample_options = tuple(self.sample_options)

		self.min_crop_size = randomsamplecrop_arg["min_crop_size"]
		self.aspect_ratio_constraint = randomsamplecrop_arg["aspect_ratio_constraint"]
		if len(self.aspect_ratio_constraint) !=2 :
			raise TypeError("Aspect Ratio Constraint Should be only two number : max and min")

	def __call__(self, image, boxes=None, labels=None, difficulties=None):
		height, width, _ = image.shape
		while True:
			# randomly choose a mode
			mode = random.choice(self.sample_options)
			if mode is None:
				return image, boxes, labels, difficulties

			min_iou, max_iou = mode
			if min_iou is None:
				min_iou = float('-inf')
			if max_iou is None:
				max_iou = float('inf')

			# max trails (50)
			for _ in range(50):
				current_image = image

				w = random.uniform(self.min_crop_size * width, width)
				h = random.uniform(self.min_crop_size * height, height)

				# aspect ratio constraint b/t .5 & 2
				if h / w < self.aspect_ratio_constraint[0] or h / w > self.aspect_ratio_constraint[1]:
					continue

				left = random.uniform(width - w)
				top = random.uniform(height - h)

				# convert to integer rect x1,y1,x2,y2
				rect = np.array([int(left), int(top), int(left+w), int(top+h)])
				
				# cut the crop from the image
				current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
											  :]
				# calculate IoU (jaccard overlap) b/t the cropped and gt boxes
				overlap = jaccard_numpy(boxes, rect)

				# is min and max overlap constraint satisfied? if not try again
				if overlap.min() < min_iou and max_iou < overlap.max():
					continue

				# keep overlap with gt box IF center in sampled patch
				centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

				# mask in all gt boxes that above and to the left of centers
				m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

				# mask in all gt boxes that under and to the right of centers
				m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

				# mask in that both m1 and m2 are true
				mask = m1 * m2

				# have any valid boxes? try again if not
				if not mask.any():
					continue

				# take only matching gt boxes
				current_boxes = boxes[mask, :].copy()

				# take only matching gt labels
				current_labels = labels[mask]

				# take difficult
				current_difficulties = difficulties[mask]

				# should we use the box left and top corner or the crop's
				current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
												rect[:2])
				# adjust to crop (by substracting crop's left,top)
				current_boxes[:, :2] -= rect[:2]

				current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
												rect[2:])
				# adjust to crop (by substracting crop's left,top)
				current_boxes[:, 2:] -= rect[:2]

				return current_image, current_boxes, current_labels, current_difficulties

# =============================
# main of augmentation
class Augmentation(object):

	def __init__(self, augmentation_config, spilt) :

		self.mean = augmentation_config["mean"]
		self.std = augmentation_config["std"]
		self.normalize_mean = augmentation_config["normalize"]["mean"]
		self.normalize_std = augmentation_config["normalize"]["std"]
		self.image_resize = augmentation_config["image_resize"]
		self.photometricdistort_arg = augmentation_config["extra_augment"]["PhotometricDistort"]
		self.randomsamplecrop_arg = augmentation_config["extra_augment"]["RandomSampleCrop"]
		self.spilt = spilt
		
		assert self.spilt in {'TRAIN', 'TEST'}

		if self.spilt == "TRAIN" :
			self.augment = [
				# 如果增加 expand / random_sample_crop 的話要注意 data type (Image / Tensor / Numpy)
				# 那兩個沒事別亂做啊！
				PhotometricDistort(self.photometricdistort_arg),
				# Expand(self.mean),
				# RandomSampleCrop(self.randomsamplecrop_arg),
				RandomMirror(),
				Resize(self.image_resize),  # include transfer to relative cooridinate
				ToTensor(),
				Normalize(self.normalize_mean,self.normalize_std)
			]
		else :
			self.augment = [
				Resize(self.image_resize),
				ToTensor(),
				Normalize(self.normalize_mean,self.normalize_std)
			]

	def __call__(self, img, boxes=None, labels=None, difficulties=None):
		
		new_image = img
		new_boxes = boxes
		new_labels = labels
		new_difficulties = difficulties

		# if self.spilt == "TEST" :

		# 	new_boxes = new_boxes.numpy()

		for each_aug in self.augment :

			new_image, new_boxes, new_labels, new_difficulties = each_aug(new_image, new_boxes, new_labels, new_difficulties)

		return new_image, new_boxes, new_labels, new_difficulties