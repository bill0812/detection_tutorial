# import basic packages
import cv2, sys, torch, glob, os
from tqdm import tqdm
import numpy as np
import os.path as osp
from PIL import Image
from pathlib import Path ,PurePath

# import pytorch packages
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
	"""
	A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
	"""

	# basic setup
	def __init__(self, current_mode, data_root, label_file, transform, size):
		
		self.label_map, self.label_color = self.map_labels(label_file)
		self.range = list(self.label_map.keys())[:-1]
		self.category = [0 for i in self.range]
		self.data_root = data_root
		self.count = 1 
		self.transform = transform
		self.mode = current_mode
		self.result = {}
		self.images = []
		self.size = size

		# use os.path get directory name
		dataset_name = os.path.basename(os.path.normpath(self.data_root))
		data_kind, check_folder = os.path.splitext(dataset_name)

		# load data
		if data_kind == "PennFudanPed" and check_folder == "":
			self.load_PennFudanPed()
		elif data_kind == "Xview" and check_folder == ".csv":
			self.load_Xivew()
		else :
			raise TypeError("| DataSet Name Should Be 'PennFudanPed' or 'Xview' !!")

	def collate_fn(self, batch, mode):
		"""
		Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

		This describes how to combine these tensors of different sizes. We use lists.

		Note: this need not be defined in this Class, can be standalone.

		:param batch: an iterable of N sets from __getitem__()
		:return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
		"""

		images = list()
		images_original = list()
		boxes = list()
		labels = list()
		difficulties = list()

		if mode == "train" or mode == "validation" :

			for b in batch:
				images.append(b[0])
				boxes.append(b[1])
				labels.append(b[2])
				difficulties.append(b[3])
			
			images = torch.stack(images, dim=0)

			return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
		else :
			for b in batch:

				images.append(b[0])
				images_original.append(b[1])

			images = torch.stack(images, dim=0)
			images_original = torch.stack(images_original, dim=0)

			return images, images_original  # tensor (N, 3, 300, 300), 3 lists of N tensors each

	def map_labels(self, label_file):
	
		label_map = {}
		label_color = {}
		labels = open(label_file, 'r')
		for line in labels:
			ids = line.split(',')
			label_map[int(ids[0])] = ids[1].split('\n')[0]
			label_color[int(ids[0])] = ids[2].split('\n')[0]
		
		return label_map, label_color

	# load image name and ground truth
	def load_PennFudanPed(self) :

		# load all image files, sorting them to
		# ensure that they are aligned
		self.image_path = list(sorted(os.listdir(os.path.join(self.data_root, "PNGImages"))))
		self.mask_path = list(sorted(os.listdir(os.path.join(self.data_root, "PedMasks"))))

		assert len(self.image_path) == len(self.mask_path)

		for index in range(len(self.image_path)) :

			# get name
			img_name = os.path.join(self.data_root, "PNGImages", self.image_path[index])

			# get number object
			mask_name = os.path.join(self.data_root, "PedMasks", self.mask_path[index])
			mask = Image.open(mask_name)
			mask = np.array(mask)
			# instances are encoded as different colors
			obj_ids = np.unique(mask)
			# first id is the background, so remove it
			obj_ids = obj_ids[1:]
			# split the color-encoded mask into a set
			# of binary masks
			masks = mask == obj_ids[:, None, None]

			# get bounding box coordinates for each mask
			num_objs = len(obj_ids)
			boxes = []

			# get positione / label / difficulties
			for i in range(num_objs):

				if img_name not in self.result :
					self.result[img_name] = []

				pos = np.where(masks[i])
				xmin = np.min(pos[1])
				xmax = np.max(pos[1])
				ymin = np.min(pos[0])
				ymax = np.max(pos[0])
				self.result[img_name].append(np.array([xmin, ymin, xmax, ymax, 1, 0]))

		self.images = list(self.result.keys())

	def load_Xivew(self,) :

		# retrieve data from ground truth
		with open(filename,"r") as file_detail :
			for line, row in enumerate(file_detail) :
			
				# retrieve xmin, ymin, xmax, ymax, class_name
				img_name, xmin, ymin, xmax, ymax, class_name = row.split(",")
				xmin = int(float(xmin))
				ymin = int(float(ymin))
				xmax = int(float(xmax))
				ymax = int(float(ymax))
				class_name = int(float(class_name.split("\n")[0]))

				if img_name not in self.result :
					self.result[img_name] = []

				if (xmin, ymin, xmax, ymax, class_name) == ('', '', '', '', '') :
					continue

				self.result[img_name].append(np.array([xmin, ymin, xmax, ymax, class_name, 0]))

		self.images = list(self.result.keys())

	def load_test_image(self) :

		all_files = glob.glob(self.current_dir)
		all_files.sort()

		# retrieve data from testing data
		for each_test_image in tqdm(all_files) :
			
			self.images.append(each_test_image)
		
	# load image
	def load_image(self, index) :
		
		# img = cv2.imread(self.images[index])		
		# height, width, channels = img.shape
		image = Image.open(self.images[index]).convert('RGB')

		return image

	# load ground truth
	def load_ground_truth(self, index) :

		# get ground truth
		groundtruth = np.asarray(self.result[self.images[index]])
		boxes = groundtruth[:,0:4]
		classes = groundtruth[:,4:5]
		difficulties = groundtruth[:,5:6]

		# Do some transformation of boxes
		boxes = np.array(boxes).astype(np.float64)

		return boxes, classes, difficulties

	def __getitem__(self, index):

		# declare some variables
		img_class = []
	
		img = self.load_image(index)

		if self.mode == "train" or self.mode == "validation":

			boxes, classes, difficulties = self.load_ground_truth(index)

			# how many bounding boxes, then you'll have that amount of classes
			assert boxes.shape[0] == classes.shape[0]

			# transfer to numpy first
			bounding_box = torch.from_numpy(boxes)
			img_class = torch.from_numpy(classes)
			difficulties = torch.from_numpy(difficulties)

			# set type
			bounding_box = bounding_box.type(torch.float32)
			img_class = img_class.type(torch.float32).squeeze(1)
			difficulties = difficulties.type(torch.uint8).squeeze(1)
			
			img, boxes, labels, difficulties = self.transform(img, bounding_box, img_class, difficulties)

			return img, boxes, labels, difficulties
		else :
			
			img_original = img.resize((self.size, self.size),Image.ANTIALIAS)
			img_original = np.asarray(img_original)
			img_original = torch.from_numpy(img_original)
			bounding_box = torch.from_numpy(np.array([[0,0,0,0]]))
			img_class = torch.from_numpy(np.array([0]))
			difficulties = torch.from_numpy(np.array([0]))

			bounding_box = bounding_box.type(torch.float32)
			img_class = img_class.type(torch.float32)
			difficulties = difficulties.type(torch.uint8)
			new_img, boxes, labels, difficulties = self.transform(img, bounding_box, img_class, difficulties)

			return new_img, img_original
	def __len__(self):
		return len(self.images)