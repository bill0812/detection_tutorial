import os

# input size define which kind of ssd / all training classes number / training,validation,testing data / 
# load state from url
# current_folder = os.getcwd() 
current_folder = "/home/bill/Desktop/graduation_VIPL/bayesian_bigscale/research_workshop/detection_tutorial"
global_config = {
	"input_size" : 512,
	"num_classes" : 2,
	"dataset_name" : "PennFudanPed",
	"data_root" : current_folder ,
	"folder_root" : current_folder,
	"gpu" : 1,
	"batch_size" : 4,
	"number_workers" : 4
}
# ===================================
ssd_model = {
	"model_name" : "Single_Stage_Dector",
	"input_size" : global_config["input_size"],
	"backbone" : {
					"net_name" : "vgg-16-based",
					"input_size" : global_config["input_size"],
					"basic_setting" : {
						"input_channel" : 3,
						'300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
								512, 512, 512],
						'512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512],
						"batch_norm" : False
					},
					"extra_setting" : {
						"input_channel" : 1024,
						'300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
						'512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
					},
					"depth" : 16
	},
	# In anchor_head part, "300" 、 "512" 在 backbone_outchannel 和 number_anchors 的數量要一樣，分別為 6、7
	"anchor_head" : {
		"head_name" : "ssd_head",
		"class_numbers" : global_config["num_classes"],
		"coordinate_numbers" : 4,
		"input_size" : global_config["input_size"],
		"backbone_outchannel" : {
			'300' : [512, 1024, 512, 256, 256, 256],
			'512' : [512, 1024, 512, 256, 256, 256, 256]
		},
		# number of boxes per feature map location
		# [len(ratios) * 2 + 2 for ratios in anchor_ratios] p.s => "aspect_ratios" is in prior_box
		"number_anchors" : {
			'300': [4, 6, 6, 6, 4, 4],		  
			'512': [4, 6, 6, 6, 6, 4, 4]
		}
	},
	"prior_box" : {
					"input_size" : global_config["input_size"],
					"num_classes" : global_config["num_classes"],
					"feature_maps" : {
						"300" : {
							'first': 38,
							'second': 19,
							'third': 10,
							'fourth': 5,
							'fifth': 3,
							'sixth': 1
						},
						"512" : {
							'first': 64,
							'second': 32,
							'third': 16,
							'fourth': 8,
							'fifth': 4,
							'sixth': 2,
							'seventh' : 1
						}
					},
					"obj_scales" : {
						"300" : {
							'first': 0.1,
							'second': 0.2,
							'third': 0.375,
							'fourth': 0.55,
							'fifth': 0.725,
							'sixth': 0.9
						},
						"512" : {
							'first': 0.1,
							'second': 0.2,
							'third': 0.35,
							'fourth': 0.525,
							'fifth': 0.675,
							'sixth': 0.8,
							'seventh': 0.9
						}
					},
					"aspect_ratios" : {
						"300" : {
							'first': [1., 2., 0.5],
							'second': [1., 2., 3., 0.5, .333],
							'third': [1., 2., 3., 0.5, .333],
							'fourth': [1., 2., 3., 0.5, .333],
							'fifth': [1., 2., 0.5],
							'sixth': [1., 2., 0.5]
						},
						"512" : {
							'first': [1., 2., 0.5],
							'second': [1., 2., 3., 0.5, .333],
							'third': [1., 2., 3., 0.5, .333],
							'fourth': [1., 2., 3., 0.5, .333],
							'fifth': [1., 2., 3., 0.5, .333],
							'sixth': [1., 2., 0.5],
							'seventh': [1., 2., 0.5]
						}
					},
					"variance" : {
						"size" : 0.2,
						"center" : 0.1
					},
					"clip" : True
	},
	"loss_function" : {
		"loss_name" : "multibox_loss",
		"regression_loss" : {
			"function_type" : "smooth_l1",
			"regression_alpha" : 1
		},
		"confidence_loss" : {
			"function_type" : "cross_entropy",
			"negative_positive_ratio" : 3,
		},
		"overlap_threshold" : 0.5,
	}
}
# =======================================================================================================================
training_config = {
	"batch_size" : global_config["batch_size"],
	"number_workers" : global_config["number_workers"],
	"epoch" : 200,
	"learning_rate" : 0.001,
	"weight_decay" : 5e-4,
	"gamma" : 0.1,
	"momentum" : 0.9,
	"gradient_clip" : True,
	"optimizer" : "SGD",
	'lr_steps': (5, 60, 90),
	# if status is True, config 要有設定
	"validation" : {
		"status" : True,
		"number_epoch" : 2, 
		"config" : {
			"max_iou_threshold" : 0.45,
			"min_score_threshold" : 0.01,
			"top_k_result" : 200
		},
		"mAP_overlap" : 0.5
	},
	"checkpoint" : {
		"directory" : "Result/",
		"model_name" : None
	},
	"mode" : "training",
	"device" : {
		# 目前只使用一個，之後會再使用 data_parallel
		"GPU" : global_config["gpu"]
	},
	"status" : True
}
# =======================================================================================================================
testing_config = {
	"main_work" : "NMS(Non-Maximum Supression)",
	"batch_size" : global_config["batch_size"],
	"number_workers" : global_config["number_workers"],
	"config" : {
		"max_iou_threshold" : 0.5,
		"min_score_threshold" : 0.2,
		"top_k_result" : 200
	},
	"visualization_size" : 512,
	"mode" : "testing",
	"status" : True,
	"checkpoint" : {
		"directory" : "Result/",
		"model_name" : "SSD.pt"
	},
	"outputs" : "outputs_result/",
	"device" : {
		# 目前只使用一個，之後會再使用 data_parallel
		"GPU" : global_config["gpu"]
	},
}
# =======================================================================================================================
dataset = {
	"dataset_name" : global_config["dataset_name"],
	"label_file" : global_config["folder_root"] + "/detection_tutorial/Datasets/label.txt",
	"training_data" : global_config["data_root"] + "/PennFudanPed/",
	"validation_data" : global_config["data_root"] + "/PennFudanPed/",
	"testing_data" : global_config["data_root"] + "/PennFudanPed/", 
	"dataloader" : {
		"number_classes" : global_config["num_classes"],
		"number_workers" : 4,
		"transformation" : {
			"status" : True,
			"mean" : (123, 117, 104),
			"std" : 1,
			# Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
			# see: https://pytorch.org/docs/stable/torchvision/models.html
			"normalize" : {
				"mean" : (0.485, 0.456, 0.406),
				"std" : (0.229, 0.224, 0.225)
			},
			"image_resize" : global_config["input_size"],
			"extra_augment" : {
				"PhotometricDistort" : {
					"contrast_range" : [0.5, 1.5],
					"saturation_range" : [0.5, 1.5],
					"hue_delta" : 18,
					"brightness_range" : [0.5, 1.5],
				},
				"RandomSampleCrop" : {
					"min_iou" : [1,0.3,0.7,0.9,None],
					"min_crop_size" : 1,
					"aspect_ratio_constraint" : [2,0.5]
				}
			}
		}
	}
}
