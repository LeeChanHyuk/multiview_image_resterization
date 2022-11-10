####################################################################
###################### 2022.09.06 ChanhyukLee ######################
##### Input: Nerf images (4000, 6000) ##############################
##### Output: Cropped nerf images, cropped multiview images ########
####################################################################

import os
import cv2
import numpy as np
from shutil import copy2
from tqdm import tqdm
from multiviewImageCreator import MultiviewImageCreator
from inputImageProcessor import InputImageProcessor

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

multi_view_image_creator = MultiviewImageCreator(width=3840, height=2160, view_point_number=72, slanted_angle=9)
input_image_processor = InputImageProcessor(
	input_path = os.path.join(BASE_PATH, 'data', 'input', 'nerf_black_an_doc'),
	resized_output_path=os.path.join(BASE_PATH, 'data', 'input', 'nerf_black_resized'),
	multiview_input_path=os.path.join(BASE_PATH, 'data', 'input', 'nerf_multiview_input'),
	input_width=3840, # for use kimsimin doc's pic this value must be 4000
	input_height=2160, # for use kimsimin doc's pic this value must be 6000
	output_width=3840,
	output_height=2160
	)

input_image_processor.image_crop_and_save(target_width=720, target_height=1080, min_h=0, max_h=6000, min_w=0, max_w=4000, sharpness=True, crop=False, use_new_file_name=True)
dir_path = os.path.join(BASE_PATH, 'data', 'output', 'video')
if not os.path.exists(dir_path):
	os.mkdir(dir_path)

pattern = multi_view_image_creator.multiview_pattern_indexing()

# if you have the pattern of the multiview image, you can pass this function
multi_view_image_creator.multiview_pattern_image_rendering(
		save_path=os.path.join(BASE_PATH, 'data', 'pattern', 'multiview_pattern'),
		pattern=pattern,
		mode='positive' # my crossover 81view 4k monitor is negative (-1/9) type
		)
for index, view_standard in enumerate(range(146, 2292, 2)):
	input_image_processor.copy_multiview_input(
		view_point_number=72,
		view_number_interval=2,
		reverse=False,
		center_view=view_standard
		) 

	imgs = input_image_processor.load_input_file(os.path.join(BASE_PATH, 'data', 'input', 'nerf_multiview_input'))
	patterns = input_image_processor.load_input_file(os.path.join(BASE_PATH, 'data', 'pattern', 'multiview_pattern'), viewpoint_center_offset = 40)# for 8k digital human the center view is 22
	merged_image = multi_view_image_creator.resterization(
		input_images=imgs,
		pattern_images=patterns
	)
	file_name = BASE_PATH + '/data/output'+ '/video' +'/' + str(index).zfill(3) + '.png'
	cv2.imwrite(file_name, merged_image)
	