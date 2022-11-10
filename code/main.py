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

multi_view_image_creator = MultiviewImageCreator(width=7680, height=4320, view_point_number=81, slanted_angle=9)
input_image_processor = InputImageProcessor(
	input_path = os.path.join(BASE_PATH, 'data', 'input', 'nerf_black_input'),
	resized_output_path=os.path.join(BASE_PATH, 'data', 'input', 'nerf_black_resized'),
	multiview_input_path=os.path.join(BASE_PATH, 'data', 'input', 'nerf_multiview_input'),
	input_width=4000,
	input_height=6000,
	output_width=7680,
	output_height=4320
	)

input_image_processor.image_crop_and_save(target_width=7680, target_height=4320, min_h=0, max_h=6000, min_w=0, max_w=4000)
input_image_processor.copy_multiview_input(
	view_point_number=81,
	view_number_interval=2,
	reverse=True
	)

pattern = multi_view_image_creator.multiview_pattern_indexing()

# if you have the pattern of the multiview image, you can pass this function
multi_view_image_creator.multiview_pattern_image_rendering(
	save_path=os.path.join(BASE_PATH, 'data', 'pattern', 'multiview_pattern'),
	pattern=pattern,
	mode='positive' # my crossover 81view 4k monitor is negative (-1/9) type # digital human 8k display is positive
	)

imgs = input_image_processor.load_input_file(os.path.join(BASE_PATH, 'data', 'input', 'nerf_multiview_input'))
patterns = input_image_processor.load_input_file(os.path.join(BASE_PATH, 'data', 'pattern', 'multiview_pattern'), viewpoint_center_offset = 40)
merged_image = multi_view_image_creator.resterization(
	input_images=imgs,
	pattern_images=patterns
)

cv2.imwrite(os.path.join(BASE_PATH, 'data', 'output', 'merged_output.png'), merged_image)
