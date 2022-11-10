#######################################################################################
########################### From ChanHyukLee (2022.09.16) #############################
#######################################################################################

import os
import numpy as np
import cv2
from tqdm import tqdm

# This class is for making multiview image
# You must load the input image with the same shape of multiview pattern image
class MultiviewImageCreator():
	def __init__(self, width, height, view_point_number, slanted_angle) -> None:
		self.width = width
		self.height = height
		self.view_point_number = view_point_number
		self.slanted_angle = slanted_angle
		pass
	
	# create pattern index with multiview point number
	def multiview_pattern_indexing(self):
		pattern = np.zeros((self.height, self.width * 3, 1), dtype=np.uint8)
		print('Multiview pattern is indexing ...')
		for i in tqdm(range(self.height)):
			for j in range(self.width * 3):
				val = (1 + (3 * j) - i) # because we use the slanted angle 1/9 or -1/9
				# Also, in this code, the viewpoint range is (1, viewpointnumber+1)
				if val > self.view_point_number:
					val %= self.view_point_number
				if val <= 0:
					step = abs(int(val / self.view_point_number))
					val += ((step+1) * self.view_point_number)
				pattern[i][j][0] = int(val)
		return pattern

	# create pattern image from pattern indices
	def multiview_pattern_image_rendering(self, save_path, pattern, mode = 'positive'):
		self.file_check_and_remove(save_path)
		print('Multiview pattern is creating ...')
		for viewpoint in tqdm(range(1, self.view_point_number+1)):
			# The pattern image must be made for each viewpoint number
			pattern_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
			for i in range(self.height):
				for j in range(self.width * 3):
					val = int(pattern[i][j][0])
					if val == viewpoint:
						channel = int(j % 3)
						x = int(j / 3)
						pattern_image[i][x][channel] = 255
			if mode == 'positive':
				cv2.imwrite(
				os.path.join(save_path, str(viewpoint) + '.png'),
					pattern_image)
			else:
				cv2.imwrite(
				os.path.join(save_path, str(viewpoint) + '.png'),
					cv2.flip(pattern_image, 1))

	def resterization(self, input_images, pattern_images):
		merged_image = np.zeros((self.height, self.width, 3), np.uint8)
		print('Resterization is processing ...')
		for i in tqdm(range(self.view_point_number)):
			bitwised_image = cv2.bitwise_and(input_images[i], pattern_images[i])
			merged_image += bitwised_image
		return merged_image

	# Check the path of input and output folder
	def file_check_and_remove(self, path : str) -> None:
		if not os.path.exists(path):
			os.mkdir(path)
		image_list = os.listdir(path)
		if len(image_list) > 0:
			for file_name in image_list:
				os.remove(os.path.join(path, file_name))






