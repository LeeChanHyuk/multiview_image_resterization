from fileinput import filename
import os
import cv2
import numpy as np
from shutil import copy2
from tqdm import tqdm
import re

class InputImageProcessor():
	def __init__(self, input_path, resized_output_path, multiview_input_path, input_width, input_height, output_width, output_height) -> None:
		self.input_path = input_path
		self.resized_output_path = resized_output_path
		self.multiview_input_path = multiview_input_path
		#self.file_check_and_remove(self.resized_output_path)
		self.file_check_and_remove(self.multiview_input_path)
		self.input_width = input_width
		self.input_height = input_height
		self.output_width = output_width
		self.output_height = output_height
		pass
	
	# Check the path of input and output folder
	def file_check_and_remove(self, path : str) -> None:
		if not os.path.exists(path):
			os.mkdir(path)
		image_list = os.listdir(path)
		if len(image_list) > 0:
			for file_name in image_list:
				os.remove(os.path.join(path, file_name))

	# check the input image size
	def image_check(self, img : np.array) -> np.array:
		img = cv2.resize(img, (self.input_width, self.input_height))
		return img

	# If you don't want to crop the image, please use this function
	def image_resize_and_save(self) -> None:
		image_list = os.listdir(self.input_path)
		print('Cropping is processing now')
		for file_name in tqdm(image_list):
			img = cv2.imread(os.path.join(self.input_path, file_name))
			img = self.image_check(img)
			crop_img = cv2.resize(crop_img, (self.output_width, self.output_height))
			cv2.imwrite(
				os.path.join(self.resized_output_path, file_name),
				crop_img
			)

	def image_crop(self, img, min_h = None, max_h = None, min_w = None, max_w = None):
		height, width = img.shape[:2]
		if min_h is None:
			min_h = 0
			for i in range(height):
				if img[i, :, :].any() > 10:
					break
				min_h = i
		if max_h is None:
			max_h = 0
			for i in range(height-1, 0, -1):
				if img[i, :, :].any() > 10:
					break
				max_h = i
		if min_w is None:
			min_w = 0
			for j in range(width):
				if img[:, j, :].any() > 10:
					break
				min_w = i
		if max_w is None:
			max_w = 0
			for j in range(width-1, 0, -1):
				if img[:, j, :].any() > 10:
					break
				max_w = i
		return img[min_h:max_h, min_w:max_w, :]

	# If you want to crop the image, please use this function
	def image_crop_and_save(self, target_width, target_height, min_h = 0, max_h = 6000, min_w = 0, max_w = 4000, sharpness=False, crop =False, use_new_file_name = True) -> None:
		image_list = os.listdir(self.input_path)
		print("Multiview image is cropping and resizing")
		for index, file_name in enumerate(tqdm(image_list)):
			# image load
			img = cv2.imread(os.path.join(self.input_path, file_name))
			img = self.image_check(img)

			if crop:
				# create black result image
				result = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8) 

				# image crop
				crop_img = self.image_crop(img, min_h=min_h, max_h=max_h, min_w=min_w, max_w=max_w)
				crop_img = img[0:2160,80:3920,:]
				crop_img = cv2.resize(crop_img, (target_width, target_height))

				# calculate the position of cropped image
				cropped_height, cropped_width = crop_img.shape[:2]
				min_x, min_y, max_x, max_y = 0, 0, 0, 0
				min_x = int((self.output_width - cropped_width) / 2)
				min_y = int((self.output_height - cropped_height) / 2)
				max_x = int(min_x + cropped_width)
				max_y = int(min_y + cropped_height)

				# put the cropped image into the result image
				result[min_y:max_y, min_x:max_x, :] = crop_img
			else:
				img = img[400:2100, 400:3400, :]
				result = cv2.resize(img, (7680, 4320))
			if sharpness:
				sharpening_kernel = np.array(
				[[-1, -1, -1],
				[-1, 9, -1],
				[-1, -1, -1]])
				result = cv2.filter2D(result, -1, sharpening_kernel)
			if use_new_file_name:
				new_file_name = str(index).zfill(4) + '.png'
				cv2.imwrite(
					os.path.join(self.resized_output_path, new_file_name),
					result
				)
			else:
				cv2.imwrite(
					os.path.join(self.resized_output_path, file_name),
					result
				)

	def copy_multiview_input(self, view_point_number : int, view_number_interval : int, reverse = True, center_view = None) -> None:
		self.file_check_and_remove(self.multiview_input_path)
		image_list = os.listdir(self.resized_output_path)
		image_num = len(image_list)
		if center_view is None:
			center_view_point = int(image_num / 2)
		else:
			center_view_point = center_view
		left_side_view_point_number = []
		right_side_view_point_number = []
		offset = view_number_interval

		while len(left_side_view_point_number) != int(view_point_number/2):
			left_side_view_point_number.append(center_view_point - offset)
			offset += view_number_interval

		offset = view_number_interval
		while len(right_side_view_point_number) != int(view_point_number/2):
			right_side_view_point_number.append(center_view_point + offset)
			offset += view_number_interval

		left_side_view_point_number.append(center_view_point)
		all_view_point_number = left_side_view_point_number + right_side_view_point_number
		all_view_point_number.sort()

		print('Multiview images are generating')
		for index, number in enumerate(tqdm(all_view_point_number)):
			file_name = str(number).zfill(4) + '.png'
			if reverse:
				save_file_name = str(len(all_view_point_number) - index -1).zfill(4) + '.png'
			else:
				save_file_name = str(index).zfill(4) + '.png'
			copy2(
				os.path.join(self.resized_output_path, file_name),
				os.path.join(self.multiview_input_path, save_file_name)
			)

	def load_input_file(self, path, viewpoint_center_offset = 0):
		img_list = []
		img_name_list = os.listdir(path)
		for index, file_name in enumerate(img_name_list):
			numbers = re.sub(r'[^0-9]', '', file_name)
			numbers = numbers.zfill(4)
			os.rename(
				os.path.join(path, file_name),
				os.path.join(path, numbers+'.jpg')
			)
			img_name_list[index] = numbers+'.jpg'
		img_name_list.sort()
		for img_name in img_name_list:
			img = cv2.imread(os.path.join(path, img_name))
			img_list.append(img)
		new_img_list = []
		for i in range(viewpoint_center_offset, len(img_list)):
			new_img_list.append(img_list[i])
		for i in range(0, viewpoint_center_offset):
			new_img_list.append(img_list[i])
		return new_img_list




