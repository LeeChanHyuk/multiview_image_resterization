import os
import cv2

base_path = 'C:/Users/user/Desktop/Project/Multi_view_renderer_from_photorealistic_image/data/pattern'
load_folder_name = '8k_pattern_for_digital_human_fliped(deprecated)'
save_folder_name = 'real'

image_name_list = os.listdir(os.path.join(base_path, load_folder_name))
for image_name in image_name_list:
	img_path = os.path.join(base_path, load_folder_name, image_name)
	img = cv2.imread(img_path)
	cv2.resize(img, (3840, 2160))
	flip_img = cv2.flip(img, 1)
	cv2.imwrite(os.path.join(base_path, save_folder_name, image_name), flip_img)