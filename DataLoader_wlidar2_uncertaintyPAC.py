import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import random
import tifffile
import cv2
from base_dataloader import get_transform
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transform

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 711.3722
baseline = 0.54

class KITTIDataloader(Dataset):

	__left = []
	__right = []
	__disp = []
	__rdisp = []
	__conf = []
	__lidar = []

	def __init__(self):
		self.filename = "/home/cvlab/Documents/pytorch-segnet-master/src/utils/KITTI_lidar3_last_file4"
		self.img_root = "/media/data1/datasets/EWHA/KITTI/Train/Left/"
		self.right_img_root = "/media/data1/datasets/EWHA/KITTI/Train/Right/"
		self.disp_root = "/media/data1/datasets/EWHA/KITTI/Train/CRL/"
		self.rdisp_root = "/media/data1/datasets/EWHA/KITTI/Train/CRL_right/"
		self.conf_root = "/media/data1/datasets/EWHA/KITTI/Train/CRL_confidence/"
		self.lidar_root = "/media/data1/Lidar_img/"
		self.toPIL = transform.ToPILImage()
		self.toTensor = transform.ToTensor()

		arr = open(self.filename, "rt").read().split("\n")[:-1]

		n_line = open(self.filename).read().count('\n')

		for line in range(n_line):
			self.__left.append(self.img_root + arr[line] + '.png')
			self.__right.append(self.right_img_root + arr[line] + '.png')
			self.__disp.append(self.disp_root + arr[line] + '.png')
			self.__rdisp.append(self.rdisp_root + arr[line] + '.png')
			self.__conf.append(self.conf_root + arr[line] + '.png')
			self.__lidar.append(self.lidar_root + arr[line] + '.tiff')

	def __getitem__(self, index):
		img1 = Image.open(self.__left[index]).convert('RGB')
		img11 = Image.open(self.__right[index]).convert('RGB')
		img2 = Image.open(self.__disp[index])
		img22 = Image.open(self.__rdisp[index])
		img3 = Image.open(self.__conf[index])
		img4 = tifffile.imread(self.__lidar[index])

		#img1, img2, img3, img4 = self.augument_image_pair(img1, img2, img3, img4)

		transform = get_transform()
		img1 = transform(img1)
		img11 = transform(img11)

		img2_ori = self.load_image2(img2)
		img22_ori = self.load_image2(img22)

		img2_disp_ori = self.load_image2_disp(img2)
		img22_disp_ori = self.load_image2_disp(img22)

		img2_mask = self.load_image2_mask(img2)
		img2_disp_mask = self.load_image2_disp_mask(img2)
		img3 = self.load_image3(img3)

		img2_ori = self.toTensor(img2_ori)
		img22_ori = self.toTensor(img22_ori)

		img2_disp_ori = self.toTensor(img2_disp_ori)
		img22_disp_ori = self.toTensor(img22_disp_ori)

		img3 = self.toTensor(img3)
		img2_mask = self.toTensor(img2_mask)
		img2_disp_mask = self.toTensor(img2_disp_mask)
		img4 = self.toPIL(np.expand_dims(img4, axis=-1))
		img4_depth = self.toTensor(img4)

		#img4_disparity = self.load_image4_disp(img4)
		#img4_disparity = self.toTensor(img4_disparity)
		#img4_disparity[torch.isinf(img4_disparity)] = 0

		#img4 = self.load_image4(img4)

		gt_mask = self.mask_compare(img2_mask, img4_depth)
		gt_disp_mask = self.mask_compare_disp(img2_disp_mask, img4_depth)
		gt_mask = self.downsample_nearest(gt_mask)
		gt_disp_mask = self.downsample_nearest(gt_disp_mask)

		#img2_ori = torch.FloatTensor(img2_ori)
		#img3 = torch.FloatTensor(img3)
		#gt_mask = torch.FloatTensor(gt_mask)

		input_dict = {'left_img':img1, 'right_img':img11, 'disp_img':img2_ori, 'rdisp_img':img22_ori, 'gt_mask' : gt_mask, 'disparity_img' : img2_disp_ori, 'rdisparity_img' : img22_disp_ori, 'gt_disp_mask' : gt_disp_mask}

		return input_dict

	def load_image4_disp(self, image): #lidar depth to disparity
		w, h = image.size
		imx_t = np.asarray(image)
		imx_t = baseline * width_to_focal[w] / imx_t
		nmimg = Image.fromarray(imx_t)

		return nmimg

	def load_image2(self,image):

		w, h = image.size
		imx_t = np.asarray(image)
		imx_t = imx_t / 256 
		imx_t = baseline * width_to_focal[w] / imx_t
		nmimg = Image.fromarray(imx_t)
		nmimg = nmimg.resize((480, 192))

		#imx_t = (np.asarray(nmimg))
		return nmimg
		#return imx_t

	def load_image2_mask(self,image):

		w, h = image.size
		imx_t = (np.asarray(image)) / 256
		imx_t = baseline * width_to_focal[w] / imx_t
		nmimg = Image.fromarray(imx_t)

		#imx_t = (np.asarray(nmimg))
		return nmimg
		#return imx_t

	def load_image2_disp(self,image):

		imx_t = np.asarray(image)
		imx_t = imx_t / 256
		nmimg = Image.fromarray(imx_t)
		nmimg = nmimg.resize((480, 192))

		return nmimg

	def load_image2_disp_mask(self,image):

		imx_t = np.asarray(image)
		imx_t = imx_t / 256
		nmimg = Image.fromarray(imx_t)

		return nmimg

	def load_image3(self,image):
		raw_image = image.resize((480, 192))
		#imx_t = np.asarray(raw_image)/256
		return raw_image
		#return imx_t

	def load_image4(self,image):
	
		imx_t = np.asarray(image)
		#imx_t = np.resize(imx_t, (375, 1242))
		#imx_t = downsample_nearest(imx_t)
		
		#imask = cv2.resize(imask, (480, 192), 0, 0, interpolation=cv2.INTER_LINEAR)
		return imx_t

	def mask_compare(self, img2, img4):
		#h, w = npimg2.shape

		#mask = np.zeros_like(npimg2, dtype=np.int32)
		#mask = torch.ones_like(img2) * -1
		invalid_index = torch.nonzero(img4 == 0)
		mask = ((torch.abs(img2 - img4)) <= 0.5).float()
		mask[invalid_index[:, 0], invalid_index[:, 1], invalid_index[:, 2]] = -1

		return mask

	def mask_compare_disp(self, img2_disp, img4):

		w = img4.shape[2] #torch.Size([1, 370, 1226])
		h = img4.shape[1]

		img4_disp = baseline * width_to_focal[w] / img4
		img4_disp[torch.isinf(img4_disp)] = 0

		invalid_index = torch.nonzero(img4_disp == 0)
		mask_disp = ((torch.abs(img2_disp - img4_disp)) <= 1).float()
		mask_disp[invalid_index[:, 0], invalid_index[:, 1], invalid_index[:, 2]] = -1

		return mask_disp

	def downsample_nearest(self, mask):

		ratio_x = mask.shape[1] / 192
		ratio_y = mask.shape[2] / 480

		valid_index = torch.nonzero(mask == 1)
		valid_index[:, 1] = (valid_index[:, 1].float() / ratio_x).long()
		valid_index[:, 2] = (valid_index[:, 2].float() / ratio_y).long()

		invalid_index = torch.nonzero(mask == -1)
		invalid_index[:, 1] = (invalid_index[:, 1].float() / ratio_x).long()
		invalid_index[:, 2] = (invalid_index[:, 2].float() / ratio_y).long()

		zero_index = torch.nonzero(mask == 0)
		zero_index[:, 1] = (zero_index[:, 1].float() / ratio_x).long()
		zero_index[:, 2] = (zero_index[:, 2].float() / ratio_y).long()

		downmask = torch.zeros((1, 192, 480))
		downmask[invalid_index[:, 0], invalid_index[:, 1], invalid_index[:, 2]] = -1
		downmask[valid_index[:, 0], valid_index[:, 1], valid_index[:, 2]] = 1
		downmask[zero_index[:, 0], zero_index[:, 1], zero_index[:, 2]] = 0

		return downmask

	def augument_image_pair(self, left_image, disp_image, conf_image, lidar_image):

		left_image = np.array(left_image)

		# print(np.amin(left_image))

		# randomly gamma shift
		random_gamma = random.uniform(0.8, 0.9)
		left_image_aug  = left_image  ** random_gamma

		random_brightness = random.uniform(0.5, 0.8)
		left_image_aug  =  left_image_aug * random_brightness

		left_image_aug = Image.fromarray(np.uint8(left_image_aug))


		return left_image_aug, disp_image, conf_image, lidar_image

	def __len__(self):
		return len(self.__left)

class KITTIValDataloader(Dataset):

	__left = []
	__disp = []

	def __init__(self, filename, img_root, disp_root):
		self.filename = filename
		self.img_root = img_root
		self.disp_root = disp_root

		arr = open(self.filename, "rt").read().split("\n")[:-1]

		n_line = open(self.filename).read().count('\n')
		#print(type(n_line))
		for line in range(n_line):
			self.__left.append(self.img_root + arr[line])
			self.__disp.append(self.disp_root + arr[line])


	def __getitem__(self, index):
		img1 = Image.open(self.__left[index]).convert('RGB')
		img2 = Image.open(self.__disp[index])

		#img1, img2 = self.augument_image_pair(img1, img2)

		transform = get_transform()
		img1 = transform(img1)

		img2 = self.load_image2(img2)

		img2 = torch.FloatTensor(img2)

		input_dict = {'left_img':img1, 'disp_img':img2}

		return input_dict

	def load_image2(self,image):
		"""
		raw_image = image.resize((512, 256))
		imx_t = np.asarray(raw_image)/256
		"""
		w, h = image.size
		imx_t = (np.asarray(image)) / 256
		imx_t = baseline * width_to_focal[w] / imx_t
		nmimg = Image.fromarray(imx_t)
		nmimg = nmimg.resize((480, 192))
		imx_t = (np.asarray(nmimg))

		return imx_t

	def augument_image_pair(self, left_image, disp_image):

		left_image = np.array(left_image)

		# print(np.amin(left_image))

		# randomly gamma shift
		random_gamma = random.uniform(0.8, 0.9)
		left_image_aug  = left_image  ** random_gamma

		random_brightness = random.uniform(0.5, 0.8)
		left_image_aug  =  left_image_aug * random_brightness

		left_image_aug = Image.fromarray(np.uint8(left_image_aug))


		return left_image_aug, disp_image

	def __len__(self):
		return len(self.__left)



