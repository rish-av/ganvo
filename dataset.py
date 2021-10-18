import torch
import os
from glob import glob
import cv2
from torchvision import transforms


class dataset(torch.utils.data.Dataset):
	def __init__(self,config):
            self.config = config
            self.basepath = config.basepath

            sequences = glob(self.basepath + "/*")

            images = []

            for seq in sequences:
                image_0_paths = sorted(glob(seq + "/image_0/*"))
                image_1_paths = sorted(glob(seq + "/image_1/*"))
                image_2_paths = sorted(glob(seq + "/image_2/*"))
                image_3_paths = sorted(glob(seq + "/image_3/*"))

                for img0, img1, img2, img3 in zip(image_0_paths, image_1_paths, image_2_paths, image_3_paths):
                    images.append((img0,img1,img2,img3))

            self.images = images



	def __getitem__(self, index):
            
            seq = self.images[index]

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(std=(0.5,),mean=(0.5,))
                ])

            img0 = transform(cv2.resize(cv2.imread(seq[0]), (self.config.w, self.config.h)))
            img1 = transform(cv2.resize(cv2.imread(seq[1]), (self.config.w, self.config.h)))
            img2 = transform(cv2.resize(cv2.imread(seq[2]), (self.config.w, self.config.h)))
            img3 = transform(cv2.resize(cv2.imread(seq[3]), (self.config.w, self.config.h)))
            datum = {"t0":img0, "t1":img1, "t2":img2, "t3":img3}

            return datum

	def __len__(self):

            return len(self.images)
