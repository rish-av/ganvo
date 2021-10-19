import torch
import os
from glob import glob
import cv2
from torchvision import transforms
import numpy as np
from itertools import chain


class dataset(torch.utils.data.IterableDataset):
    def __init__(self,config,mode='train'):

        self.mode = mode
        self.config = config
        self.basepath = config.basepath

        sequences = glob(self.basepath + "/raw/sequences/*")

        image_2_seqs = {}
        image_3_seqs = {}

        image_2_seqs_test = {}
        image_3_seqs_test = {}

        count_train = 0
        count_test = 0

        self.test_seqs = config.test_seqs
        self.train_seqs = config.train_seqs

        for seq in sequences:
            seq_num = seq.split('/')[-1]
            if seq_num in self.test_seqs:

                image_2_seqs_test[seq_num] = sorted(glob(seq + "/image_2/*"))
                image_3_seqs_test[seq_num] = sorted(glob(seq + "/image_3/*"))
                count_test+=len(image_2_seqs_test[seq_num])
                count_test+=len(image_3_seqs_test[seq_num])

            else:
                image_2_seqs[seq_num] = sorted(glob(seq + "/image_2/*"))
                image_3_seqs[seq_num] = sorted(glob(seq + "/image_3/*"))

                count_train+=len(image_2_seqs[seq_num])
                count_train+=len(image_2_seqs[seq_num])

        self.image_2_seqs = image_2_seqs
        self.image_3_seqs = image_3_seqs
        self.image_2_seqs_test = image_2_seqs_test
        self.image_3_seqs_test = image_3_seqs_test
        self.count_train = count_train
        self.count_test = count_test

    def __iter__(self):
        if self.mode == 'train':
            im2_seq = self.image_2_seqs
            im3_seq = self.image_3_seqs
        else:
            im2_seq = self.image_2_seqs_test
            im3_seq = self.image_3_seqs_test


        for img_number in range(0,2):

            if img_number == 0:
                img_num_used = im2_seq
            else:
                img_num_used = im3_seq

            
            for seq_number_idx, v in img_num_used.items():
                sequence = img_num_used[seq_number_idx]
                for time_stamp_idx in range(1,len(sequence)-len(sequence)%3):
                    img0 = sequence[time_stamp_idx-1]
                    img1 = sequence[time_stamp_idx]
                    img2 = sequence[time_stamp_idx+1] 

                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(std=(0.5,),mean=(0.5,))
                        ])

                    img0 = transform(cv2.resize(cv2.imread(img0), (self.config.w, self.config.h)))
                    img1 = transform(cv2.resize(cv2.imread(img1), (self.config.w, self.config.h)))
                    img2 = transform(cv2.resize(cv2.imread(img2), (self.config.w, self.config.h)))
                    datum = {"t0":img0, "t1":img1, "t2":img2}

                    yield datum

    def __len__(self):
        if self.mode == 'train':
            return self.count_train
        else:
            return self.count_test
