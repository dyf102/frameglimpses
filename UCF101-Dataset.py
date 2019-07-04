#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip3 uninstall opencv-python


# In[ ]:


#!pip3 install pretrainedmodels
#!CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd


import glob
import time
import os
import argparse
import json
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torchvision.models.vgg import VGG, make_layers, cfgs, vgg16
import torch.backends.cudnn as cudnn
import torch.distributed as dist


# In[2]:


# TRAINING_DATA_PATH = "/home/s1902744/disk2/UCF101"
TRAINING_DATA_PATH = "/home/yuwei/dataset/UCF101"
# CLASS_ID_DATA_PATH = '/home/s1902744/classInd.txt'
CLASS_ID_DATA_PATH = "/home/yuwei/classInd.txt"
args = {
    "batch_size": 100,
    "workers": 4, 
    "GPU":True 
}



# In[3]:



# generate_train_data_file(TRAINING_DATA_PATH)
#class_id_dict = load_class_id(CLASS_ID_DATA_PATH)

#filenames = os.listdir(TRAINING_DATA_PATH)
# print(filenames)

#classes = list(set(map(lambda x: x.split("_")[1], filenames)))
#groups = list(map(lambda x: x.split("_")[2], filenames))
#videos = list(map(lambda x: x.split("_")[3], filenames))
#class_ids = list(map(lambda x: class_id_dict[x], classes))


# In[4]:


def move_to_class_id(root, filenames, class_id_dict):
    for filename in filenames:
        try:
            class_name = filename.split("_")[1]
        except:
            continue
        des_folder = os.path.join(root, "new", class_id_dict[class_name])
        src_folder = os.path.join(root, filename)
        try:    
            os.mkdir(des_folder)
        except:
            pass
        i = 0
        for file in os.listdir(src_folder):
            os.rename(os.path.join(src_folder, file), os.path.join(des_folder, "%s_%s.jpg"%(filename, i)))
            # print()
        i += 1
# move_to_class_id(TRAINING_DATA_PATH, filenames, class_id_dict)
        


# In[5]:


random.seed(0)
torch.manual_seed(0)
if args["GPU"] == True:
    cudnn.deterministic = True


# In[6]:


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# train_dataset = ImageFolder(TRAINING_DATA_PATH,
#                      transforms.Compose([
#                         transforms.RandomResizedCrop(224),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.ToTensor(),
#                         normalize,
#                     ]))


# In[7]:


class MyVgg(VGG):

    def __init__(self):
        super().__init__(make_layers(cfgs['D']))
        self.classifier = self.classifier[:-1]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    def forward(self, x):
        # here, implement the forward function, keep the feature maps you like
        # and return them
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



def read_video_to_frames(input_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # print ("Number of frames: ", video_length)
    count = 0
    # Start converting the video
    buffer = np.zeros((video_length, 3,240,320))
    while cap.isOpened():
        # Extract the frame
        ret, img = cap.read()
        if img is None or ret is False:
            print(input_loc, count, video_length)
            break
        # Write the results back to output location.
        # cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        img = img.transpose((2, 0, 1)) / 255.0
        buffer[count] = img
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            break
    return buffer


# In[12]:


def get_video_frames(file_path):
    if file_path is None:
        return 0
    cap = cv2.VideoCapture(file_path)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    length = int(cv2.VideoCapture.get(cap, property_id))
    return length

# def generate_train_data_file(directory_path, class_id_dict):
#     for file in glob.glob(os.path.join(directory_path, "*.avi")):
#         class_name = file.split("_")[1]
#         class_id = class_id_dict[class_name]
#         try:
#             os.mkdir(ps.path.join(directory_path), class_id)
#         for (file_path, data) in video_to_frames(file, folder):
            


# In[13]:


class CustomDatasetFromVide(Dataset):
    def _load_video_lens(self, tmp_folder):
        tmp_file_path = os.path.join(tmp_folder, "video_lens_cache.json")
        try:
            if not os.path.exists(tmp_folder):
                os.mkdir(tmp_folder)
            self.video_lens = json.loads(open(tmp_file_path, "r").read())
        except:
            self.video_lens = list(map(lambda x: get_video_frames(x), self.video_file_full_path))
            with open(tmp_file_path, "w+") as f:
                f.write(json.dumps(self.video_lens))
    
    def _compute_video_lens_sum(self):
        self.data_each_len = []
        tmp_sum = 0
        for i in self.video_lens:
            tmp_sum += i
            self.data_each_len.append(tmp_sum)

    def __init__(self, video_path, class_id_file_path, tmp_folder="./tmp/"):
        self.video_path = video_path
        self.video_file_full_path = list(map(lambda x: os.path.join(video_path, x), os.listdir(video_path)))
        self.video_filename = os.listdir(video_path)
        
        self._load_video_lens(tmp_folder)
        self._compute_video_lens_sum()
        
        self.data_total_len = sum(self.video_lens)
        
        # Load class id file to a dict
        self.class_id_dict = self.load_class_id(class_id_file_path)
        # Convert the class name from file name to class id
        self.labels = list(map(lambda x: self.class_id_dict[x.split("_")[1]], self.video_filename))
        # A list of data for each video file
        self.data = [None for _ in self.video_filename]
        
    def __getitem__(self, index):
        # print(self.data_each_len)
        for i,c in enumerate(self.data_each_len):
            if index < c:
                if self.data[i] is None:
                    self.data[i] = read_video_to_frames(self.video_file_full_path[i])
                if i > 0:
                    frame_idx = index - self.data_each_len[i - 1]
                    return (self.labels[i], self.video_filename[i],  self.data[i][frame_idx])
                else:
                    print(i, index, c, self.data[i].shape)
                    return (self.labels[i], self.data[i][index])
        raise IndexError()
    def __len__(self):
        return self.data_total_len
    
    def load_class_id(self, filename):
        f = open(filename)
        return dict(map(lambda x: (x.split(" ")[1].rstrip(), x.split(" ")[0]), f))



# In[ ]:





# In[ ]:


train_dataset = CustomDatasetFromVide(TRAINING_DATA_PATH, CLASS_ID_DATA_PATH)

model = MyVgg()
print(model)
model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'), strict=False)

if args["GPU"]:
    model = model.cuda()
#data = model(torch.rand(1, 3, 224, 224))
#print(data.shape)
# In[ ]:

#vgg16 = models.vgg16(pretrained=True)
#print(vgg16.features)
# cut the part that i want:

#model = pretrainedmodels.__dict__['vgg16']()
# del model.last_linear
model.eval()

train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["workers"], pin_memory=True, sampler=None)

#class_names = train_dataset.classes
#len(class_names), len(train_dataset)


# In[10]:


with torch.no_grad():
     for i, (target, images) in enumerate(train_loader):
         start_t = time.time()
         if args["GPU"]:
              images = images.float().cuda(non_blocking=True)
         output = model(images)
         end_t = time.time()
         print(output.shape, end_t - start_t)


# In[ ]:


#vgg16 = pretrainedmodels.__dict__['vgg16']()


# In[ ]:


#del vgg16.last_linear


# In[ ]:


#vgg16


# In[ ]:



#for file in glob.glob(os.path.join(directory_path, "*.avi")):


# In[ ]:




