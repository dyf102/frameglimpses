#!/usr/bin/env python
# coding: utf-8
import json
import os
import time
from multiprocessing import Pool, Process, Queue, current_process, freeze_support



import cv2
import numpy as np
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


from class_filter import filter_video_classes

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

def producer(input_q, output_q):
    for class_id, file_name, file_full_path in iter(input_q.get, None):
        for start_frame_num, end_frame_num, datum in  read_video_to_chunk_frames(file_full_path, 10):
            output_q.put((class_id, file_name, start_frame_num, end_frame_num, datum))
    output_q.put(None)
    print("Done the worker")

def get_vgg_model():
    model = MyVgg()
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'), strict=False)
    return model.cuda()

def consumer(input_q):
    print("Start Consumer")
    metadata = []
    model = get_vgg_model()
    model.eval()
    with torch.no_grad():
        for class_id, file_name, start_frame_num, end_frame_num, datum in iter(input_q.get, None):
            meta = {}
            meta["vidName"] = file_name
            meta["seq"] = [start_frame_num, end_frame_num]
            meta["dets"] = [{} for _ in range(20)]
            meta["dets"][int(class_id) - 1] = [start_frame_num, end_frame_num]
            metadata.append(meta)
            images = torch.from_numpy(datum).float().cuda(non_blocking=True)
            output = model(images)
            print(output.shape)
            break


    with open("matedata.json", "w+") as f:
        f.write(json.dumps(metadata))

def read_video_to_chunk_frames(input_loc, chunk=10):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        chunk: output data by chunk
    Returns:
        Generator
    """
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # print ("Number of frames: ", video_length, "chunk:", chunk)
    count = 0
    # Start converting the video
    buffer = np.zeros((chunk, 3,240,320))
    while cap.isOpened():
        # Extract the frame
        ret, img = cap.read()
        if img is None or ret is False:
            # print(input_loc, count, video_length)
            break
        # Write the results back to output location.
        # cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        img = img.transpose((2, 0, 1)) / 255.0
        buffer[count % chunk] = img
        
        if (count + 1) % chunk == 0:
            yield (count - chunk + 1, count, buffer)
            buffer = np.zeros((chunk, 3, 240, 320))
        count = count + 1
        # If there are no more frames left
        if (count > (video_length)):
            if count % chunk != 0:
                yield (count - (count % chunk), count, buffer)
                # Release the feed
            cap.release()
            # Print stats
            break

def load_config(config_path):
    return json.loads(open(config_path).read())

def main():
    input_q = Queue()
    output_q = Queue()


    config = load_config("config.json")
    files = os.listdir(config["down_sampling_path"])

    count = 0
    for (class_id, file_name) in filter_video_classes(files, config):
        file_full_path = os.path.join(config["down_sampling_path"], file_name)
        input_q.put((class_id, file_name, file_full_path))
        count += 1
    input_q.put(None)

    p1 = Process(target=producer, args=(input_q, output_q))

    p2 = Process(target=consumer, args=(output_q, ))
    p1.start()
    p2.start()
    
    print("Start processing result")
    p1.join()
    p2.join()

if __name__ == '__main__':
    main()
    #for start_frame_num, end_frame_num, datum in read_video_to_chunk_frames("/home/yuwei/v_YoYo_g25_c05.avi", chunk=10):
    #   print(start_frame_num, end_frame_num, datum.shape)
