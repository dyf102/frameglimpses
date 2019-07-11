#!/usr/bin/env python
# coding: utf-8
import json
import os
import time
from multiprocessing import Pool, Process, Queue, current_process, JoinableQueue

import os
os.environ["OMP_NUM_THREADS"] = "1"


import cv2
import numpy as np
import h5py
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from class_filter import filter_video_classes, load_class_map
from cnn_features import Vgg16
from downsampling import downsampling


#@profile
def producer(input_q, output_q, global_config):
    print("Start divide videos to chunks {}".format(os.getpid()))
    for class_id, file_name, file_full_path in iter(input_q.get, None):
        for start_frame_num, end_frame_num, datum in read_video_to_chunk_frames(file_full_path, global_config["chunk_size"]):
            output_q.put((class_id, file_name, start_frame_num, end_frame_num, datum))
            # print((class_id, file_name, start_frame_num, end_frame_num, datum))
        input_q.task_done()
    output_q.put(None)
    print("Done the worker")

def get_vgg_model():
    model = Vgg16()
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'), strict=False)
    return model.cuda()
#@profile
def downsampling_worker(input_q, output_q, config) -> None:
    print("Start downsampling, pid: {}".format(os.getpid()))
    input_folder = config["raw_video_path"]
    output_folder = config["down_sampling_path"]
    for class_id, file_name, start_frame_num, end_frame_num in iter(input_q.get, None):
        file_full_path = downsampling(input_folder, file_name, output_folder, start_frame_num, end_frame_num,fps=5)
        output_q.put((class_id, file_name, file_full_path))
        #break
    output_q.put(None)

def consumer(input_q, output_q, config) -> None:
    print("Start Consumer, pid: {}".format(os.getpid()))
    metadata = []
    model = get_vgg_model()
    model.eval()
    #batch_size = 5 * config["chunk_size"]
    with torch.no_grad():
        count = 0
        total_time = 0
        for class_id, file_name, start_frame_num, end_frame_num, datum in iter(input_q.get, None):
            t1 = time.time()
            meta = {}
            meta["vidName"] = file_name
            meta["seq"] = [start_frame_num, end_frame_num]
            meta["dets"] = [{} for _ in range(20)]
            meta["dets"][int(class_id) - 1] = [start_frame_num, end_frame_num]
            metadata.append(meta)
            images = torch.from_numpy(datum).float().cuda(non_blocking=True)
            output = model(images).cpu().numpy().astype(np.float32)
            del datum

            output_q.put((file_name, class_id, output))
            input_q.task_done()
            t2 = time.time()
            total_time += (t2 - t1)
            count += 1
            if count % 10 == 0:
                print("{},{},{} {}, {}".format(file_name, start_frame_num, end_frame_num, total_time / 10, count))
                total_time = 0
    with open(config["val"]["features_meta_file"], "w+") as f:
        f.write(json.dumps(metadata))
    output_q.put(None)

#@profile
def write_to_hdf5(input_q, config):
    print("start saving to hdf5, pid: {}".format(os.getpid()))
    class_id_dict = {}
    total_time = 0
    count = 0
    for _, class_id, datum in iter(input_q.get, None):
        datum = np.expand_dims(datum, axis = 0)
        if class_id in class_id_dict.keys():
            data_list = class_id_dict[class_id]
            data_list.append(datum)
            class_id_dict[class_id] = data_list
        else:
            class_id_dict[class_id] = [datum]
        input_q.task_done()
        t2 = time.time()
        count += 1
        #if count % 10 == 0:
        print("write_to_hdf5: {},{}".format(count, class_id))
        #    total_time = 0
    
    f = h5py.File(config["features_file"], 'w')
    for k, v in tqdm(class_id_dict.items()):
        f.create_dataset('data/' + k, data=np.concatenate(v, axis=0), dtype='f')
    f.close()

def read_video_to_chunk_frames(input_loc, chunk=10):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        chunk: output data by chunk
    Returns:
        Generator
    """
    print(input_loc)
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # print ("Number of frames: ", video_length, "chunk:", chunk)
    count = 0
    # Start converting the video
    buffer = np.zeros((chunk, 3, 180, 320))
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
            buffer = np.zeros((chunk, 3, 180, 320))
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
    feature_q = Queue()

    config = load_config("config.json")
    files = os.listdir(config["down_sampling_path"])

    count = 0
    for (class_id, file_name) in filter_video_classes(files, config):
        file_full_path = os.path.join(config["down_sampling_path"], file_name)
        input_q.put((class_id, file_name, file_full_path))
        count += 1
    input_q.put(None)

    p1 = Process(target=producer, args=(input_q, output_q))

    p2 = Process(target=consumer, args=(output_q, feature_q))
    p3 = Process(target=consumer, args=(output_q, feature_q))

    p4 = Process(target=write_to_hdf5, args=(feature_q,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    try:
        p1.join()
        p2.join()
        p3.join()
        p4.join()
    except KeyboardInterrupt:
        print("Stop P1 P2 P3")
        p1.terminate()
        p2.terminate()
        p3.terminate()
        p4.terminate()

def load_val_meta(config):
    meta_val_file = config["meta_file_path"]
    df = pd.read_csv(meta_val_file).sort_values(by=['video'])
    for index, row in df.iterrows():
        yield row["video"], row["type_idx"], row["startFrame"], row["endFrame"]

def process_val_dataset():
    config = load_config("config.json")
    val_config = config["val"]

    input_q = JoinableQueue()
    sampling_output_q = JoinableQueue(5000)
    producer_output_q = JoinableQueue(5000)
    feature_q = JoinableQueue(50000)

    required_classes_map = load_class_map(config["class_id_map"])
    print(required_classes_map)
    for file_name, class_id, start_frame_num, end_frame_num in load_val_meta(val_config):
        class_id = str(class_id)
        if class_id in required_classes_map.keys():
            new_class_id = required_classes_map[class_id]
            input_q.put((new_class_id, "{}.mp4".format(file_name), 
                start_frame_num, end_frame_num))
    input_q.put(None)
    p1 = Process(target=downsampling_worker, args=(input_q, sampling_output_q, val_config))
    p2 = Process(target=producer, args=(sampling_output_q, producer_output_q, config))
    p3 = Process(target=consumer, args=(producer_output_q, feature_q, config))
    # p5 = Process(target=consumer, args=(producer_output_q, feature_q, config))
    p4 = Process(target=write_to_hdf5, args=(feature_q, val_config))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    # p5.start()
    try:
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        #p5.join()
    except KeyboardInterrupt:
        print("Stop P1 P2 P3")
        p1.terminate()
        p2.terminate()
        p3.terminate()
        p4.terminate()
        #p5.terminate()

if __name__ == '__main__':
    process_val_dataset()
    #for start_frame_num, end_frame_num, datum in read_video_to_chunk_frames("/home/yuwei/dataset/UCF101-val-5fps/2022_2275_video_validation_0000051.mp4", chunk=10):
    #   print(start_frame_num, end_frame_num, datum.shape)
