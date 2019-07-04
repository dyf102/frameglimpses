#!/usr/bin/env python
# coding: utf-8

import os
import sys


def load_class_map(file_path):
    if not os.path.exists(file_path):
        print("%s not exists".format(file_path))
        return
    return dict(((line.split(" ")[1].strip(), line.split(" ")[0].strip()) for line in open(file_path)))


def filter_video_classes(video_files, config):
    required_classes_map = load_class_map(config["class_id_map"])
    class_id_map = load_class_map(config["class_name_map"])
    for file_name in video_files:
        class_name = file_name.split("_")[1]
        class_id = class_id_map.get(class_name)
        if class_id is not None:
            new_class_id = required_classes_map.get(class_id)
            if new_class_id is not None:
                yield (new_class_id, file_name)



if __name__ == "__main__":
    config = {
        "class_id_map": "/home/yuwei/frameglimpses/thumos_class_mapping.txt",
        "class_name_map": "/home/yuwei/frameglimpses/classInd.txt",
        "raw_video_path": "/home/yuwei/dataset/UCF101",
        "down_sampling_path": "/home/yuwei/dataset/UCF101-5fps/"
    }
    for (class_id, file_name) in filter_video_classes( os.listdir(config["down_sampling_path"]), config):
        print(class_id, file_name)