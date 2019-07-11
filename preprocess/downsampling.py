
import os
import json
import subprocess
import datetime
import math

def read_frame(fileloc,start_frame_num,fps,end_frame_num,output_file) :
    ORIGINAL_FPS = 30.0
    command = ['ffmpeg',
               '-loglevel', 'quiet',
               '-ss', str(datetime.timedelta(seconds=start_frame_num/ORIGINAL_FPS)),

               '-i', fileloc,

               #'-vf', 'select=between(n\\,{}\\,{}),setpts=PTS-STARTPTS'.format(frame, end_frame_num),
               # '-vf', 'scale=%d:%d'%(t_w,t_h),
               '-vframes', str(math.ceil((end_frame_num - start_frame_num + 1) * fps / ORIGINAL_FPS)),
               '-r', str(fps),
               output_file
               ]
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
    out, err = ffmpeg.communicate()
    if(err) : print('error',err); return None;

def downsampling(input_folder, filename, output_folder, 
    start_frame_num=None, end_frame_num=None ,fps=5, skip_existed=True):
    """
    downsample the video file to specfic fps

    """
    input_file = os.path.join(input_folder, filename)
    output_file = os.path.join(output_folder, filename)
    if start_frame_num is None:
        if skip_existed and os.path.exists(output_file):
            return output_file
        (
            ffmpeg
            .input(input_file)
            .filter('fps', fps=fps, round='up')
            .output(output_file)
            .run(overwrite_output=False)
        )
    else:
        output_file = "{}{}_{}_{}".format(output_folder, start_frame_num, end_frame_num, filename)
        if skip_existed and os.path.exists(output_file):
            print("Skipped file: " + output_file)
            return output_file
        read_frame(input_file, start_frame_num , fps, end_frame_num, output_file)
    return output_file

def load_config(config_path):
    return json.loads(open(config_path).read())

if __name__ == '__main__':
    downsampling("/home/yuwei/dataset/UCF101/validation", 
        "video_validation_0000051.mp4", 
        "/home/yuwei/", 
        start_frame_num=2022, 
        end_frame_num=2275, 
        fps=5)
