
import os
import json

import ffmpeg


def downsampling(input_folder, filename, output_folder, fps=5):
    input_file = os.path.join(input_folder, filename)
    output_file = os.path.join(output_folder, filename)
    (
        ffmpeg
        .input(input_file)
        .filter('fps', fps=fps, round='up')
        .output(output_file)
        .run()
    )

def load_config(config_path):
    return json.loads(open(config_path).read())

if __name__ == '__main__':
    config = load_config("config.json")
    for file in os.listdir(config["raw_video_path"]):
        downsampling(config["raw_video_path"], file, config["down_sampling_path"])