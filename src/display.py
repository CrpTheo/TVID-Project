import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import time

from PIL import Image
from converter import yuv_to_rgb
from multiprocessing import Pool
from itertools import repeat


def compute_frame(frame_name: str) -> np.ndarray:
    img = np.array(Image.open(frame_name)).astype(np.float16)
    return yuv_to_rgb(img)

def update(frame_name):
    img = compute_frame(f"data/elementary/lci/{frame_name}.pgm")
    im.set_array(img)

    return im,

def count_files(directory: str) -> int:
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)

    return file_count

def get_sequence_infos(directory: str) -> (int, int, int, np.ndarray):
    num_files = count_files(directory)
    img = yuv_to_rgb(plt.imread(f"{directory}/0.pgm").astype(np.float16))
    height, width, _ = img.shape

    return height, width, num_files, img

def compute_and_write_frame(frame_data):
    frame_name, input_dir, output_dir = frame_data
    img = compute_frame(os.path.join(input_dir, frame_name))
    cv2.imwrite(os.path.join(output_dir, frame_name.replace(".pgm", ".ppm")), img)

def write_all_frames(input_dir: str, output_dir: str, num_files: int) -> None:
    with Pool() as pool:
        frame_names = [f"{i}.pgm" for i in range(num_files)]
        pool.map(compute_and_write_frame, zip(frame_names, repeat(input_dir), repeat(output_dir)))


if __name__ == "__main__":
    args = sys.argv[1:]

    directory = args[0]
    
    if args[1] == "display":
        if len(args) == 3:
            frame_rate = int(args[2])

        else:
            frame_rate = 25

        height, width, num_files, frame0 = get_sequence_infos(directory)

        fig, ax = plt.subplots()

        im = ax.imshow(frame0)

        ani = animation.FuncAnimation(fig, update, frames=range(1, num_files), interval=int(1000/frame_rate), blit=True, repeat=False)
        start_time = time.time()
        plt.show()
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Time to read and write all ppm: {elapsed_time} seconds : {elapsed_time/60} ips")
    
    elif args[1] == "write":
        num_files = count_files(directory)

        start_time = time.time()
        write_all_frames(directory, "data/elementary/lci_rgb", num_files)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Time to read and write all ppm: {elapsed_time} seconds : {num_files/elapsed_time} ips")
