import cv2
import glob
import sys
from natsort import natsorted


def create_video_from_ppm(input_folder, output_video, fps=30):
    ppm_files = natsorted(glob.glob(f"{input_folder}/*.ppm"))

    if not ppm_files:
        print("No PPM files found.")
        return

    first_image = cv2.imread(ppm_files[0])
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for ppm_file in ppm_files:
        img = cv2.imread(ppm_file)
        video_writer.write(img)

    video_writer.release()

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    fps = 25

    create_video_from_ppm(input_dir, output_file, fps)
