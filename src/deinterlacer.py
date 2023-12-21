import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def deinterlace_bob(input_dir: str, output_dir: str, deinterlace_mask: np.ndarray, top_field_first=True) -> None:
    """
    Deinterlaces a PGM sequence using the BOB algorithm.
    """
    os.makedirs(output_dir, exist_ok=True)

    images = np.array([np.array(Image.open(os.path.join(input_dir, fname))) for fname in sorted(os.listdir(input_dir))])

    count = 0
    for i, img in enumerate(images):
        if deinterlace_mask[i]:
            if top_field_first:
                lines_top = img[::2]
                lines_bottom = img[1::2]
                
            else:
                lines_top = img[1::2]
                lines_bottom = img[::2]

            cv2.imwrite(os.path.join(output_dir, f"{count}.pgm"), np.repeat(lines_top, 2, axis=0))
            cv2.imwrite(os.path.join(output_dir, f"{count + 1}.pgm"), np.repeat(lines_bottom, 2, axis=0))
            count += 2

        else:
            cv2.imwrite(os.path.join(output_dir, f"{i}.pgm"), img)
            count += 1
    
    print(f"Total frames: {count}, from {len(images)} input frames.")


def interlace_weave(input_dir: str, output_dir: str) -> None:
    """
    Interlaces a PGM sequence using the weave algorithm.
    """
    os.makedirs(output_dir, exist_ok=True)

    images = [np.array(Image.open(os.path.join(input_dir, fname))) for fname in sorted(os.listdir(input_dir))]

    count = 0
    for i in range(0, len(images), 2):
        if i + 1 < len(images):
            lines_top = images[i]
            lines_bottom = images[i + 1]

            interlaced_frame = np.vstack((lines_top, lines_bottom))
            cv2.imwrite(os.path.join(output_dir, f"{count}.pgm"), interlaced_frame)
            count += 1
        else:
            cv2.imwrite(os.path.join(output_dir, f"{count}.pgm"), images[i])
            count += 1