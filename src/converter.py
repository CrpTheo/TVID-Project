import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


file = "data/elementary/lci/2676.pgm"


def read_yuv_pgm(filename: str) -> np.ndarray:
    """
    Reads a YUV PGM file into a np array.
    """
    return np.array(Image.open(filename)).astype(np.float16)

def yuv_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Converts a YUV image to RGB.
    """
    height, width = img.shape

    y_height = height * 2 // 3
    uv_width = width // 2

    y = img[:y_height, :width]

    v = img[y_height:, :uv_width] - 128
    u = img[y_height:, uv_width:] - 128

    u = u.repeat(2, axis=0).repeat(2, axis=1)
    v = v.repeat(2, axis=0).repeat(2, axis=1)

    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u

    return np.clip(np.stack((r, g, b), axis=-1), 0, 255).astype(np.uint8)

def write_ppm(filename, img):
    """
    Writes a PPM file from np array image
    """
    with open(filename, "wb") as f:
        f.write(b"P6\n")
        f.write(b"%d %d\n" % (img.shape[1], img.shape[0]))
        f.write(b"255\n")
        f.write(img.tobytes())

def rgb_from_yuv(filename, output_type="ppm"):
    """
    Converts a YUV PGM file to RGB PPM.
    """
    img = read_yuv_pgm(filename)
    img = yuv_to_rgb(img)

    if output_type == "ppm":
        write_ppm(filename[:-3] + "ppm", img)

    elif output_type == "display":
        plt.imshow(img)
        plt.show()

    else:
        raise ValueError("Invalid output type: %s" % output_type) 

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    rgb_from_yuv(file, output_type=args[0] if args else "ppm")