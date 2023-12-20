import numpy as np
import matplotlib.pyplot as plt


file = "data/elementary/lci/2676.pgm"


def read_yuv_pgm(filename):
    """
    Reads a YUV PGM file into a np array.
    """
    with open(filename, "rb") as f:
        header = f.readline()
        assert header == b"P5\n"

        size = f.readline()
        width, height = [int(x) for x in size.split()]

        img = plt.imread(filename).astype(np.uint8)
        img = img.reshape((height, width))

    y_width = width
    y_height = height * 2 // 3
    uv_width = width // 2

    y = img[:y_height, :y_width]

    u = img[y_height:, :uv_width]
    v = img[y_height:, uv_width:]

    u = np.repeat(u, 2, axis=0)
    v = np.repeat(v, 2, axis=0)

    u = np.repeat(u, 2, axis=1)
    v = np.repeat(v, 2, axis=1)

    return np.dstack((y, u, v)).astype(np.uint8)

def yuv_to_rgb(yuv_img):
    """
    Converts a YUV image to RGB.
    """
    y = yuv_img[:, :, 0].astype(np.float32)
    u = yuv_img[:, :, 1].astype(np.float32)
    v = yuv_img[:, :, 2].astype(np.float32)

    r = np.clip(y + 1.402 * (v - 128), 0, 255)
    g = np.clip(y - 0.344136 * (u - 128) - 0.714136 * (v - 128), 0, 255)
    b = np.clip(y + 1.772 * (u - 128), 0, 255)

    return np.stack((r, g, b), axis=-1).astype(np.uint8)

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