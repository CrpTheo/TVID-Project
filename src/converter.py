import numpy as np
import matplotlib.pyplot as plt


def yuv_to_ppmrgb(yuv_entry):
    """
    Convert a YUV image to RGB.
    """
    yuv_entry = np.reshape(yuv_entry, (-1, 3))
    yuv_entry = yuv_entry.astype(np.float32)

    y = yuv_entry[:720, :720]
    u = yuv_entry[720:, :720]
    v = yuv_entry[720:, 720:]

    r = y + 1.402 * (v - 128)
    g = y - 0.344136 * (u - 128) - 0.714136 * (v - 128)
    b = y + 1.772 * (u - 128)

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    rgb_image = np.stack((r, g, b), axis=-1)
    rgb_image = rgb_image.astype(np.uint8)

    return rgb_image
    

def open_yuv(filename, width, height):
    """
    Open a YUV image file and return a numpy array containing the image data.
    """
    with open(filename, "rb") as f:
    # read the header
        header = f.readline()
        assert header == b"P5\n"

        # read the size
        size = f.readline()
        width, height = [int(x) for x in size.split()]

        # read the max value
        maxval = f.readline()
        assert maxval == b"255\n"

        # read the data
        data = f.read()

        # convert the data to a numpy array
        img = np.frombuffer(data, dtype=np.uint8)
        img = img.reshape((height, width))

        return img 

def save_rgb(filename, rgb_image):
    """
    Save a ppm RGB encoded image to a file.
    """
    with open(filename, 'wb') as file:
        file.write(b'P6\n')
        file.write(b'# Created by YUV2RGB\n')
        file.write(b'720 720\n')
        file.write(b'255\n')
        file.write(rgb_image.tobytes())

if __name__ == '__main__':
    filename = input('Enter the filename of the YUV image: ')

    yuv_image = open_yuv(filename, 720, 720)

    rgb_image = yuv_to_ppmrgb(yuv_image)

    save_rgb('rgb_image.ppm', rgb_image)

    plt.imshow(rgb_image)
