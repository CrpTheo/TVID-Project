import numpy as np
import os
import sys

from deinterlacer import deinterlace_bob


def get_deinterlace_mask(input_dir: str) -> np.ndarray:
    """
    Returns a mask of which frames should be deinterlaced.
    """
    metadata_file = os.path.join(input_dir, "metadata")
    with open(metadata_file, "r") as f:
        metadata = f.read().split("\n")

    total_frames = len(os.listdir(input_dir)) - 1
    
    i, prog_count, overflow = 0, 0, 0
    TFF_mask = np.zeros(total_frames).astype(np.uint8)
    for line in metadata:
        if len(line) > 15:
            continue

        if i + overflow >= total_frames:
            overflow += 1
            continue

        if line.__contains__("PROG"):
            TFF_mask[i] = 0
            prog_count += 1
        elif line.__contains__("TFF"):
            TFF_mask[i] = 1
        elif line.__contains__("BFF"):
            TFF_mask[i] = 2
        else:
            raise ValueError("Invalid metadata line: %s" % line)
        
        i += 1

    print(TFF_mask)
    print(f"Total frames to deinterlace: {total_frames - prog_count}, from {total_frames} input frames.")
    print(f"Overflow: {overflow} frames.")

    return TFF_mask


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 deinterlace.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    deinterlace_mask = get_deinterlace_mask(input_dir)
    deinterlace_bob(input_dir, output_dir, deinterlace_mask)