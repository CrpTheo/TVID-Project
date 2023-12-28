import sys
from src.display import write_all_frames
from src.video_compiler import create_video_from_ppm
from src.parser import get_deinterlace_mask, deinterlace_bob

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 deinterlace.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    deinterlace_mask = get_deinterlace_mask(input_dir)
    deinterlace_bob(input_dir, output_dir, deinterlace_mask)

    write_all_frames(output_dir, f"{output_dir}_rgb", len(deinterlace_mask))
    create_video_from_ppm(output_dir, output_dir + ".mp4")