import argparse
import pickle
import numpy as np

def trim_pkl(input_path: str, output_path: str, start_frame: int, end_frame: int):
    with open(input_path, "rb") as f:
        motion_data = pickle.load(f)

    # Trim arrays
    for key in ["root_pos", "root_rot", "dof_pos"]:
        if key in motion_data:
            motion_data[key] = motion_data[key][start_frame:end_frame]

    if "local_body_pos" in motion_data and motion_data["local_body_pos"] is not None:
        motion_data["local_body_pos"] = motion_data["local_body_pos"][start_frame:end_frame]

    # Save trimmed data
    with open(output_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"Trimmed {input_path} from frame {start_frame} to {end_frame-1}, saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim frames from a GMR pkl file")
    parser.add_argument("--input_pkl", required=True, help="Path to input pkl file")
    parser.add_argument("--output_pkl", required=True, help="Path to output pkl file")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame (0-based)")
    parser.add_argument("--end_frame", type=int, required=True, help="End frame (exclusive)")

    args = parser.parse_args()
    trim_pkl(args.input_pkl, args.output_pkl, args.start_frame, args.end_frame)