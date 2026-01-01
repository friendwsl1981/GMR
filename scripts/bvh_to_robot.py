import argparse
import pathlib
import time
import sys
import select
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from rich import print
from tqdm import tqdm
import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "figurobot_2nd"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )
    
    parser.add_argument(
        "--max_frames",
        default=None,
        type=int,
        help="Maximum number of frames to process.",
    )

    parser.add_argument(
        "--pause_after_first_frame",
        action="store_true",
        default=False,
        help="Pause after rendering the first frame (after viewer.sync). Useful for comparing coordinate frames.",
    )

    parser.add_argument(
        "--dump_qpos",
        default=None,
        type=str,
        help="Dump the first frame's full mujoco qpos (root + joints) to a .npy file.",
    )

    parser.add_argument(
        "--dump_rot_offsets",
        default=None,
        type=str,
        help=(
            "Dump suggested rot_offset (wxyz) for each ik_match_table1 entry to a JSON file. "
            "Computed from first frame as: R_offset = inv(R_human) * R_robot."
        ),
    )

    parser.add_argument(
        "--dump_target_frames",
        default=None,
        type=str,
        help=(
            "Dump the BVH-driven target frames shown in viewer1 (retargeter.scaled_human_data) to JSON. "
            "This can be loaded into the second viewer to manually align robot joint frames."
        ),
    )
    
    args = parser.parse_args()
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    if args.max_frames:
        lafan1_data_frames = lafan1_data_frames[:args.max_frames]
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = args.motion_fps
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while True:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)

        if args.dump_qpos is not None:
            dump_dir = os.path.dirname(args.dump_qpos)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            np.save(args.dump_qpos, qpos)
            print(f"Dumped qpos to {args.dump_qpos}")
            args.dump_qpos = None

        if args.dump_target_frames is not None:
            dump_dir = os.path.dirname(args.dump_target_frames)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            # retargeter.scaled_human_data: {human_body_name: (pos, quat_wxyz)}
            payload = {
                k: {
                    "pos": [float(x) for x in v[0]],
                    "quat": [float(x) for x in v[1]],
                }
                for k, v in retargeter.scaled_human_data.items()
            }
            with open(args.dump_target_frames, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"Dumped target frames to {args.dump_target_frames}")
            args.dump_target_frames = None
        

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=True,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        if args.dump_rot_offsets is not None:
            dump_dir = os.path.dirname(args.dump_rot_offsets)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)

            suggested_table = {}
            missing = []
            for frame_name, entry in retargeter.ik_match_table1.items():
                human_body_name, pos_weight, rot_weight, pos_offset, _rot_offset = entry

                if human_body_name not in retargeter.scaled_human_data:
                    missing.append((frame_name, human_body_name, "human_missing"))
                    continue

                try:
                    body_id = robot_motion_viewer.model.body(frame_name).id
                except Exception:
                    missing.append((frame_name, human_body_name, "robot_body_missing"))
                    continue

                _, human_quat_wxyz = retargeter.scaled_human_data[human_body_name]
                human_R = R.from_quat(human_quat_wxyz, scalar_first=True)

                robot_xmat = robot_motion_viewer.data.xmat[body_id].reshape(3, 3)
                robot_R = R.from_matrix(robot_xmat)

                rot_offset = (human_R.inv() * robot_R).as_quat(scalar_first=True)
                rot_offset = [float(x) for x in rot_offset]

                suggested_table[frame_name] = [
                    human_body_name,
                    pos_weight,
                    rot_weight,
                    pos_offset,
                    rot_offset,
                ]

            payload = {
                "robot_root_name": retargeter.robot_root_name,
                "human_root_name": retargeter.human_root_name,
                "ground_height": float(np.linalg.norm(retargeter.ground)),
                "human_height_assumption": None,
                "use_ik_match_table1": True,
                "use_ik_match_table2": False,
                "human_scale_table": retargeter.human_scale_table,
                "ik_match_table1": suggested_table,
            }

            with open(args.dump_rot_offsets, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            print(f"Dumped suggested rot_offsets to {args.dump_rot_offsets}")
            if missing:
                print(f"[yellow]Skipped {len(missing)} entries (missing human body or robot body).[/yellow]")
                for item in missing[:10]:
                    print("  ", item)
                if len(missing) > 10:
                    print("  ...")
            args.dump_rot_offsets = None

        if args.pause_after_first_frame:
            args.pause_after_first_frame = False
            print("Paused after first frame. Press Enter to continue...")
            # Keep rendering so the window stays interactive (camera/mouse), and poll stdin for Enter.
            while True:
                robot_motion_viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=False,
                    follow_camera=False,
                )
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.readline()
                    break
                time.sleep(1.0 / 60.0)

        if args.loop:
            i = (i + 1) % len(lafan1_data_frames)
        else:
            i += 1
            if i >= len(lafan1_data_frames):
                break
   
        
        if args.save_path is not None:
            qpos_list.append(qpos)
    
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
