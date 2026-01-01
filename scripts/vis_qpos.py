import argparse
import os
import numpy as np
import mujoco as mj
import time
import json
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.robot_motion_viewer import draw_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "pal_talos",
            "figurobot_2nd",
        ],
        required=True,
    )
    parser.add_argument(
        "--qpos",
        required=True,
        type=str,
        help="Path to a .npy file dumped by scripts/bvh_to_robot.py --dump_qpos",
    )
    parser.add_argument(
        "--show_body_frames",
        action="store_true",
        default=True,
        help="Show body coordinate frames in the MuJoCo viewer.",
    )

    parser.add_argument(
        "--targets",
        default=None,
        type=str,
        help="Path to target frames JSON dumped by scripts/bvh_to_robot.py --dump_target_frames",
    )
    parser.add_argument(
        "--ik_config",
        default=None,
        type=str,
        help="IK config JSON (to know frame_name -> human body mapping) for dumping rot_offsets.",
    )
    parser.add_argument(
        "--dump_rot_offsets_out",
        default=None,
        type=str,
        help="If set, dump rot_offset(wxyz) per ik_match_table1 frame_name when you exit (Ctrl-C or close window).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.qpos):
        raise FileNotFoundError(args.qpos)

    qpos = np.load(args.qpos)
    if qpos.ndim != 1:
        raise ValueError(f"Expected 1D qpos array, got shape {qpos.shape}")

    targets = None
    if args.targets is not None:
        with open(args.targets, "r", encoding="utf-8") as f:
            raw = json.load(f)
        targets = {
            k: (np.array(v["pos"], dtype=float), np.array(v["quat"], dtype=float))
            for k, v in raw.items()
        }

    viewer = RobotMotionViewer(robot_type=args.robot, motion_fps=30, transparent_robot=0)

    if args.show_body_frames:
        viewer.viewer.opt.frame = mj.mjtFrame.mjFRAME_BODY

    # Initialize pose once, then DO NOT overwrite qpos each frame.
    # This keeps the viewer interactive so you can adjust joints in the UI.
    viewer.data.qpos[: len(qpos)] = qpos
    mj.mj_forward(viewer.model, viewer.data)

    def dump_offsets_if_requested():
        if args.dump_rot_offsets_out is None:
            return
        if args.ik_config is None or targets is None:
            raise ValueError("--dump_rot_offsets_out requires --ik_config and --targets")

        with open(args.ik_config, "r", encoding="utf-8") as f:
            ik = json.load(f)
        table = ik.get("ik_match_table1", {})
        out = {}
        skipped = []
        for frame_name, entry in table.items():
            human_body_name = entry[0]
            if human_body_name not in targets:
                skipped.append((frame_name, human_body_name, "target_missing"))
                continue
            try:
                body_id = viewer.model.body(frame_name).id
            except Exception:
                skipped.append((frame_name, human_body_name, "robot_body_missing"))
                continue
            human_quat = targets[human_body_name][1]
            human_R = R.from_quat(human_quat, scalar_first=True)
            robot_R = R.from_matrix(viewer.data.xmat[body_id].reshape(3, 3))
            rot_offset = (human_R.inv() * robot_R).as_quat(scalar_first=True)
            out[frame_name] = [float(x) for x in rot_offset]

        dump_dir = os.path.dirname(args.dump_rot_offsets_out)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(args.dump_rot_offsets_out, "w", encoding="utf-8") as f:
            json.dump({"rot_offsets_wxyz": out, "skipped": skipped}, f, indent=2, ensure_ascii=False)
        print(f"Dumped rot_offsets to {args.dump_rot_offsets_out}")
        if skipped:
            print(f"[yellow]Skipped {len(skipped)} entries (missing target or robot body).[/yellow]")

    try:
        print("Interactive viewer: adjust joints in the UI.")
        print("Close the window or press Ctrl-C to exit.")
        if targets is not None:
            print("Overlaying target frames (viewer1) as colored axes.")
        if args.dump_rot_offsets_out is not None:
            print("Will dump rot_offsets on exit.")

        while True:
            mj.mj_forward(viewer.model, viewer.data)
            if targets is not None:
                viewer.viewer.user_scn.ngeom = 0
                for name, (pos, quat) in targets.items():
                    draw_frame(
                        pos,
                        R.from_quat(quat, scalar_first=True).as_matrix(),
                        viewer.viewer,
                        size=0.1,
                        joint_name=None,
                    )
            viewer.viewer.sync()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        dump_offsets_if_requested()
    finally:
        try:
            dump_offsets_if_requested()
        except Exception:
            # Avoid masking close errors; user can re-run with correct flags.
            pass
        viewer.close()


if __name__ == "__main__":
    main()
