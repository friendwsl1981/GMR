import argparse
import pickle
import os

import numpy as np


def _get_dof_names_from_mujoco(robot: str, dof_count: int) -> list[str] | None:
    """Return DoF names in the same order as qpos[7:] for the given robot.

    Uses MuJoCo model joint order (sorted by qpos address). Returns None if
    mujoco/model loading isn't available.
    """
    try:
        import mujoco as mj
        from general_motion_retargeting.params import ROBOT_XML_DICT
    except Exception:
        return None

    xml_path = ROBOT_XML_DICT.get(robot)
    if xml_path is None:
        return None

    try:
        model = mj.MjModel.from_xml_path(str(xml_path))
    except Exception:
        return None

    joints = []
    for j in range(model.njnt):
        qposadr = int(model.jnt_qposadr[j])
        if qposadr < 7:
            continue
        name = model.jnt(j).name
        joints.append((qposadr, name))

    joints.sort(key=lambda x: x[0])
    dof_names = [name for _, name in joints]
    if len(dof_names) != dof_count:
        return None
    return dof_names


def _make_header(robot: str | None, dof_count: int) -> list[str]:
    header = [
        "root_x",
        "root_y",
        "root_z",
        "root_qx",
        "root_qy",
        "root_qz",
        "root_qw",
    ]

    dof_names = None
    if robot:
        dof_names = _get_dof_names_from_mujoco(robot, dof_count)

    if dof_names is None:
        dof_names = [f"dof_{i}" for i in range(dof_count)]

    return header + dof_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot name (e.g., figurobot_2nd). Used to generate joint-name headers when --with_header is set.",
    )
    parser.add_argument(
        "--with_header",
        action="store_true",
        help="Write a first-line header: root pose + joint names (requires --robot for named joints).",
    )
    args = parser.parse_args()

    out_folder = os.path.join(args.folder, "csv")
    os.makedirs(out_folder, exist_ok=True)

    for i, file in enumerate(os.listdir(args.folder)):
        if file.endswith(".pkl"):
            with open(os.path.join(args.folder, file), "rb") as f:
                motion_data = pickle.load(f)
        else:
            continue

        dof_pos = motion_data["dof_pos"]
        frame_rate = motion_data["fps"]            
        motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
        motion[:, :3] = motion_data["root_pos"]
        motion[:, 3:7] = motion_data["root_rot"]
        motion[:, 7:] = dof_pos
        
        if frame_rate > 30:
            # downsample to 30 fps
            downsample_factor = frame_rate / 30.0
            indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
            old_length = motion.shape[0]
            motion = motion[indices]
            print(f"Downsampled from {old_length} to {motion.shape[0]} frames")
        
        header = None
        if args.with_header:
            header = ",".join(_make_header(args.robot, dof_pos.shape[1]))

        np.savetxt(
            os.path.join(args.folder, "csv", file.replace(".pkl", ".csv")),
            motion,
            delimiter=",",
            header=header or "",
            comments="" if header else "# ",
        )
        print(f"({i}/{len(os.listdir(args.folder))}) Saved to {os.path.join(args.folder, 'csv', file.replace('.pkl', '.csv'))}")
