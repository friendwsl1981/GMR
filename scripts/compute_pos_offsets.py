# scripts/compute_pos_offsets.py
import json
from pathlib import Path
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
import mujoco as mj

# CONFIG
BVH_FILE = "/home/wenshulong/下载/lafan1/dance1_subject1.bvh"
FORMAT = "lafan1"
ROBOT = "figurobot_2nd"
N_FRAMES = 100
JOINTS = {
    "chest_pitch_link": "Spine2",
    "neck_yaw_link": "Head",
    "left_hand_roll_link": "LeftHand",
    "right_hand_roll_link": "RightHand",
}

def world_pos_of_body(model, data, body_name):
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    return np.array(mj.mj_jnt2axis(model, data, bid)) if False else data.xpos[bid].copy()

def main():
    frames, _ = load_bvh_file(BVH_FILE, format=FORMAT)
    ret = GMR(src_human=f"bvh_{FORMAT}", tgt_robot=ROBOT, verbose=False)
    model = ret.model
    # prepare arrays
    sums = {k: np.zeros(3) for k in JOINTS.keys()}
    counts = {k: 0 for k in JOINTS.keys()}

    for i, frame in enumerate(frames[:N_FRAMES]):
        # update targets and configuration for this frame
        ret.update_targets(frame, offset_to_ground=False)
        # After update_targets, ret.scaled_human_data contains human positions (world)
        for body_name, human_bone in JOINTS.items():
            if human_bone not in ret.scaled_human_data:
                continue
            human_pos, human_quat = ret.scaled_human_data[human_bone]
            # get robot body world pos from model/data (requires a configuration/data)
            # ret.configuration.data.xpos holds body positions (MuJoCo frame) if available
            try:
                bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
                robot_pos = ret.configuration.data.xpos[bid].copy()
            except Exception:
                # fallback: if body not found, skip
                continue
            delta = human_pos - robot_pos  # world-frame vector: human - robot
            # convert delta into human_local: apply inverse of human rotation
            # human_quat is scalar-first (w,x,y,z)
            from scipy.spatial.transform import Rotation as R
            human_R = R.from_quat(human_quat, scalar_first=True)
            local_delta = human_R.inv().apply(delta)
            sums[body_name] += local_delta
            counts[body_name] += 1

    suggestions = {}
    for body_name in JOINTS.keys():
        if counts[body_name] == 0:
            continue
        mean_local = sums[body_name] / counts[body_name]
        suggestions[body_name] = [float(round(x,5)) for x in mean_local.tolist()]

    print("# Suggested pos_offsets (human-local coordinates):")
    print(json.dumps(suggestions, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()