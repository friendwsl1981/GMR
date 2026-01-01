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
        "--show_ui",
        action="store_true",
        default=False,
        help="Show MuJoCo left/right UI panels so you can manually adjust joint states.",
    )

    parser.add_argument(
        "--step_sim",
        action="store_true",
        default=False,
        help=(
            "Step the MuJoCo simulation (mj_step) each frame instead of only calling mj_forward. "
            "Useful if you tweak actuator controls in the UI; qpos will then update. "
            "Gravity is disabled in this mode to avoid the robot falling while you edit."
        ),
    )

    parser.add_argument(
        "--joint_step",
        type=float,
        default=0.02,
        help="Keyboard joint edit step in radians (hinge) or meters (slide).",
    )

    parser.add_argument(
        "--root_step_trans",
        type=float,
        default=0.01,
        help="Keyboard root translation step in meters (for the free joint).",
    )

    parser.add_argument(
        "--root_step_rot_deg",
        type=float,
        default=5.0,
        help="Keyboard root rotation step in degrees (for the free joint).",
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

    viewer = None
    initial_qpos = None

    # Build a list of joints that are likely meaningful to edit (prefer those driven by actuators).
    def build_editable_joint_list(model: mj.MjModel):
        driven_joint_ids = []
        if model.nu > 0:
            try:
                driven_joint_ids = sorted(set(int(jid) for jid in model.actuator_trnid[:, 0]))
            except Exception:
                driven_joint_ids = []

        joint_infos = []
        candidate_joint_ids = driven_joint_ids if driven_joint_ids else list(range(model.njnt))
        for jid in candidate_joint_ids:
            jtype = int(model.jnt_type[jid])
            # Skip free joint here; we add special pseudo-joints for root editing below.
            if jtype == int(mj.mjtJoint.mjJNT_FREE):
                continue
            if jtype == int(mj.mjtJoint.mjJNT_BALL):
                continue

            qadr = int(model.jnt_qposadr[jid])
            name = model.joint(jid).name

            lo, hi = None, None
            try:
                if int(model.jnt_limited[jid]) == 1:
                    lo = float(model.jnt_range[jid, 0])
                    hi = float(model.jnt_range[jid, 1])
            except Exception:
                lo, hi = None, None

            joint_infos.append({"jid": jid, "name": name, "qadr": qadr, "lo": lo, "hi": hi, "jtype": jtype})

        # Deduplicate by qpos address (some actuators may target same joint).
        seen = set()
        out = []
        for ji in joint_infos:
            if ji["qadr"] in seen:
                continue
            seen.add(ji["qadr"])
            out.append(ji)
        # Add pseudo-joints for root (free joint) editing if present.
        # MuJoCo free joint qpos layout: [tx, ty, tz, qw, qx, qy, qz].
        # We expose translation components and small incremental rotations.
        try:
            root_jid = None
            for jid in range(model.njnt):
                if int(model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE):
                    root_jid = jid
                    break
            if root_jid is not None:
                qadr = int(model.jnt_qposadr[root_jid])
                root_entries = [
                    {"jid": root_jid, "name": "__root_tx", "qadr": qadr + 0, "lo": None, "hi": None, "jtype": "root_tx"},
                    {"jid": root_jid, "name": "__root_ty", "qadr": qadr + 1, "lo": None, "hi": None, "jtype": "root_ty"},
                    {"jid": root_jid, "name": "__root_tz", "qadr": qadr + 2, "lo": None, "hi": None, "jtype": "root_tz"},
                    {"jid": root_jid, "name": "__root_roll", "qadr": qadr + 3, "lo": None, "hi": None, "jtype": "root_roll"},
                    {"jid": root_jid, "name": "__root_pitch", "qadr": qadr + 3, "lo": None, "hi": None, "jtype": "root_pitch"},
                    {"jid": root_jid, "name": "__root_yaw", "qadr": qadr + 3, "lo": None, "hi": None, "jtype": "root_yaw"},
                ]
                out = root_entries + out
        except Exception:
            pass

        return out

    edit_state = {"enabled": True, "sel": 0, "joint_infos": [], "pending": None}

    def _apply_delta(ji, direction: float):
        """Apply an edit to qpos.

        IMPORTANT: Must only be called from the main loop thread.
        """
        jtype = ji.get("jtype")
        if jtype in ("root_tx", "root_ty", "root_tz"):
            viewer.data.qpos[ji["qadr"]] = float(
                viewer.data.qpos[ji["qadr"]] + direction * args.root_step_trans
            )
            return
        if jtype in ("root_roll", "root_pitch", "root_yaw"):
            # Root quaternion starts at qadr+3.
            qadr = int(ji["qadr"])
            qwxyz = np.array(viewer.data.qpos[qadr : qadr + 4], dtype=float)
            curr = R.from_quat(qwxyz, scalar_first=True)
            step_rad = float(np.deg2rad(args.root_step_rot_deg) * direction)
            axis = {"root_roll": "x", "root_pitch": "y", "root_yaw": "z"}[jtype]
            delta = R.from_euler(axis, step_rad)
            new_q = (curr * delta).as_quat(scalar_first=True)
            viewer.data.qpos[qadr : qadr + 4] = new_q
            return
        # Default: hinge/slide joint
        viewer.data.qpos[ji["qadr"]] = float(viewer.data.qpos[ji["qadr"]] + direction * args.joint_step)

    def keyboard_callback(keycode: int):
        # Called from viewer thread.
        if not edit_state["enabled"] or viewer is None:
            return

        try:
            ch = chr(keycode)
        except Exception:
            return

        jis = edit_state["joint_infos"]
        if not jis:
            return

        if ch == '[':
            edit_state["sel"] = (edit_state["sel"] - 1) % len(jis)
            ji = jis[edit_state["sel"]]
            print(f"[joint-edit] selected {edit_state['sel']}/{len(jis)-1}: {ji['name']}")
            return
        if ch == ']':
            edit_state["sel"] = (edit_state["sel"] + 1) % len(jis)
            ji = jis[edit_state["sel"]]
            print(f"[joint-edit] selected {edit_state['sel']}/{len(jis)-1}: {ji['name']}")
            return

        # Queue edits; apply in the main loop to avoid MuJoCo thread-safety issues.
        if ch in ('=', '+'):
            edit_state["pending"] = ("delta", edit_state["sel"], +1.0)
            return
        if ch == '-':
            edit_state["pending"] = ("delta", edit_state["sel"], -1.0)
            return
        if ch in ('0', 'r') and initial_qpos is not None:
            edit_state["pending"] = ("reset", edit_state["sel"], 0.0)
            return
        if ch == 'p':
            edit_state["pending"] = ("print", edit_state["sel"], 0.0)
            return

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=30,
        transparent_robot=0,
        show_left_ui=args.show_ui,
        show_right_ui=args.show_ui,
        keyboard_callback=keyboard_callback,
    )

    if args.show_body_frames:
        viewer.viewer.opt.frame = mj.mjtFrame.mjFRAME_BODY

    # Initialize pose once, then DO NOT overwrite qpos each frame.
    # This keeps the viewer interactive so you can adjust joints in the UI.
    nq = int(viewer.model.nq)
    qpos_len = int(len(qpos))
    if qpos_len != nq:
        print(
            f"[yellow]Warning: qpos length mismatch: file has {qpos_len}, model expects {nq}. "
            "Will truncate or pad with zeros. For best results, regenerate qpos with the same --robot.[/yellow]"
        )

    n = min(qpos_len, nq)
    viewer.data.qpos[:n] = qpos[:n]
    if qpos_len < nq:
        viewer.data.qpos[n:nq] = 0.0

    mj.mj_forward(viewer.model, viewer.data)
    initial_qpos = viewer.data.qpos.copy()

    edit_state["joint_infos"] = build_editable_joint_list(viewer.model)

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
        if args.show_ui:
            print("Interactive viewer: UI panels are visible.")
        else:
            print("Viewer UI is hidden.")
        if edit_state["joint_infos"]:
            print("Keyboard joint edit:")
            print("  - '[' / ']' : select previous/next joint")
            print("  - '+' / '-' : increase/decrease selected joint")
            print("  - '0' or 'r': reset selected joint to initial")
            print("  - 'p'       : print selected joint info")
            print(f"  - step size : {args.joint_step} (set via --joint_step)")
            print(f"  - root trans step : {args.root_step_trans} (set via --root_step_trans)")
            print(f"  - root rot step(deg): {args.root_step_rot_deg} (set via --root_step_rot_deg)")
        else:
            print("Keyboard joint edit: no editable joints found.")

        if args.step_sim:
            try:
                viewer.model.opt.gravity[:] = 0
            except Exception:
                pass
            print("Simulation stepping is ON (--step_sim). Gravity disabled.")
        else:
            print("Simulation stepping is OFF (forward-kinematics only).")

        print("Close the window or press Ctrl-C to exit.")
        if targets is not None:
            print("Overlaying target frames (viewer1) as colored axes.")
        if args.dump_rot_offsets_out is not None:
            print("Will dump rot_offsets on exit.")

        while True:
            # Apply queued edits (main thread only).
            pending = edit_state.get("pending")
            if pending is not None and edit_state["joint_infos"]:
                try:
                    action, sel, direction = pending
                    jis = edit_state["joint_infos"]
                    if 0 <= int(sel) < len(jis):
                        ji = jis[int(sel)]
                        jtype = ji.get("jtype")
                        if action == "delta":
                            _apply_delta(ji, float(direction))
                            if jtype not in ("root_roll", "root_pitch", "root_yaw"):
                                if ji["lo"] is not None and ji["hi"] is not None:
                                    viewer.data.qpos[ji["qadr"]] = float(
                                        np.clip(viewer.data.qpos[ji["qadr"]], ji["lo"], ji["hi"])
                                    )
                        elif action == "reset" and initial_qpos is not None:
                            if jtype in ("root_tx", "root_ty", "root_tz"):
                                viewer.data.qpos[ji["qadr"]] = float(initial_qpos[ji["qadr"]])
                            elif jtype in ("root_roll", "root_pitch", "root_yaw"):
                                qadr = int(ji["qadr"])
                                viewer.data.qpos[qadr : qadr + 4] = initial_qpos[qadr : qadr + 4]
                            else:
                                viewer.data.qpos[ji["qadr"]] = float(initial_qpos[ji["qadr"]])
                        elif action == "print":
                            if jtype in ("root_roll", "root_pitch", "root_yaw"):
                                qadr = int(ji["qadr"])
                                qwxyz = [float(x) for x in viewer.data.qpos[qadr : qadr + 4]]
                                print(
                                    f"[joint-edit] sel={sel}/{len(jis)-1} name={ji['name']} root_quat_wxyz={qwxyz}"
                                )
                            else:
                                val = float(viewer.data.qpos[ji["qadr"]])
                                print(
                                    f"[joint-edit] sel={sel}/{len(jis)-1} name={ji['name']} qpos={val:.4f}"
                                )
                finally:
                    edit_state["pending"] = None

            if args.step_sim:
                mj.mj_step(viewer.model, viewer.data)
            else:
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
