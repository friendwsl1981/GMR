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
        help="Show the MuJoCo LEFT UI panel so you can manually adjust joint states. (Right panel is hidden.)",
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

    parser.add_argument(
        "--dump_updated_ik_config_out",
        default=None,
        type=str,
        help=(
            "If set, write a full ik_config JSON on exit with updated rot_offset entries (same structure as --ik_config). "
            "Convenient for directly replacing the config."
        ),
    )

    parser.add_argument(
        "--dump_update_tables",
        choices=["table1", "table2", "both"],
        default="table1",
        help="Which tables to update in --dump_updated_ik_config_out: ik_match_table1, ik_match_table2, or both.",
    )

    parser.add_argument(
        "--dump_root_basis",
        choices=["none", "yaw", "full"],
        default="yaw",
        help=(
            "Change-of-basis applied when dumping rot_offsets, matching scripts/auto_calibrate_rot_offsets.py. "
            "'none' uses world rotations; 'yaw' removes human root yaw; 'full' removes full human root rotation."
        ),
    )

    parser.add_argument(
        "--dump_w_positive",
        action="store_true",
        default=False,
        help="If set, flip dumped quaternion sign so w>=0 (cosmetic; same rotation).",
    )

    parser.add_argument(
        "--dump_match_config_sign",
        action="store_true",
        default=True,
        help=(
            "If set (default), choose between q and -q so the dumped quaternion is closest to the ik_config's "
            "existing rot_offset entry (stable JSON diffs). Ignored if --dump_w_positive is set."
        ),
    )

    args = parser.parse_args()

    # Convenience: if user is already dumping rot_offsets but didn't provide an explicit
    # updated-config output path, write one next to outputs/ for easy replacement.
    if args.dump_rot_offsets_out is not None and args.dump_updated_ik_config_out is None and args.ik_config:
        base = os.path.basename(str(args.ik_config))
        stem = base[:-5] if base.lower().endswith(".json") else base
        args.dump_updated_ik_config_out = os.path.join("outputs", f"{stem}.manual.json")

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

    robot_frame_to_human_body = None
    ik_match_table1 = None
    if args.ik_config is not None and os.path.exists(args.ik_config):
        try:
            with open(args.ik_config, "r", encoding="utf-8") as f:
                _ik = json.load(f)
            table = _ik.get("ik_match_table1", {})
            if isinstance(table, dict) and table:
                robot_frame_to_human_body = {k: v[0] for k, v in table.items() if isinstance(v, list) and v}
                ik_match_table1 = table
        except Exception:
            robot_frame_to_human_body = None
            ik_match_table1 = None

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
    joint_frame_state = {"mode": "all"}  # 'all' or 'selected'
    ik_body_frame_state = {"mode": "off"}  # 'off' | 'selected' | 'all'

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

        if ch == 'f':
            joint_frame_state["mode"] = (
                "selected" if joint_frame_state["mode"] == "all" else "all"
            )
            print(f"[joint-frames] mode: {joint_frame_state['mode']} (toggle with 'f')")
            return

        if ch == 'g':
            # Cycle IK body-frame overlay: off -> selected -> all -> off
            if ik_body_frame_state["mode"] == "off":
                ik_body_frame_state["mode"] = "selected"
            elif ik_body_frame_state["mode"] == "selected":
                ik_body_frame_state["mode"] = "all"
            else:
                ik_body_frame_state["mode"] = "off"
            print(f"[ik-body-frames] mode: {ik_body_frame_state['mode']} (cycle with 'g')")
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
        show_right_ui=False,
        keyboard_callback=keyboard_callback,
    )

    if args.show_body_frames:
        # MuJoCo version differences: some builds don't expose mjFRAME_JOINT.
        # We draw joint frames ourselves via user_scn geoms.
        viewer.viewer.opt.frame = mj.mjtFrame.mjFRAME_NONE

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

    # If user is in "manual calibration" mode (dumping offsets), default to showing IK body frames
    # so what you align visually matches what gets dumped.
    if args.dump_rot_offsets_out is not None and ik_match_table1 is not None:
        ik_body_frame_state["mode"] = "all"

    def _heading_yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> float:
        r = R.from_quat(q_wxyz, scalar_first=True)
        # zyx -> [yaw, pitch, roll]
        return float(r.as_euler("zyx", degrees=False)[0])

    def _force_w_positive(q_wxyz: np.ndarray) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=float)
        if float(q_wxyz[0]) < 0.0:
            return -q_wxyz
        return q_wxyz

    def _normalize_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=float)
        n = float(np.linalg.norm(q_wxyz))
        if n <= 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q_wxyz / n

    dumped_once = {"done": False}

    def _write_json(path: str, payload) -> None:
        dump_dir = os.path.dirname(path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def dump_offsets_if_requested():
        if dumped_once["done"]:
            return
        if args.dump_rot_offsets_out is None and args.dump_updated_ik_config_out is None:
            return
        if args.ik_config is None or targets is None:
            raise ValueError("Dumping requires --ik_config and --targets")

        with open(args.ik_config, "r", encoding="utf-8") as f:
            ik = json.load(f)
        table1 = ik.get("ik_match_table1", {})
        table2 = ik.get("ik_match_table2", {})

        # We'll always compute offsets based on table1 mapping (frame_name -> human_body). If table2 exists
        # and you choose to update it, we update entries with the same keys.
        if not isinstance(table1, dict) or not table1:
            raise ValueError("ik_match_table1 missing or empty in --ik_config")

        out = {}
        skipped = []

        # Match auto-calib: apply a change-of-basis when computing rot_offset.
        basis_R = R.identity()
        basis_mode = str(args.dump_root_basis)
        if basis_mode != "none":
            root_name = str(ik.get("human_root_name", "Hips"))
            if root_name in targets:
                root_quat = np.asarray(targets[root_name][1], dtype=float)
                root_R = R.from_quat(root_quat, scalar_first=True)
                if basis_mode == "yaw":
                    yaw = _heading_yaw_from_quat_wxyz(root_quat)
                    basis_R = R.from_euler("z", -yaw, degrees=False)
                elif basis_mode == "full":
                    basis_R = root_R.inv()

        for frame_name, entry in table1.items():
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
            human_R = basis_R * R.from_quat(human_quat, scalar_first=True)
            robot_R = basis_R * R.from_matrix(viewer.data.xmat[body_id].reshape(3, 3))
            rot_offset = (human_R.inv() * robot_R).as_quat(scalar_first=True)
            rot_offset = _normalize_quat_wxyz(rot_offset)

            if args.dump_w_positive:
                rot_offset = _force_w_positive(rot_offset)
            elif args.dump_match_config_sign:
                # Stabilize sign to match the config entry style: pick hemisphere closest to existing rot_offset.
                try:
                    ref = np.asarray(entry[4], dtype=float)
                    ref = _normalize_quat_wxyz(ref)
                    if float(np.dot(ref, rot_offset)) < 0.0:
                        rot_offset = -rot_offset
                except Exception:
                    pass

            out[frame_name] = [float(x) for x in rot_offset]

        if args.dump_rot_offsets_out is not None:
            _write_json(args.dump_rot_offsets_out, {"rot_offsets_wxyz": out, "skipped": skipped})
            print(f"Dumped rot_offsets to {args.dump_rot_offsets_out}")
            if skipped:
                print(f"[yellow]Skipped {len(skipped)} entries (missing target or robot body).[/yellow]")

        if args.dump_updated_ik_config_out is not None:
            out_ik = dict(ik)
            update_tables = str(args.dump_update_tables)

            def _apply_updates_to_table(table_dict: dict) -> dict:
                if not isinstance(table_dict, dict):
                    return table_dict
                new_table = dict(table_dict)
                for frame_name, entry in new_table.items():
                    if not (isinstance(entry, list) and len(entry) >= 5):
                        continue
                    q = out.get(frame_name)
                    if q is None:
                        continue
                    new_entry = list(entry)
                    new_entry[4] = q
                    new_table[frame_name] = new_entry
                return new_table

            if update_tables in ("table1", "both"):
                out_ik["ik_match_table1"] = _apply_updates_to_table(table1)
            if update_tables in ("table2", "both"):
                out_ik["ik_match_table2"] = _apply_updates_to_table(table2)

            out_ik["_manual_calib"] = {
                "source_ik_config": args.ik_config,
                "qpos": args.qpos,
                "targets": args.targets,
                "dump_root_basis": str(args.dump_root_basis),
                "dump_w_positive": bool(args.dump_w_positive),
                "dump_match_config_sign": bool(args.dump_match_config_sign),
                "dump_update_tables": update_tables,
                "skipped": skipped,
            }

            _write_json(args.dump_updated_ik_config_out, out_ik)
            print(f"Wrote updated ik_config to {args.dump_updated_ik_config_out}")

        dumped_once["done"] = True

    try:
        if args.show_ui:
            print("Interactive viewer: LEFT UI panel is visible (right panel hidden).")
        else:
            print("Viewer UI is hidden.")
        if edit_state["joint_infos"]:
            print("Keyboard joint edit:")
            print("  - '[' / ']' : select previous/next joint")
            print("  - '+' / '-' : increase/decrease selected joint")
            print("  - '0' or 'r': reset selected joint to initial")
            print("  - 'p'       : print selected joint info")
            print("  - 'f'       : toggle robot joint frames: all <-> selected")
            if ik_match_table1 is not None:
                print("  - 'g'       : cycle IK body frames: off -> selected -> all")
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
        if args.dump_updated_ik_config_out is not None:
            print("Will dump updated ik_config on exit.")

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

            # Robot joint frame visibility mode:
            # - all: draw frames for all (non-ball) joints
            # - selected: draw only the selected joint frame
            viewer.viewer.user_scn.ngeom = 0

            sel_human_target = None
            sel_body_id = None
            sel_jid = None
            if edit_state["joint_infos"]:
                ji = edit_state["joint_infos"][edit_state["sel"]]
                try:
                    jid = int(ji["jid"])
                    sel_jid = jid
                    sel_body_id = int(viewer.model.jnt_bodyid[jid])
                    sel_body_name = viewer.model.body(sel_body_id).name
                    if robot_frame_to_human_body is not None:
                        sel_human_target = robot_frame_to_human_body.get(sel_body_name)
                        if sel_human_target is None:
                            # Fallback: sometimes the key matches the joint name.
                            sel_human_target = robot_frame_to_human_body.get(ji.get("name"))
                except Exception:
                    sel_body_id = None
                    sel_jid = None
                    sel_human_target = None

            def _draw_joint_frame(jid: int, size: float = 0.06) -> None:
                # Draw an approximate joint coordinate frame.
                # Origin at joint anchor; Z axis along joint axis.
                try:
                    jtype = int(viewer.model.jnt_type[jid])
                except Exception:
                    return
                if jtype == int(mj.mjtJoint.mjJNT_BALL):
                    return

                try:
                    body_id = int(viewer.model.jnt_bodyid[jid])
                    body_R = np.array(viewer.data.xmat[body_id], dtype=float).reshape(3, 3)
                    body_p = np.array(viewer.data.xpos[body_id], dtype=float)
                except Exception:
                    return

                if jtype == int(mj.mjtJoint.mjJNT_FREE):
                    # Free joint: just show the body frame.
                    draw_frame(body_p, body_R, viewer.viewer, size=size, joint_name=None)
                    return

                jpos_local = np.array(viewer.model.jnt_pos[jid], dtype=float)
                axis_local = np.array(viewer.model.jnt_axis[jid], dtype=float)
                pos = body_p + body_R @ jpos_local
                z = body_R @ axis_local
                nz = float(np.linalg.norm(z))
                if nz < 1e-9:
                    z = np.array([0.0, 0.0, 1.0])
                else:
                    z = z / nz
                x0 = np.array([1.0, 0.0, 0.0])
                if abs(float(np.dot(x0, z))) > 0.9:
                    x0 = np.array([0.0, 1.0, 0.0])
                x = np.cross(x0, z)
                nx = float(np.linalg.norm(x))
                if nx < 1e-9:
                    x = np.array([1.0, 0.0, 0.0])
                else:
                    x = x / nx
                y = np.cross(z, x)
                mat = np.stack([x, y, z], axis=1)
                draw_frame(pos, mat, viewer.viewer, size=size, joint_name=None)

            if args.show_body_frames:
                if joint_frame_state["mode"] == "selected":
                    # If user selected a pseudo root_* entry, sel_jid points to the free joint.
                    if sel_jid is not None:
                        _draw_joint_frame(int(sel_jid), size=0.10)
                else:
                    # Draw all joints. Keep within the user_scn geom budget.
                    # Each frame uses 3 geoms.
                    max_frames = max(1, int((viewer.viewer.user_scn.maxgeom - viewer.viewer.user_scn.ngeom) / 3))
                    count = 0
                    for jid in range(int(viewer.model.njnt)):
                        if count >= max_frames:
                            break
                        # Skip ball joints to avoid confusing axes.
                        if int(viewer.model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_BALL):
                            continue
                        _draw_joint_frame(jid, size=0.06)
                        count += 1

            # Draw IK body frames (these are the frames used for dumping rot_offsets).
            if ik_match_table1 is not None and ik_body_frame_state["mode"] != "off":
                if ik_body_frame_state["mode"] == "selected":
                    # Only draw the body frame corresponding to the currently selected joint's body (if present).
                    if sel_body_id is not None:
                        try:
                            sel_body_name = viewer.model.body(int(sel_body_id)).name
                        except Exception:
                            sel_body_name = None
                        if sel_body_name is not None and sel_body_name in ik_match_table1:
                            try:
                                bid = viewer.model.body(sel_body_name).id
                                pos = np.array(viewer.data.xpos[bid], dtype=float)
                                mat = np.array(viewer.data.xmat[bid], dtype=float).reshape(3, 3)
                                draw_frame(pos, mat, viewer.viewer, size=0.12, joint_name=None)
                            except Exception:
                                pass
                else:
                    # Draw all IK frames, respecting geom budget.
                    keys = list(ik_match_table1.keys())
                    max_frames = max(1, int((viewer.viewer.user_scn.maxgeom - viewer.viewer.user_scn.ngeom) / 3))
                    count = 0
                    for frame_name in keys:
                        if count >= max_frames:
                            break
                        try:
                            bid = viewer.model.body(frame_name).id
                        except Exception:
                            continue
                        pos = np.array(viewer.data.xpos[bid], dtype=float)
                        mat = np.array(viewer.data.xmat[bid], dtype=float).reshape(3, 3)
                        draw_frame(pos, mat, viewer.viewer, size=0.06, joint_name=None)
                        count += 1

            # Draw target frames (if any). This is independent of the robot joint-frame toggle.
            if targets is not None:
                for _name, (pos, quat) in targets.items():
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
