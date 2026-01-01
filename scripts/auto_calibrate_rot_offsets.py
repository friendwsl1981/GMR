import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import mujoco as mj
import mink
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
	q = np.asarray(q, dtype=float)
	n = float(np.linalg.norm(q))
	if n == 0.0:
		return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
	return q / n


def _avg_quats_wxyz(quats: List[np.ndarray]) -> Optional[np.ndarray]:
	if not quats:
		return None
	ref = _normalize_quat_wxyz(quats[0])
	acc = np.zeros(4, dtype=float)
	for q in quats:
		qn = _normalize_quat_wxyz(q)
		# Keep a consistent hemisphere to avoid cancellation.
		if float(np.dot(ref, qn)) < 0.0:
			qn = -qn
		acc += qn
	return _normalize_quat_wxyz(acc)


def _quat_angle_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
	q1 = _normalize_quat_wxyz(q1_wxyz)
	q2 = _normalize_quat_wxyz(q2_wxyz)
	d = float(abs(np.dot(q1, q2)))
	d = float(np.clip(d, -1.0, 1.0))
	# 2*acos(|dot|)
	return float(np.degrees(2.0 * np.arccos(d)))


def _avg_quats_wxyz_robust(
	quats: List[np.ndarray],
	*,
	trim_frac: float = 0.0,
	max_angle_deg: Optional[float] = None,
) -> Optional[np.ndarray]:
	if not quats:
		return None
	trim_frac = float(np.clip(trim_frac, 0.0, 0.49))
	q0 = _avg_quats_wxyz(quats)
	if q0 is None:
		return None
	# Compute angular deviations from initial mean.
	angles = [(_quat_angle_deg(q, q0), i) for i, q in enumerate(quats)]
	angles.sort(key=lambda x: x[0])
	kept = angles
	if max_angle_deg is not None:
		max_angle_deg = float(max_angle_deg)
		kept = [ai for ai in kept if ai[0] <= max_angle_deg]
	if trim_frac > 0.0 and kept:
		k = int(round(len(kept) * (1.0 - trim_frac)))
		k = max(1, k)
		kept = kept[:k]
	return _avg_quats_wxyz([quats[i] for _a, i in kept])


def _force_w_positive(q: np.ndarray) -> np.ndarray:
	q = np.asarray(q, dtype=float)
	if float(q[0]) < 0.0:
		return -q
	return q


def _heading_yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> float:
	"""Return yaw (around +Z) in radians, using scipy's zyx convention."""
	r = R.from_quat(q_wxyz, scalar_first=True)
	# zyx -> [yaw, pitch, roll]
	yaw = float(r.as_euler("zyx", degrees=False)[0])
	return yaw


def _load_ik_config(path: str) -> Dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def _write_json(path: str, payload: Dict) -> None:
	out_dir = os.path.dirname(path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
	parser = argparse.ArgumentParser(
		description=(
			"Auto-calibrate per-frame rot_offset (wxyz) by iterating: "
			"(1) run IK with current offsets -> (2) compute rot_offset = inv(R_human)*R_robot -> "
			"(3) update offsets. Writes a new ik_config JSON."
		)
	)
	parser.add_argument("--bvh_file", required=True, type=str)
	parser.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
	parser.add_argument(
		"--robot",
		required=True,
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
	)
	parser.add_argument(
		"--ik_config",
		default=None,
		type=str,
		help=(
			"Path to the ik_config JSON to update. If omitted, uses the default from IK_CONFIG_DICT "
			"for the given --format/--robot."
		),
	)
	parser.add_argument(
		"--out",
		required=True,
		type=str,
		help="Where to write the updated ik_config JSON.",
	)
	parser.add_argument(
		"--calib_max_frames",
		type=int,
		default=1,
		help="How many BVH frames to use for calibration (starting from frame 0).",
	)
	parser.add_argument(
		"--frame_indices",
		nargs="*",
		type=int,
		default=None,
		help=(
			"Optional explicit BVH frame indices to use for calibration (overrides --calib_max_frames/--select_*). "
			"Example: --frame_indices 0 10 20"
		),
	)

	parser.add_argument(
		"--select_mode",
		choices=["first", "closest_to_first", "upright"],
		default="first",
		help=(
			"How to select frames from the first --calib_max_frames frames. "
			"'first' uses frames [0..N). 'closest_to_first' picks the K frames most similar to frame 0. "
			"'upright' picks the K frames with smallest hip/spine pitch+roll magnitude."
		),
	)
	parser.add_argument(
		"--select_k",
		type=int,
		default=None,
		help=(
			"If set, use only K frames from the selection pool (see --select_mode). "
			"Smaller K often matches the 'manual constant offsets' better."
		),
	)

	parser.add_argument(
		"--solve_max_iter",
		type=int,
		default=20,
		help="Max IK iterations per BVH frame in calibration solve.",
	)

	parser.add_argument(
		"--ori_cost",
		type=float,
		default=0.05,
		help=(
			"Orientation cost used for all calibration tasks (targets the raw human rotations, no offsets). "
			"Set to 0 to disable orientation constraints (not recommended if you want stable offsets)."
		),
	)

	parser.add_argument(
		"--pos_cost",
		type=float,
		default=1.0,
		help="Position cost used for all calibration tasks.",
	)

	parser.add_argument(
		"--reset_qpos_each_frame",
		action="store_true",
		default=False,
		help="Reset robot qpos to initial state before solving each calibration BVH frame.",
	)
	parser.add_argument(
		"--iters",
		type=int,
		default=3,
		help="How many offset-update iterations to run.",
	)
	parser.add_argument(
		"--w_positive",
		action="store_true",
		default=False,
		help="If set, flip quaternion sign so w>=0 (cosmetic; same rotation).",
	)

	parser.add_argument(
		"--remove_heading",
		action="store_true",
		default=True,
		help=(
			"(Legacy) Equivalent to --root_basis yaw. Kept for backward compatibility."
		),
	)

	parser.add_argument(
		"--root_basis",
		choices=["none", "yaw", "full"],
		default=None,
		help=(
			"Change-of-basis applied when computing rot_offset: "
			"'none' uses world rotations; 'yaw' removes human root yaw; 'full' removes full human root rotation. "
			"This can make offsets closer to constant manual tables."
		),
	)

	parser.add_argument(
		"--trim_frac",
		type=float,
		default=0.2,
		help="Outlier rejection: keep the closest (1-trim_frac) quats to the mean before averaging.",
	)
	parser.add_argument(
		"--max_angle_deg",
		type=float,
		default=45.0,
		help="Outlier rejection: drop per-frame quats with angle-to-mean larger than this (degrees).",
	)
	parser.add_argument(
		"--only_frames",
		nargs="*",
		default=None,
		help="Optional list of robot body frame_names to calibrate (others keep their original rot_offset).",
	)

	args = parser.parse_args()

	src_human = f"bvh_{args.format}"

	if args.ik_config is None:
		ik_path = str(IK_CONFIG_DICT[src_human][args.robot])
	else:
		ik_path = args.ik_config

	ik = _load_ik_config(ik_path)
	table1: Dict[str, List] = dict(ik.get("ik_match_table1", {}))
	if not table1:
		raise ValueError(f"No ik_match_table1 found in {ik_path}")

	# Restrict calibration to requested frames.
	target_frame_names = list(table1.keys())
	if args.only_frames:
		requested = set(args.only_frames)
		target_frame_names = [k for k in target_frame_names if k in requested]
		if not target_frame_names:
			raise ValueError("--only_frames did not match any ik_match_table1 keys")

	bvh_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
	if args.frame_indices:
		idx = [int(i) for i in args.frame_indices]
		idx = [i for i in idx if 0 <= i < len(bvh_frames)]
		if not idx:
			raise ValueError("--frame_indices are all out of range")
		bvh_frames = [bvh_frames[i] for i in idx]
	else:
		if args.calib_max_frames is not None:
			bvh_frames = bvh_frames[: max(1, int(args.calib_max_frames))]

	if args.frame_indices is None and args.select_k is not None:
		k = int(args.select_k)
		k = max(1, min(k, len(bvh_frames)))
		if args.select_mode == "first":
			bvh_frames = bvh_frames[:k]
		elif args.select_mode == "closest_to_first":
			# Pick frames closest to frame 0 by joint orientation similarity.
			ref = bvh_frames[0]
			# Prefer joints that exist in lafan1 and are informative.
			key_joints = [
				"Hips",
				"Spine2",
				"LeftUpLeg",
				"RightUpLeg",
				"LeftLeg",
				"RightLeg",
				"LeftArm",
				"RightArm",
			]
			ref_quats = {j: ref[j][1] for j in key_joints if j in ref}
			if not ref_quats:
				bvh_frames = bvh_frames[:k]
			else:
				scored = []
				for i, fr in enumerate(bvh_frames):
					angles = []
					for j, q_ref in ref_quats.items():
						if j not in fr:
							continue
						q = fr[j][1]
						angles.append(_quat_angle_deg(q, q_ref))
					score = float(np.mean(angles)) if angles else float("inf")
					scored.append((score, i))
				scored.sort(key=lambda x: x[0])
				keep_idx = sorted([i for _s, i in scored[:k]])
				bvh_frames = [bvh_frames[i] for i in keep_idx]
		else:
			# 'upright': small pitch/roll on hips+spine2.
			def pr_cost(frame):
				cost = 0.0
				for name in ("Hips", "Spine2"):
					if name not in frame:
						continue
					_qpos, q = frame[name]
					yaw, pitch, roll = R.from_quat(q, scalar_first=True).as_euler("zyx", degrees=False)
					cost += float(abs(pitch) + abs(roll))
				return cost

			scored = [(pr_cost(fr), i) for i, fr in enumerate(bvh_frames)]
			scored.sort(key=lambda x: x[0])
			keep_idx = sorted([i for _s, i in scored[:k]])
			bvh_frames = [bvh_frames[i] for i in keep_idx]

	retargeter = GMR(
		src_human=src_human,
		tgt_robot=args.robot,
		actual_human_height=actual_human_height,
	)

	# Build calibration tasks. We target the *raw human rotation* (no offsets) with a small
	# orientation cost so the robot orientations are not underdetermined.
	calib_tasks: Dict[str, mink.FrameTask] = {}
	for frame_name in target_frame_names:
		calib_tasks[frame_name] = mink.FrameTask(
			frame_name=frame_name,
			frame_type="body",
			position_cost=float(args.pos_cost),
			orientation_cost=float(args.ori_cost),
			lm_damping=1,
		)

	initial_qpos = retargeter.configuration.data.qpos.copy()
	skipped: List[Tuple[str, str, str]] = []

	# We keep the outer "iters" loop to allow re-solving from a stable starting point
	# multiple times (useful when the solver is sensitive to initial conditions).
	for it in range(int(args.iters)):
		per_frame_quats: Dict[str, List[np.ndarray]] = {k: [] for k in target_frame_names}
		skipped = []

		for human_data in bvh_frames:
			# Compute scaled human data (same preprocessing as retarget) but do NOT apply any rot_offset.
			hd = retargeter.to_numpy(dict(human_data))
			hd = retargeter.scale_human_data(hd, retargeter.human_root_name, retargeter.human_scale_table)
			hd = retargeter.apply_ground_offset(hd)
			retargeter.scaled_human_data = hd

			if args.reset_qpos_each_frame:
				retargeter.configuration.data.qpos[:] = initial_qpos
				mj.mj_forward(retargeter.model, retargeter.configuration.data)

			# Set targets. Orientation target is the raw human rotation (no offsets).
			# NOTE: We intentionally do NOT apply any heading removal to the IK targets.
			# Heading removal is applied only when computing rot_offset as a change-of-basis.
			for frame_name in target_frame_names:
				entry = table1.get(frame_name)
				if entry is None:
					continue
				human_body_name = entry[0]
				if human_body_name not in hd:
					skipped.append((frame_name, human_body_name, "human_missing"))
					continue
				target_pos, target_quat = hd[human_body_name]
				calib_tasks[frame_name].set_target(
					mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos)
				)

			# Solve IK with position-only tasks.
			dt = retargeter.configuration.model.opt.timestep
			tasks_list = list(calib_tasks.values())
			for _ in range(int(args.solve_max_iter)):
				vel = mink.solve_ik(
					retargeter.configuration,
					tasks_list,
					dt,
					retargeter.solver,
					retargeter.damping,
					retargeter.ik_limits,
				)
				retargeter.configuration.integrate_inplace(vel, dt)
				mj.mj_forward(retargeter.model, retargeter.configuration.data)

			# Compute rot_offset = inv(R_human) * R_robot.
			basis_R = R.identity()
			basis_mode = args.root_basis
			if basis_mode is None:
				basis_mode = "yaw" if args.remove_heading else "none"
			if basis_mode != "none":
				root_name = str(ik.get("human_root_name", "Hips"))
				if root_name in hd:
					_root_pos, root_quat = hd[root_name]
					root_R = R.from_quat(root_quat, scalar_first=True)
					if basis_mode == "yaw":
						yaw = _heading_yaw_from_quat_wxyz(root_quat)
						basis_R = R.from_euler("z", -yaw, degrees=False)
					elif basis_mode == "full":
						basis_R = root_R.inv()
			# Apply change-of-basis: R' = B * R
			for frame_name in target_frame_names:
				entry = table1.get(frame_name)
				if entry is None:
					continue
				human_body_name = entry[0]
				if human_body_name not in hd:
					continue
				try:
					body_id = retargeter.model.body(frame_name).id
				except Exception:
					skipped.append((frame_name, human_body_name, "robot_body_missing"))
					continue

				_pos, human_quat = hd[human_body_name]
				human_R = basis_R * R.from_quat(human_quat, scalar_first=True)
				robot_xmat = retargeter.configuration.data.xmat[body_id].reshape(3, 3)
				robot_R = basis_R * R.from_matrix(robot_xmat)
				rot_offset = (human_R.inv() * robot_R).as_quat(scalar_first=True)
				per_frame_quats[frame_name].append(np.asarray(rot_offset, dtype=float))

		updated = 0
		out_offsets: Dict[str, np.ndarray] = {}
		for frame_name, qs in per_frame_quats.items():
			q_avg = _avg_quats_wxyz_robust(
				qs,
				trim_frac=float(args.trim_frac),
				max_angle_deg=float(args.max_angle_deg) if args.max_angle_deg is not None else None,
			)
			if q_avg is None:
				continue
			if args.w_positive:
				q_avg = _force_w_positive(q_avg)
			out_offsets[frame_name] = q_avg
			updated += 1

		print(
			f"[auto-calib] pass {it+1}/{args.iters}: computed {updated}/{len(target_frame_names)} offsets; skipped={len(skipped)}"
		)

	# Use the last computed offsets.
	offset_guess: Dict[str, R] = {}
	for frame_name in target_frame_names:
		# fall back to original if missing
		entry = table1[frame_name]
		rot_wxyz = entry[4]
		q = out_offsets.get(frame_name)
		if q is None:
			q = np.asarray(rot_wxyz, dtype=float)
		offset_guess[frame_name] = R.from_quat(q, scalar_first=True)

	out_ik = dict(ik)
	out_table1 = dict(table1)
	for frame_name in target_frame_names:
		entry = list(out_table1[frame_name])
		q = offset_guess[frame_name].as_quat(scalar_first=True)
		entry[4] = [float(x) for x in q]
		out_table1[frame_name] = entry

	out_ik["ik_match_table1"] = out_table1
	out_ik["_auto_calib"] = {
		"source_ik_config": ik_path,
		"bvh_file": args.bvh_file,
		"format": args.format,
		"robot": args.robot,
		"calib_max_frames": int(args.calib_max_frames) if args.calib_max_frames is not None else None,
		"frame_indices": args.frame_indices,
		"select_mode": args.select_mode,
		"select_k": args.select_k,
		"iters": int(args.iters),
		"only_frames": args.only_frames,
		"pos_cost": float(args.pos_cost),
		"ori_cost": float(args.ori_cost),
		"root_basis": args.root_basis if args.root_basis is not None else ("yaw" if args.remove_heading else "none"),
		"trim_frac": float(args.trim_frac),
		"max_angle_deg": float(args.max_angle_deg) if args.max_angle_deg is not None else None,
		"skipped_last_iter": skipped,
	}

	_write_json(args.out, out_ik)
	print(f"[auto-calib] wrote updated ik_config to: {args.out}")


if __name__ == "__main__":
	main()