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


def _force_w_positive(q: np.ndarray) -> np.ndarray:
	q = np.asarray(q, dtype=float)
	if float(q[0]) < 0.0:
		return -q
	return q


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
		"--solve_max_iter",
		type=int,
		default=20,
		help="Max IK iterations per BVH frame in calibration solve.",
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
		default=True,
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
		default=True,
		help="If set, flip quaternion sign so w>=0 (cosmetic; same rotation).",
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
	if args.calib_max_frames is not None:
		bvh_frames = bvh_frames[: max(1, int(args.calib_max_frames))]

	retargeter = GMR(
		src_human=src_human,
		tgt_robot=args.robot,
		actual_human_height=actual_human_height,
	)

	# Build calibration tasks: position-only (orientation_cost=0) so the solver doesn't
	# "eat" the offset via orientation constraints.
	calib_tasks: Dict[str, mink.FrameTask] = {}
	for frame_name in target_frame_names:
		calib_tasks[frame_name] = mink.FrameTask(
			frame_name=frame_name,
			frame_type="body",
			position_cost=float(args.pos_cost),
			orientation_cost=0.0,
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

			# Set targets (position-only). We still provide rotation in SE3 target but orientation_cost=0.
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
				human_R = R.from_quat(human_quat, scalar_first=True)
				robot_xmat = retargeter.configuration.data.xmat[body_id].reshape(3, 3)
				robot_R = R.from_matrix(robot_xmat)
				rot_offset = (human_R.inv() * robot_R).as_quat(scalar_first=True)
				per_frame_quats[frame_name].append(np.asarray(rot_offset, dtype=float))

		updated = 0
		out_offsets: Dict[str, np.ndarray] = {}
		for frame_name, qs in per_frame_quats.items():
			q_avg = _avg_quats_wxyz(qs)
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
		"calib_max_frames": int(args.calib_max_frames),
		"iters": int(args.iters),
		"only_frames": args.only_frames,
		"skipped_last_iter": skipped,
	}

	_write_json(args.out, out_ik)
	print(f"[auto-calib] wrote updated ik_config to: {args.out}")


if __name__ == "__main__":
	main()