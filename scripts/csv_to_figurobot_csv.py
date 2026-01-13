# 将csv转成figurobot专用的csv格式
# csv表头是 root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw,left_hip_pitch,left_thigh_roll,left_knee_yaw,left_shank_pitch,left_ankle_pitch,left_foot_roll,right_hip_pitch,right_thigh_roll,right_knee_yaw,right_shank_pitch,right_ankle_pitch,right_foot_roll,lumbar2_yaw,lumbar1_roll,chest_pitch,neck_yaw,left_shoulder_pitch,left_upper_arm_roll,left_elbow_yaw,left_forearm_roll,left_hand_roll,right_shoulder_pitch,right_upper_arm_roll,right_elbow_yaw,right_forearm_roll,right_hand_roll
# figurobot csv 去掉 root_x,root_y,root_z,root_qx,root_qy,root_qz,root_qw 这7列
# 然后剩余的列的表头 需要映射成 figurobot 的关节ID
# 映射关系见 assets/figurobot_2nd/motors_mapping.yml
# 例如：
# motor_hardware:
#   # Left Leg (6 motors)
#   - name: left_hip_pitch
#     joint_index: 0
#     hardware_id: 19
#     online: true
    
#   - name: left_thigh_roll
#     joint_index: 1
#     hardware_id: 21
#     online: true
# 其中 name 对应 csv 表头， hardware_id 就是 figurobot csv 里的关节ID
# 然后csv文件里面的弧度值转为角度值

import csv
import math
import os
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert CSV to figurobot format')
    parser.add_argument('--input_csv', required=True, help='Input CSV file path')
    parser.add_argument('--output_csv', required=True, help='Output CSV file path')
    args = parser.parse_args()

    # Path to motors_mapping.yml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, '..', 'assets', 'figurobot_2nd', 'motors_mapping.yml')

    # Load mapping
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    name_to_id = {motor['name']: motor['hardware_id'] for motor in data['motor_hardware']}

    # Expected header
    expected_header = ['root_x','root_y','root_z','root_qx','root_qy','root_qz','root_qw','left_hip_pitch','left_thigh_roll','left_knee_yaw','left_shank_pitch','left_ankle_pitch','left_foot_roll','right_hip_pitch','right_thigh_roll','right_knee_yaw','right_shank_pitch','right_ankle_pitch','right_foot_roll','lumbar2_yaw','lumbar1_roll','chest_pitch','neck_yaw','left_shoulder_pitch','left_upper_arm_roll','left_elbow_yaw','left_forearm_roll','left_hand_roll','right_shoulder_pitch','right_upper_arm_roll','right_elbow_yaw','right_forearm_roll','right_hand_roll']

    # Read input CSV
    with open(args.input_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != expected_header:
            raise ValueError(f"Header mismatch. Expected: {expected_header}, Got: {header}")
        
        # New header: skip first 7, map to hardware_id
        new_header = [str(name_to_id[col]) for col in header[7:]]
        
        # Read and convert rows
        rows = []
        for row in reader:
            # Convert radians to degrees for joint columns
            new_row = [float(x) * 180 / math.pi for x in row[7:]]
            rows.append(new_row)

    # Write output CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(rows)

if __name__ == '__main__':
    main()
