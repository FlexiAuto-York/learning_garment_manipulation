import os
import time
import argparse
import threading
import math
import numpy as np
import cv2
import yaml
import pyrealsense2 as rs

from real_robot.robot.xarm_lite6 import XArmLite6

# --- Defaults ---
DEFAULT_LEFT_IP = os.environ.get('XARM_LEFT_IP', '192.168.1.155')
DEFAULT_RIGHT_IP = os.environ.get('XARM_RIGHT_IP', '192.168.1.170')
DEFAULT_SQUARES_X = 3
DEFAULT_SQUARES_Y = 2
DEFAULT_SQUARE_SIZE = 0.035
DEFAULT_MARKER_SIZE = 0.026

# --- Math Utilities for Hand-Eye Calibration ---
def pose_list_to_matrix(p):
    p = np.asarray(p, dtype=float).reshape(6)
    t = p[:3]
    rvec = p[3:]
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invert_homogeneous_matrix(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def matrix_to_pose_lists(T):
    return T[:3, :3], T[:3, 3]

# --- Hand-Eye Solver ---
def run_hand_eye(samples, output_file):
    if len(samples) < 3:
        print(f"Not enough samples for {output_file}. Need at least 3 valid detections.")
        return

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    print(f"\nProcessing {len(samples)} samples for {output_file}...")

    for s in samples:
        T_base2gripper = pose_list_to_matrix(s['robot_pose'])
        T_gripper2base = invert_homogeneous_matrix(T_base2gripper)
        R_g, t_g = matrix_to_pose_lists(T_gripper2base)
        
        R_gripper2base_list.append(R_g)
        t_gripper2base_list.append(t_g)

        rvec = np.asarray(s['rvec']).reshape(3, 1)
        tvec = np.asarray(s['tvec']).reshape(3, 1)
        R_tc, _ = cv2.Rodrigues(rvec)
        R_target2cam_list.append(R_tc)
        t_target2cam_list.append(tvec.reshape(3,))

    print("Solving AX=XB (Eye-to-Hand) using Park & Martin...")
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_PARK
    )

    X = np.eye(4)
    X[:3, :3] = R_cam2base
    X[:3, 3] = t_cam2base.reshape(3,)

    if X[2, 3] < 0.5:
        print("Warning: Result seems inverted. Inverting matrix...")
        X = invert_homogeneous_matrix(X)

    out = {
        'camera_to_base': {'matrix': X.tolist()},
        'meta': {'samples': len(samples), 'board_type': 'charuco'}
    }

    with open(output_file, 'w') as f:
        yaml.dump(out, f)

    np.set_printoptions(precision=6, suppress=True)
    print(f"\nCalibration Result (T_base^camera) for {output_file}:\n")
    print(X)
    print(f"\nWrote calibration to {output_file}")

# --- Custom YAML Saver ---
def save_poses_yaml(filename, ip, gripper, sq_x, sq_y, sq_size, mk_size, poses_rad):
    poses_deg = [[round(math.degrees(rad), 4) for rad in pose] for pose in poses_rad]
    
    with open(filename, 'w') as f:
        f.write(f'robot:\n  ip: "{ip}"\n  gripper: "{gripper}"\n\n')
        f.write(f'# ChArUco Board Settings\n')
        f.write(f'board:\n')
        f.write(f'  squares_x: {sq_x}\n')
        f.write(f'  squares_y: {sq_y}\n')
        f.write(f'  square_length: {sq_size}\n')
        f.write(f'  marker_length: {mk_size}\n\n\n')
        f.write(f'# =============================================================================\n')
        f.write(f'# CALIBRATION POSES\n')
        f.write(f'# =============================================================================\n')
        f.write(f'# Format: [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3] (Degrees)\n')
        f.write(f'# Total Poses: {len(poses_deg)}\n')
        f.write(f'poses:\n')
        for p in poses_deg:
            f.write(f'  - {p}\n')

# --- Interactive Terminal Thread ---
def input_thread(state_dict):
    while state_dict['running']:
        cmd = input("\n[Terminal] Enter 'R' (Record Pose), 'F' (Finish Recording), or 'Q' (Quit): ").strip().upper()
        if cmd in ['R', 'F', 'Q']:
            state_dict['cmd'] = cmd
            if cmd in ['F', 'Q']:
                state_dict['running'] = False
                break

# --- Execution ---
def main():
    parser = argparse.ArgumentParser(description='Single-arm xArm Hand-to-Eye calibration using RealSense + ChArUco')
    parser.add_argument('--mode', type=str, choices=['auto', 'manual'], default='auto', help='Run mode: auto uses saved poses if available, manual forces recording.')
    parser.add_argument('--arm', type=str, choices=['left', 'right'], default='left', help='Which arm to calibrate (left or right)')
    args = parser.parse_args()

    calib_dir = "calibration"
    os.makedirs(calib_dir, exist_ok=True)
    
    pose_file = os.path.join(calib_dir, f"xarm-{args.arm}-poses.yaml")
    calib_file = os.path.join(calib_dir, f"xarm-{args.arm}-calib.yaml")

    poses_exist = os.path.exists(pose_file)
    recorded_poses = []
    
    # 1. Load configuration (from YAML if it exists, otherwise use defaults)
    if poses_exist:
        print(f"Loading configuration from {pose_file}...")
        with open(pose_file, 'r') as f:
            data = yaml.safe_load(f)
            
        target_ip = data.get('robot', {}).get('ip', DEFAULT_LEFT_IP if args.arm == 'left' else DEFAULT_RIGHT_IP)
        sq_x = data.get('board', {}).get('squares_x', DEFAULT_SQUARES_X)
        sq_y = data.get('board', {}).get('squares_y', DEFAULT_SQUARES_Y)
        sq_size = data.get('board', {}).get('square_length', DEFAULT_SQUARE_SIZE)
        mk_size = data.get('board', {}).get('marker_length', DEFAULT_MARKER_SIZE)
        
        if args.mode == 'auto':
            poses_deg = data.get('poses', [])
            recorded_poses = [[math.radians(deg) for deg in p] for p in poses_deg]
    else:
        target_ip = DEFAULT_LEFT_IP if args.arm == 'left' else DEFAULT_RIGHT_IP
        sq_x = DEFAULT_SQUARES_X
        sq_y = DEFAULT_SQUARES_Y
        sq_size = DEFAULT_SQUARE_SIZE
        mk_size = DEFAULT_MARKER_SIZE

    # 2. Start RealSense Pipeline
    print("Starting RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # 3. Configure OpenCV ChArUco Detectors (Using UR Reference Logic)[cite: 2]
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        board = cv2.aruco.CharucoBoard((sq_x, sq_y), sq_size, mk_size, aruco_dict)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    except AttributeError:
        print("⚠️ Warning: Using legacy OpenCV 4.x API.[cite: 2]")
        board = cv2.aruco.CharucoBoard_create(sq_x, sq_y, sq_size, mk_size, aruco_dict)
        detector = None 

    # 4. Connect to Arm
    print(f"Connecting to {args.arm} arm at {target_ip}...")
    robot = XArmLite6(target_ip, gripper='lite6', side=args.arm, walls=False)

    robot.arm.clean_error()
    robot.arm.clean_warn()
    robot.arm.motion_enable(enable=True)
    robot.arm.set_mode(0)
    robot.arm.set_state(0)

    # 5. Manual Recording Mode Logic
    if args.mode == 'manual' or not recorded_poses:
        if args.mode == 'manual':
            print(f"\nManual mode explicitly selected. Entering manual recording mode.")
        else:
            print(f"\nNo poses loaded. Falling back to manual recording mode.")
            
        print(f"\nOpening {args.arm} gripper...")
        robot.open_gripper()
        time.sleep(1.0) 

        input(f"\n>>> Place the ChArUco board into the {args.arm} gripper and press ENTER to close it: ")

        print(f"Closing {args.arm} gripper...")
        robot.close_gripper()
        time.sleep(1.0)

        input(f"\n>>> FIRMLY HOLD THE {args.arm.upper()} ARM NOW to support its weight, then press ENTER to enable Free-Drive: ")
        
        robot.arm.clean_error()
        robot.arm.set_mode(2)
        robot.arm.set_state(0)

        state = {'running': True, 'cmd': None}
        ui_thread = threading.Thread(target=input_thread, args=(state,), daemon=True)
        ui_thread.start()

        try:
            while state['running']:
                if robot.arm.error_code != 0:
                    err = robot.arm.error_code
                    if err == 37:
                        print("\n⚠️ Error 37 detected! The arm locked itself. Clearing error and re-enabling free-drive...")
                        robot.arm.clean_error()
                        time.sleep(0.1)
                        robot.arm.set_mode(2)
                        robot.arm.set_state(0)
                    else:
                        print(f"\n⚠️ Unexpected Error {err} detected. Clearing...")
                        robot.arm.clean_error()

                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    continue

                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                if color_frame:
                    color_img = np.asanyarray(color_frame.get_data())
                    cv2.imshow('Camera View (Terminal is Active)', color_img)
                    cv2.waitKey(1)
                
                if state['cmd'] == 'R':
                    code, q = robot.arm.get_servo_angle(is_radian=True)
                    if code == 0:
                        recorded_poses.append(np.array(q).tolist())
                        print(f"✅ Pose #{len(recorded_poses)} recorded.")
                    else:
                        print(f"❌ Failed to read joints (code {code}).")
                    state['cmd'] = None

        finally:
            print("\nFree-drive off, position mode restored.")
            robot.arm.set_mode(0)
            robot.arm.set_state(0)
            time.sleep(0.5)
            
        if state['cmd'] == 'Q':
            print("Quitting without sampling.")
            pipeline.stop()
            cv2.destroyAllWindows()
            robot.open_gripper()
            robot.disconnect()
            return

        if len(recorded_poses) > 0:
            ans = input(f"\nDo you want to save these {len(recorded_poses)} joint poses to {pose_file}? (y/n): ").strip().lower()
            if ans == 'y':
                print(f"Saving configuration and {len(recorded_poses)} poses to {pose_file}...")
                save_poses_yaml(pose_file, target_ip, 'lite6', sq_x, sq_y, sq_size, mk_size, recorded_poses)
        else:
            print("No poses recorded. Exiting.")
            pipeline.stop()
            cv2.destroyAllWindows()
            robot.open_gripper()
            robot.disconnect()
            return

    else:
        print(f"\nOpening {args.arm} gripper...")
        robot.open_gripper()
        time.sleep(1.0) 
        input(f"\n>>> File loaded. Place the ChArUco board into the {args.arm} gripper and press ENTER to close it and begin sampling: ")
        print(f"Closing {args.arm} gripper...")
        robot.close_gripper()
        time.sleep(1.0)

    # 6. Automatic Execution and Sampling
    print(f"\nStarting automatic sampling. Targets: {len(recorded_poses)}")
    samples = []
    
    for i, q in enumerate(recorded_poses):
        print(f"\nMoving {args.arm} arm to pose {i+1}/{len(recorded_poses)}...")
        robot.movej(q, blocking=True)
        
        print("Waiting for arm to settle and camera auto-exposure to adjust...")
        for _ in range(60):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                aligned = align.process(frames)
                color_frame = aligned.get_color_frame()
                if color_frame:
                    color_img = np.asanyarray(color_frame.get_data())
                    cv2.imshow('Camera View', color_img)
                    cv2.waitKey(1)
            except RuntimeError:
                pass
        
        pose_tcp = robot.get_tcp_pose()
        
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            print("Warning: RealSense frame timeout, skipping capture for this pose.")
            continue

        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            continue
            
        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=float)
        dist_coeffs = np.zeros((5, 1), dtype=float) 

        # --- ChArUco Detection (Two-Step Process)[cite: 2] ---
        if detector is not None:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
            
        if corners and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(color, corners, ids)
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            
            # --- AUTO-ROTATION FALLBACK ---
            if charuco_corners is None or len(charuco_corners) < 4:
                try:
                    swapped_board = cv2.aruco.CharucoBoard((sq_y, sq_x), sq_size, mk_size, aruco_dict)
                except AttributeError:
                    swapped_board = cv2.aruco.CharucoBoard_create(sq_y, sq_x, sq_size, mk_size, aruco_dict)
                    
                ret_s, charuco_corners_swap, charuco_ids_swap = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, swapped_board)
                if charuco_corners_swap is not None and len(charuco_corners_swap) >= 4:
                    print("🔄 Automatically swapped squares_x and squares_y to match board orientation.")
                    charuco_corners = charuco_corners_swap
                    charuco_ids = charuco_ids_swap
                    board = swapped_board 

            if charuco_corners is not None and len(charuco_corners) >= 4:
                # SolvePnP Robust Fix[cite: 2]
                all_obj_points = board.getChessboardCorners()
                obj_points = all_obj_points[charuco_ids.flatten()]
                
                valid, rvec, tvec = cv2.solvePnP(obj_points, charuco_corners, camera_matrix, dist_coeffs)
                if valid:
                    samples.append({'robot_pose': pose_tcp, 'rvec': rvec.flatten().tolist(), 'tvec': tvec.flatten().tolist()})
                    print(f"✅ Sample {i+1} captured.")
                    cv2.drawFrameAxes(color, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    cv2.aruco.drawDetectedCornersCharuco(color, charuco_corners, charuco_ids)
                else:
                    print("❌ Pose estimation failed.[cite: 2]")
            else:
                num_found = len(charuco_corners) if charuco_corners is not None else 0
                print(f"❌ Not enough ChArUco corners (Found {num_found}, Need 4+).[cite: 2]")
        else:
            print("❌ No ArUco markers detected. (Check Dict 50 vs 250)[cite: 2]")

        cv2.imshow('Camera View', color)
        cv2.waitKey(1500)

    # 7. Process Solutions
    if samples:
        run_hand_eye(samples, calib_file)

    print("\nCalibration sequence complete.")
    print(f"Opening {args.arm} gripper to release board...")
    robot.open_gripper()
    time.sleep(1.0)
    
    pipeline.stop()
    cv2.destroyAllWindows()
    robot.disconnect()

if __name__ == "__main__":
    main()