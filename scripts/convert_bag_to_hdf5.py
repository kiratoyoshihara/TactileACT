import argparse
import h5py
import numpy as np
import cv2
import rosbag
from cv_bridge import CvBridge

TOPICS = {
    'vision':  '/camera/image_raw',
    'tactile': '/gelsight/image_raw',
    'kuka':    '/lbr_state',           
    'hande':   '/hande/joint_states',  
}

def interpolate_data(target_time, source_times, source_values):
    """
    Linearly interpolate data to match the target timestamp.
    source_values: [T, D]
    """
    if len(source_times) == 0:
        return np.zeros(source_values.shape[1])

    idx = np.searchsorted(source_times, target_time)
    
    if idx == 0:
        return source_values[0]
    if idx == len(source_times):
        return source_values[-1]

    t0 = source_times[idx - 1]
    t1 = source_times[idx]
    v0 = source_values[idx - 1]
    v1 = source_values[idx]

    if t1 == t0: return v0

    alpha = (target_time - t0) / (t1 - t0)
    alpha = np.clip(alpha, 0, 1)
    
    return v0 + alpha * (v1 - v0)

def process_bag(bag_path, output_path):
    print(f"Processing ROS 1 bag: {bag_path} ...")
    
    bridge = CvBridge()

    vision_data = [] 
    tactile_data = []
    
    kuka_times = []
    kuka_values = []
    
    hande_times = []
    hande_values = []

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=list(TOPICS.values())):
            timestamp = msg.header.stamp.to_sec() if hasattr(msg, 'header') else t.to_sec()

            if topic == TOPICS['vision']:
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                    vision_data.append((timestamp, cv_image))
                except Exception as e:
                    print(f"Error decoding vision image: {e}")
            
            elif topic == TOPICS['tactile']:
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                    tactile_data.append((timestamp, cv_image))
                except Exception as e:
                    print(f"Error decoding tactile image: {e}")
                
            elif topic == TOPICS['kuka']:
                if hasattr(msg, 'measured_joint_position'):
                    vals = msg.measured_joint_position
                elif hasattr(msg, 'position'):
                    vals = msg.position
                else:
                    continue
                
                kuka_times.append(timestamp)
                kuka_values.append(np.array(vals, dtype=np.float32))

            elif topic == TOPICS['hande']:
                if hasattr(msg, 'position') and len(msg.position) > 0:
                    val = np.array([msg.position[0]], dtype=np.float32)
                    hande_times.append(timestamp)
                    hande_values.append(val)

    kuka_times = np.array(kuka_times)
    kuka_values = np.array(kuka_values)
    hande_times = np.array(hande_times)
    hande_values = np.array(hande_values)

    print(f"  Found: Vision={len(vision_data)}, KUKA={len(kuka_values)}, HandE={len(hande_values)}")

    # Synchronization (Master Clock = Tactile)
    final_qpos = []
    final_vision = []
    final_tactile = []
    
    # Prepare vision timestamps for search
    vision_times = [t for t, _ in vision_data]
    
    if len(tactile_data) == 0:
        print("Error: No tactile data found.")
        return

    # Loop based on Tactile (approx 15-20Hz)
    for i in range(len(tactile_data)):
        t_tac, img_tac = tactile_data[i]

        # Find closest Vision image
        if len(vision_times) > 0:
            idx_vis = (np.abs(np.array(vision_times) - t_tac)).argmin()
            _, img_vis = vision_data[idx_vis]
        else:
            img_vis = np.zeros_like(img_tac)

        # Interpolate KUKA
        if len(kuka_values) > 0:
            k_curr = interpolate_data(t_tac, kuka_times, kuka_values)
        else:
            k_curr = np.zeros(7) 

        # Interpolate Hand-E
        if len(hande_values) > 0:
            h_curr = interpolate_data(t_tac, hande_times, hande_values)
        else:
            h_curr = np.zeros(1)

        q_curr = np.concatenate([k_curr, h_curr]) # [8]

        final_vision.append(img_vis)
        final_tactile.append(img_tac)
        final_qpos.append(q_curr)

    # Create Actions
    final_qpos = np.array(final_qpos)
    final_action = np.zeros_like(final_qpos)
    final_action[:-1] = final_qpos[1:] 
    final_action[-1] = final_qpos[-1]

    # Save
    final_vision = np.array(final_vision)
    final_tactile = np.array(final_tactile)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('qpos', data=final_qpos)
        f.create_dataset('action', data=final_action)
        f.create_dataset('vision', data=final_vision, compression="gzip")
        f.create_dataset('tactile', data=final_tactile, compression="gzip")
        f.attrs['num_samples'] = len(final_qpos)

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=str, required=True, help='Path to input .bag file')
    parser.add_argument('--out', type=str, required=True, help='Path to output .h5 file')
    args = parser.parse_args()
    
    process_bag(args.bag, args.out)