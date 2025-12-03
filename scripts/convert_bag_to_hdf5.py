import argparse
import h5py
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

# Define Topic Names
TOPICS = {
    'vision': '/camera/image_raw',
    'tactile': '/gelsight/image_raw',
    'qpos': '/joint_states', # KUKA joint states
    # For now, assuming gripper is part of joint_states 
}

def interpolate_joints(target_time, joint_times, joint_values):
    """
    Linearly interpolate joint positions to match the target image timestamp.
    """
    # Find the index where target_time fits
    idx = np.searchsorted(joint_times, target_time)
    
    if idx == 0 or idx == len(joint_times):
        return joint_values[0] if idx == 0 else joint_values[-1]

    t0 = joint_times[idx - 1]
    t1 = joint_times[idx]
    v0 = joint_values[idx - 1]
    v1 = joint_values[idx]

    # Linear interpolation
    alpha = (target_time - t0) / (t1 - t0)
    # Clip alpha to [0, 1] to avoid extrapolation issues
    alpha = np.clip(alpha, 0, 1)
    
    interpolated = v0 + alpha * (v1 - v0)
    return interpolated

def process_bag(bag_path, output_path):
    print(f"Processing: {bag_path} ...")
    
    # Temporary storage
    vision_data = [] # (time, image_bytes)
    tactile_data = []
    qpos_times = []
    qpos_values = []

    # Read Rosbag
    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            
            # Vision
            if topic == TOPICS['vision']:
                # msg = deserialize_cdr(rawdata, connection.msgtype)
                # Store raw bytes for later decoding to save memory during loop
                vision_data.append((timestamp, rawdata, connection.msgtype))
                
            # Tactile
            elif topic == TOPICS['tactile']:
                tactile_data.append((timestamp, rawdata, connection.msgtype))
                
            # Joint States (KUKA)
            elif topic == TOPICS['qpos']:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                # TODO:Ensure the order of joints matches KUKA's joint
                qpos_times.append(timestamp)
                qpos_values.append(np.array(msg.position, dtype=np.float32))

    # Convert joints to numpy for faster access
    qpos_times = np.array(qpos_times)
    qpos_values = np.array(qpos_values)

    print(f"  Found {len(vision_data)} vision frames, {len(qpos_values)} joint states.")

    # Synchronization (Master Clock = Vision)
    # We will align tactile and qpos to vision timestamps
    final_qpos = []
    final_vision = []
    final_tactile = []
    final_action = [] 

    # Sort tactile data by time for search
    tactile_times = [t for t, _, _ in tactile_data]
    
    for i in range(len(vision_data)):
        t_vis, vis_raw, vis_type = vision_data[i]
        
        # Decode Vision Image
        vis_msg = deserialize_cdr(vis_raw, vis_type)
        # Assuming sensor_msgs/Image (encoding='rgb8' or 'bgr8')
        np_arr = np.frombuffer(vis_msg.data, dtype=np.uint8)
        img_vis = np_arr.reshape((vis_msg.height, vis_msg.width, -1))
        if 'bgr' in vis_msg.encoding:
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        # Find Corresponding Tactile Image (Nearest Neighbor)
        # Find closest tactile timestamp
        idx_tac = (np.abs(np.array(tactile_times) - t_vis)).argmin()
        
        t_tac, tac_raw, tac_type = tactile_data[idx_tac]
        tac_msg = deserialize_cdr(tac_raw, tac_type)
        np_arr_tac = np.frombuffer(tac_msg.data, dtype=np.uint8)
        img_tac = np_arr_tac.reshape((tac_msg.height, tac_msg.width, -1))
        if 'bgr' in tac_msg.encoding:
            img_tac = cv2.cvtColor(img_tac, cv2.COLOR_BGR2RGB)

        # Interpolate Joint States 
        q_current = interpolate_joints(t_vis, qpos_times, qpos_values)
       
        final_vision.append(img_vis)
        final_tactile.append(img_tac)
        final_qpos.append(q_current)

    final_qpos = np.array(final_qpos) # [T, 8]
    final_vision = np.array(final_vision) # [T, H, W, 3]
    final_tactile = np.array(final_tactile) # [T, H, W, 3]

    # Create Actions (Shifted qpos)
    # qpos: [q_0, q_1, ..., q_{T-1}, q_T]
    # action: [q_1, q_2, ..., q_T, q_T] 
    final_action = np.zeros_like(final_qpos)
    # copy one step ahead qpos
    final_action[:-1] = final_qpos[1:] 
    final_action[-1] = final_qpos[-1]

    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('qpos', data=final_qpos)
        f.create_dataset('vision', data=final_vision, compression="gzip") # Compress images
        f.create_dataset('tactile', data=final_tactile, compression="gzip")
        f.create_dataset('action', data=final_action) 
        
        # Metadata
        f.attrs['num_samples'] = len(final_qpos)

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=str, required=True, help='Path to rosbag folder')
    parser.add_argument('--out', type=str, required=True, help='Path to output hdf5')
    args = parser.parse_args()
    
    process_bag(args.bag, args.out)