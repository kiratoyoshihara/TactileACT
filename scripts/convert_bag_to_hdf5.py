import argparse
import h5py
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

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
        # Return zeros if no data found
        return np.zeros(source_values.shape[1])

    # Find the insertion index for target_time
    idx = np.searchsorted(source_times, target_time)
    
    if idx == 0:
        return source_values[0]
    if idx == len(source_times):
        return source_values[-1]

    t0 = source_times[idx - 1]
    t1 = source_times[idx]
    v0 = source_values[idx - 1]
    v1 = source_values[idx]

    # Avoid division by zero
    if t1 == t0: return v0

    alpha = (target_time - t0) / (t1 - t0)
    alpha = np.clip(alpha, 0, 1)
    
    return v0 + alpha * (v1 - v0)

def process_bag(bag_path, output_path):
    print(f"Processing: {bag_path} ...")
    
    vision_data = [] 
    tactile_data = []
    
    kuka_times = []
    kuka_values = []
    
    hande_times = []
    hande_values = []

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            
            if topic == TOPICS['vision']:
                vision_data.append((timestamp, rawdata, connection.msgtype))
            
            elif topic == TOPICS['tactile']:
                tactile_data.append((timestamp, rawdata, connection.msgtype))
                
            elif topic == TOPICS['kuka']:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                if hasattr(msg, 'measured_joint_position'):
                    vals = msg.measured_joint_position
                elif hasattr(msg, 'position'):
                    vals = msg.position
                else:
                    continue 
                
                kuka_times.append(timestamp)
                kuka_values.append(np.array(vals, dtype=np.float32))

            elif topic == TOPICS['hande']:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                # TODO: Assuming JointState message
                if hasattr(msg, 'position') and len(msg.position) > 0:
                    val = np.array([msg.position[0]], dtype=np.float32)
                    hande_times.append(timestamp)
                    hande_values.append(val)

    # for faster access
    kuka_times = np.array(kuka_times)
    kuka_values = np.array(kuka_values)
    hande_times = np.array(hande_times)
    hande_values = np.array(hande_values)

    print(f"  Found: Vision={len(vision_data)}, KUKA={len(kuka_values)}, HandE={len(hande_values)}")

    # Synchronization (Master Clock = Vision)
    final_qpos = []
    final_vision = []
    final_tactile = []
    
    # Prepare tactile timestamps for Nearest Neighbor search
    tactile_times = [t for t, _, _ in tactile_data]
    
    for i in range(len(vision_data)):
        t_vis, vis_raw, vis_type = vision_data[i]
        
        vis_msg = deserialize_cdr(vis_raw, vis_type)
        np_arr = np.frombuffer(vis_msg.data, dtype=np.uint8)
        img_vis = np_arr.reshape((vis_msg.height, vis_msg.width, -1))
        if 'bgr' in vis_msg.encoding: 
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        if len(tactile_times) > 0:
            idx_tac = (np.abs(np.array(tactile_times) - t_vis)).argmin()
            t_tac, tac_raw, tac_type = tactile_data[idx_tac]
            tac_msg = deserialize_cdr(tac_raw, tac_type)
            np_arr_tac = np.frombuffer(tac_msg.data, dtype=np.uint8)
            img_tac = np_arr_tac.reshape((tac_msg.height, tac_msg.width, -1))
            if 'bgr' in tac_msg.encoding: 
                img_tac = cv2.cvtColor(img_tac, cv2.COLOR_BGR2RGB)
        else:
            # create black image if no tactile data found
            img_tac = np.zeros_like(img_vis)

        # Interpolate KUKA 
        if len(kuka_values) > 0:
            k_curr = interpolate_data(t_vis, kuka_times, kuka_values)
        else:
            k_curr = np.zeros(7) 

        # Interpolate Hand-E 
        if len(hande_values) > 0:
            h_curr = interpolate_data(t_vis, hande_times, hande_values)
        else:
            h_curr = np.zeros(1)

        q_curr = np.concatenate([k_curr, h_curr]) # [8]

        final_vision.append(img_vis)
        final_tactile.append(img_tac)
        final_qpos.append(q_curr)

    # Action at t is defined as qpos at t+1
    final_qpos = np.array(final_qpos)
    final_action = np.zeros_like(final_qpos)
    
    # Shift forward by 1 step
    final_action[:-1] = final_qpos[1:] 
    # Duplicate the last step
    final_action[-1] = final_qpos[-1]

    final_vision = np.array(final_vision)
    final_tactile = np.array(final_tactile)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('qpos', data=final_qpos)
        f.create_dataset('action', data=final_action)
        f.create_dataset('vision', data=final_vision, compression="gzip")
        f.create_dataset('tactile', data=final_tactile, compression="gzip")
        
        # Metadata
        f.attrs['num_samples'] = len(final_qpos)

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=str, required=True, help='Path to rosbag folder')
    parser.add_argument('--out', type=str, required=True, help='Path to output hdf5')
    args = parser.parse_args()
    
    process_bag(args.bag, args.out)