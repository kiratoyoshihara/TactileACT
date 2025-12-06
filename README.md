# Tactile-ACT: Visuo-Tactile Action Chunking for Robust Robotic Disassembly
**Tactile-ACT** is a multimodal imitation learning framework extending [Action Chunking with Transformers (ACT)](https://arxiv.org/abs/2304.13705 "ACT") to incorporate high-resolution tactile feedback ([GelSight](https://www.gelsight.com/gelsightmini/ "GelSight")). This project aims to achieve robust robotic disassembly tasks under occlusion and variable lighting conditions by fusing vision and tactile modalities.

## 🤖 Hardware Setup
- **Robot Arm**: KUKA LBR iiwa 14 R820 (7 DoF)
- **Gripper**: Robotiq Hand-E (1 DoF)
- **Vision Sensor**: Intel RealSense D435 (RGB)
- **Tactile Sensor**: GelSight Mini
- **Environment** Ubuntu 22.04/ROS2 Humble 

## 📂 Project Structure
    TactileACT/
    ├── configs/
    │   └── config.py              # Hyperparameters 
    ├── data/
    │   └── dataset.py             # PyTorch Dataset class 
    ├── models/
    │   ├── backbones.py           # ResNet encoders (Vision & Tactile)
    │   ├── cvae.py                # CVAE module for action generation
    │   ├── tactile_act.py         # Main ACT model with Dual Encoders
    │   └── vis_transformer.py     # Custom Transformer for attention visualization
    ├── scripts/                   
    │   ├── bringup_launch.py      # Master launch file (Sensors + Drivers)
    │   ├── convert_bag_to_hdf5.py # Data processing (rosbag -> hdf5)
    │   ├── hande_driver_node.py   # Hand-E gripper driver
    │   ├── launch_sensors.py      # Camera & GelSight startup
    │   ├── record_episode.sh      # Data collection helper script
    │   ├── teleop_gripper.py      # Keyboard teleop for gripper during data collection
    │   └── vis_dataset.py         # Data verification tool
    ├── utils/
    │   ├── utils.py               # Helper functions
    │   ├── vis_attention.py       # Attention map visualization
    │   └── vis_trajectory.py      # Trajectory plotting
    ├── debug.py                   # Debugging script
    ├── inference_node.py          # Real-time inference ROS node
    ├── train.py                   # Training loop
    └── requirements.txt           # Python dependencies

## 🚀 Workflow
1. **Hardware Bring-up & Data Collection**:

    We collect human demonstration data using Kinesthetic Teaching (gravity compensation mode) for the arm and keyboard teleoperation for the gripper.
    
    1. Start KUKA Driver (In a separate terminal)

        ```bash
        ros2 launch lbr_fri_ros2_stack lbr_fri_ros2.launch.py
        ```
    2. Start System (Sensors & Gripper)

        ```bash
        ros2 launch scripts/bringup_launch.py
        ```
        - Starts RealSense `/camera/image_raw`
        - Starts GelSight `/gelsight/image_raw`
        - Starts Hand-E Driver `/hande/command` `/hande/joint_states`
    3. Start Teleoperation
        ```bash
        python3 scripts/teleop_gripper.py
        ```
        - Use [SPACE] key to toggle Open/Close
    4. Record Episode
        ```bash
        scripts/record_episode.sh episode_001
        ```        

2. **Data Processing**:

    Convert raw ROS bags into HDF5 format for training. This script handles synchronization between 1kHz robot states and ~30Hz camera/gelsight.
    ```bash
    # Convert single episode
    python3 scripts/convert_bag_to_hdf5.py --bag data/rosbags/episode_001 --out data/episode_001.h5

    # Verify the converted data (Check images and alignment)
    python3 scripts/vis_dataset.py --dataset data/dataset.h5
    ```
3. **Training**:

    Train the Tactile-ACT model. The model uses **Dual Encoders** (one for Vision, one for Tactile) and **Modality Dropout** to prevent visual dominance.
    ```bash
    # Edit configs/config.py if needed
    python3 train.py
    ```
    - Checkpoints are saved in `checkpoints/`
    - Statistics (mean/std) are saved in `data/stats.json`
4. **Real-World Inference**:

    Run the trained policy on the real robot.
    ```bash
    # Ensure 'checkpoints/best_model.pth' and 'data/stats.json' exist
    python3 inference_node.py
    ```
    - The node performs **Temporal Aggregation** for smooth control.
    - Real-time Attention Maps are published to `/debug/attention_map`

## ⭐️ Key Features
- **Dual Encoder Architecture**:

    Separate ResNet backbones for RGB and GelSight images to capture global context and local geometry respectively.
- **Modality Dropout**:

    Randomly masks vision tokens during training to force the policy to rely on tactile feedback.
- **Robust Synchronization**:

    Aligns multi-modal data (1kHz Proprioception, 30Hz Vision, ~15-25Hz Tactile) using the tactile timestamp as the master clock.

## 🔧 Dependencies
- ROS 2 Humble
- PyTorch
- `lbr_fri_ros2_stack` (For KUKA control)
- `usb_cam` (For camera streaming)
- Python packages: `h5py` `opencv-python` `rosbags` `matplotlib`