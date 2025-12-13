# Tactile-ACT: Visuo-Tactile Action Chunking for Robust Robotic Disassembly
**Tactile-ACT** is a multimodal imitation learning framework extending [Action Chunking with Transformers (ACT)](https://arxiv.org/abs/2304.13705 "ACT") to incorporate high-resolution tactile feedback ([GelSight](https://www.gelsight.com/gelsightmini/ "GelSight")). This project aims to achieve robust robotic disassembly tasks under occlusion and variable lighting conditions by fusing vision and tactile modalities.

## 🤖 Hardware Setup
- **Robot Arm**: KUKA LBR iiwa 14 R820 (7 DoF)
- **Gripper**: Robotiq Hand-E (1 DoF)
- **Vision Sensor**: Intel RealSense D435 (RGB)
- **Tactile Sensor**: GelSight Mini
- **Environment** Ubuntu 20.04/ROS Noetic 

## 📂 Project Structure
    TactileACT/
    ├── configs/
    │   └── config.py                   # Hyperparameters 
    ├── data/
    │   └── dataset.py                  # PyTorch Dataset class 
    ├── launch/
    │   ├── bringup.launch              # launch Hand-E driver and sensors
    │   ├── sensors.launch              # launch GelSight/RealSense topic 
    │   └── tf_publisher.launch         # launch topic for Rviz
    ├── models/
    │   ├── backbones.py                # ResNet encoders (Vision & Tactile)
    │   ├── cvae.py                     # CVAE module for action generation
    │   ├── tactile_act.py              # Main ACT model with Dual Encoders
    │   └── vis_transformer.py          # Custom Transformer for attention visualization
    ├── scripts/     
    │   ├── utilities/                  # utilities from gsrobotics           
    │   ├── convert_bag_to_hdf5.py      # Data processing (rosbag -> hdf5)
    │   ├── hande_driver_node.py        # Hand-E gripper node driver
    │   ├── gelsight_driver_node.py     # Gelsight Mini node driver
    │   ├── record_episode.sh           # Data collection helper script
    │   ├── teleop_gripper.py           # Keyboard teleop for gripper during data collection
    │   ├── teach_mode.py               # Impdedance Control Mode
    │   ├── replay_motion.py            # Replay KUKA's motion
    │   └── vis_dataset.py              # Data verification tool
    ├── utils/
    │   ├── utils.py                    # Helper functions
    │   ├── vis_attention.py            # Attention map visualization
    │   └── vis_trajectory.py           # Trajectory plotting
    ├── debug.py                        # Debugging script
    ├── inference_node.py               # Real-time inference ROS node
    ├── train.py                        # Training loop
    ├── package.xml                     # package dependencies
    ├── CMakeLists.txt                  # build config
    └── requirements.txt                # Python dependencies

## 🐳 Docker Setup (Recommended)

This project allows you to develop inside a Docker container using **VS Code Dev Containers**. This ensures a consistent environment with ROS Noetic and all dependencies pre-installed.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine on Linux)
- [Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code

### How to Start
1. Open the project in VS Code.
2. You should see a notification: *"Folder contains a Dev Container configuration file. Reopen to folder to develop in a container."*
3. Click **Reopen in Container**. (Or press `F1` and select **Dev Containers: Reopen in Container**).
4. VS Code will build the Docker image and start the container. This may take a few minutes the first time.

### Build Workspace inside Container
Once the container is running, the VS Code terminal will open inside the repository folder. You need to build the workspace:

```bash
# 1. Go to the workspace root
cd /root/catkin_ws

# 2. Build the package
catkin_make

# 3. Source the environment
source devel/setup.bash
```

Now you can use `roslaunch tactile_act ...` command directly inside the VS Code terminal.

**Note on GUI**: The container is configured with X11 forwarding (`--net=host`, `/tmp/.X11-unix1`, etc.). You should be able to launch GUI applications like `Rviz` or `OpenCV windows` directly from the container.

## 🚀 Workflow

### 1. Hardware Bring-up & Data Collection:

We employ a **"Ghost Replay" strategy** to collect high-quality visual data without human occlusion. 

1. **Phase 1**: Record the human's motion (trajectory & gripper commands) using Kinesthetic Teaching.
2. **Phase 2**: The robot replays the motion automatically while recording all camera/tactile streams.

---

#### Phase 1: Human Demonstration (Trajectory Recording)
*Goal: Record `_human.bag` containing only robot states and gripper commands.*

1.  **Start KUKA Driver (Impedance Mode)**
    * On SmartPad: Select **Impedance Control Mode**.
    * Terminal 1:
        ```bash
        roslaunch lbr_fri_ros_stack lbr_driver.launch
        ```

2.  **Start Drivers**
    * Terminal 2:
        ```bash
        roslaunch tactile_act bringup.launch
        ```

3.  **Start Teleoperation (Gripper)**
    * Terminal 3:
        ```bash
        python3 scripts/teleop_gripper.py
        ```
        * Use **[SPACE]** key to toggle Open/Close.

4.  **Start Teach Mode (Arm Follower)**
    * To prevent the robot from springing back in Impedance Mode (Transparency Control):
    * Terminal 4:
        ```bash
        python3 scripts/teach_mode.py
        ```

5.  **Record Trajectory**
    * Terminal 5:
        ```bash
        # Save as a temporary file (e.g., episode_001_human)
        scripts/record_episode.sh episode_001_human
        ```
    * Perform the task by holding the robot. **Stop recording (Ctrl+C)** when finished.

---

#### Phase 2: Ghost Replay (Clean Data Collection)
*Goal: Record the final `episode_001.bag` with clean visual data (no human hands).*

1.  **Switch KUKA Mode**
    * **Stop** `teach_mode.py` (Ctrl+C in Terminal 4).
    * On SmartPad: Switch to **Position Control Mode**.

2.  **Check Sensors**
    * Ensure cameras and GelSight are positioned correctly.
    * Terminal 5 (Check topics):
        ```bash
        rostopic hz /camera_top/image_raw /camera_side/image_raw /camera_hand/image_raw /gelsight_1/image_raw /gelsight_2/image_raw
        ```

3.  **Start Final Recording**
    * Terminal 5:
        ```bash
        # This is the dataset file used for training
        scripts/record_episode.sh episode_001
        ```

4.  **Run Replay**
    * Terminal 4:
        ```bash
        # Replay the motion from Phase 1
        python3 scripts/replay_motion.py --bag data/rosbags/episode_001_human.bag
        ```
    * The robot will move to the start position and replay the motion.
    * **Stop recording** in Terminal 5 once the motion is complete.

---

### 2. Data Processing:

Convert raw ROS1 bags (`.bag`) into H5 format for training. This script handles synchronization between 1kHz robot states and ~30Hz camera/gelsight.

```bash
# Convert single episode
python3 scripts/convert_bag_to_hdf5.py --bag data/rosbags/episode_001.bag --out data/episode_001.h5

# Verify the converted data (Check images and alignment)
python3 scripts/vis_dataset.py --dataset data/episode_001.h5
```

### 3. **Training**:

Train the Tactile-ACT model. The model uses **Dual Encoders** (one for Vision, one for Tactile) and **Modality Dropout** to prevent visual dominance.
```bash
# Edit configs/config.py if needed
python3 train.py
```
- Checkpoints are saved in `checkpoints/`
- Statistics (mean/std) are saved in `data/stats.json`

### 4. **Real-World Inference**:

Run the trained policy on the real robot.
```bash
# Ensure 'checkpoint/best_model.pth' and 'data/stats.json' exist
rosrun tactile_act inference_node.py
```
or
```python
# Ensure 'checkpoints/best_model.pth' and 'data/stats.json' exist
python3 inference_node.py
```
- The node performs **Temporal Aggregation** for smooth control.
- Real-time Attention Maps are published to `/debug/attention_map`

## ⭐️ Key Features
- **Separate Encoder Architecture**:

    Separate ResNet backbones for RGB and GelSight images to capture global context and local geometry respectively.
- **Modality Dropout**:

    Randomly masks vision tokens during training to force the policy to rely on tactile feedback.
- **Robust Synchronization**:

    Aligns multi-modal data (1kHz Proprioception, 30Hz Vision, ~15-25Hz Tactile) using the tactile timestamp as the master clock.

## 🔧 Dependencies
- ROS 1 Humble
- PyTorch
- `lbr_fri_ros_stack` (For KUKA control)
- `usb_cam` (For camera streaming)
- Python packages: `h5py` `opencv-python` `rosbags` `matplotlib`