import rclpy
from rclpy.node import Node
import torch
import numpy as np
import json
import os
import cv2
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from torchvision import transforms
from PIL import Image as PILImage

from configs.config import Config
from models.tactile_act import TactileACT
from models.tactile_diffusion import TactileDiffusion 

class TemporalEnsembler:
    def __init__(self, horizon, action_dim, device):
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device

        # cumulative buffer: [Horizon, Action_Dim]
        self.action_sum = torch.zeros((horizon, action_dim), device=device)
        self.weight_sum = torch.zeros((horizon, 1), device=device)

        # Exponential Weighting
        k = 0.01
        # [Horizon, 1]
        self.weights = torch.exp(-k * torch.arange(horizon, device=device)).unsqueeze(1)

    def update(self, new_action_chunk):
        self.action_sum += new_action_chunk * self.weights
        self.weight_sum += self.weights

        current_action = self.action_sum[0] / self.weight_sum[0]

        self.action_sum[:-1] = self.action_sum[1:].clone()
        self.action_sum[-1] = 0.0 

        self.weight_sum[:-1] = self.weight_sum[1:].clone()
        self.weight_sum[-1] = 0.0

        return current_action

class InferenceNode(Node):
    def __init__(self):
        super().__init__('tactile_act_inference') # tactile_diffusion_inference
        
        self.cfg = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Inference device: {self.device}")
        
        if self.cfg.MODEL_TYPE == 'ACT':
            self.model = TactileACT(self.cfg).to(self.device)
        else:
            self.model = TactileDiffusion(self.cfg).to(self.device)
        
        chunk_size = self.cfg.CHUNK_SIZE
        action_dim = self.cfg.ACTION_DIM

        self.ensembler = TemporalEnsembler(chunk_size, action_dim, self.device)
        self.get_logger().info(f"Temporal Ensembling initialized (Horizon: {chunk_size})")

        ckpt_path = "checkpoints/best_model.pth" 
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            self.get_logger().info(f"Loaded weights from {ckpt_path}")
        else:
            self.get_logger().error(f"Checkpoint not found: {ckpt_path}")
        
        self.model.eval()

        # Denormalize
        stats_path = "data/stats.json" 
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        self.qpos_mean = torch.tensor(stats['action_mean'], device=self.device)
        self.qpos_std = torch.tensor(stats['action_std'], device=self.device)
        self.get_logger().info("Loaded stats.json for denormalization")

        # preprocess as training
        self.transform = transforms.Compose([
            transforms.Resize(self.cfg.IMG_HEIGHT),
            transforms.CenterCrop(self.cfg.IMG_HEIGHT),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # ROS Setup
        self.bridge = CvBridge()
        
        # Subscribers
        # TODO: check KUKA's topic name
        self.sub_joint = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 1)
        self.sub_vis = self.create_subscription(
            Image, '/camera/image_raw', self.vis_callback, 1)
        self.sub_tac = self.create_subscription(
            Image, '/gelsight/image_raw', self.tac_callback, 1)
        
        # Publisher
        # TODO: check KUKA's package name
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, '/forward_position_controller/commands', 1)
        # attention map visualizer
        self.pub_vis_debug = self.create_publisher(Image, '/debug/attention_map', 1)
       
        # Buffers
        self.curr_qpos = None
        self.curr_vis = None
        self.curr_tac = None

        self.step_cnt = 0     
        self.vis_interval = 20 # every 1 sec for 20Hz

        # TODO: may need some tuning
        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz

    def joint_callback(self, msg):
        # TODO: ensure JointState order is the same order as training data
        self.curr_qpos = np.array(msg.position, dtype=np.float32)

    def vis_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.curr_vis = PILImage.fromarray(cv_img)

    def tac_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.curr_tac = PILImage.fromarray(cv_img)

    def control_loop(self):
        # wait for data
        if self.curr_qpos is None or self.curr_vis is None or self.curr_tac is None:
            return

        # Joint State 
        qpos_tensor = torch.from_numpy(self.curr_qpos).to(self.device)
        qpos_norm = (qpos_tensor - self.qpos_mean) / self.qpos_std
        qpos_norm = qpos_norm.unsqueeze(0) # Batch dim: [1, 8]

        # Images 
        vis_tensor = self.transform(self.curr_vis).to(self.device).unsqueeze(0) # [1, 3, 224, 224]
        tac_tensor = self.transform(self.curr_tac).to(self.device).unsqueeze(0) # [1, 3, 224, 224]

        with torch.no_grad():
            if self.cfg.MODEL_TYPE == 'ACT':
                # ACT: [1, Chunk, 8] -> Temporal Aggregation
                pred_chunk, _, _ = self.model(qpos_norm, vis_tensor, tac_tensor, is_training=False)
                raw_action_chunk = pred_chunk[0] 

                if (self.step_cnt % self.vis_interval == 0):
                    self.publish_attention_map(vis_tensor, tac_tensor)
                
            elif self.cfg.MODEL_TYPE == 'DIFFUSION':
                # Diffusion: [1, Chunk, 8] -> run 100 times loop
                pred_chunk = self.model.sample(qpos_norm, vis_tensor, tac_tensor)
                raw_action_chunk = pred_chunk[0]

        # Temporal Aggregation (Update & Smooth)
        smooth_action_norm = self.ensembler.update(raw_action_chunk) # [8]
        next_action_real = (smooth_action_norm * self.qpos_std) + self.qpos_mean
    
        # Publish 
        cmd_msg = Float64MultiArray()
        cmd_msg.data = next_action_real.cpu().numpy().tolist()
        self.pub_cmd.publish(cmd_msg)

        self.step_cnt += 1

    def publish_attention_map(self, vis_tensor, tac_tensor):
            try:
                # [1, Chunk, Tokens]
                attn_weights = self.model.get_attention_weights()
                if attn_weights is None: return

                attn_t0 = attn_weights[0, 0, :].cpu()
                
                start_vis = 2
                end_vis = 2 + 49
                start_tac = 2 + 49
                end_tac = 2 + 49 + 49
                
                attn_vis = attn_t0[start_vis:end_vis].reshape(7, 7).numpy()
                attn_tac = attn_t0[start_tac:end_tac].reshape(7, 7).numpy()

                def make_overlay(img_t, attn_map):
                    img = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img = (img * 0.229 + 0.485) * 255
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
                    
                    heatmap = cv2.resize(attn_map, (224, 224))

                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

                vis_out = make_overlay(vis_tensor, attn_vis)
                tac_out = make_overlay(tac_tensor, attn_tac)
                
                combined_img = np.hstack([vis_out, tac_out])
                
                msg = self.bridge.cv2_to_imgmsg(combined_img, encoding="bgr8")
                self.pub_vis_debug.publish(msg)
                
            except Exception as e:
                self.get_logger().warn(f"Vis Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()