import rospy
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

from lbr_fri_msgs.msg import LBRState, LBRPositionCommand

from configs.config import Config
from models.tactile_act import TactileACT
from models.tactile_diffusion import TactileDiffusion 

class TemporalEnsembler:
    def __init__(self, horizon, action_dim, device):
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device

        # cumulative buffer: [Horizon, Action_DIm]
        self.action_sum = torch.zeros((horizon, action_dim), device=device)
        self.weight_sum = torch.zeros((horizon, 1), device=device)
        
        # Exponential Weighting
        k = 0.01
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

class InferenceNode:
    def __init__(self):
        rospy.init_node('tactile_act_inference', anonymous=True) # tactile_diffusion_inference
        
        self.cfg = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Inference device: {self.device}")
        
        # Model Load
        if self.cfg.MODEL_TYPE == 'ACT':
            self.model = TactileACT(self.cfg).to(self.device)
        else:
            self.model = TactileDiffusion(self.cfg).to(self.device)
        
        self.model.eval()
        
        # Checkpoint Loading
        ckpt_path = "checkpoints/best_model.pth" 
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            rospy.loginfo(f"Loaded weights from {ckpt_path}")
        else:
            rospy.logwarn(f"Checkpoint not found: {ckpt_path}")

        # Stats Loading
        stats_path = "data/stats.json" 
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.qpos_mean = torch.tensor(stats['action_mean'], device=self.device)
            self.qpos_std = torch.tensor(stats['action_std'], device=self.device)
            rospy.loginfo("Loaded stats.json")
        else:
            rospy.logerr("stats.json not found! Inference will be wrong.")
            # Dummy to prevent crash
            self.qpos_mean = torch.zeros(8, device=self.device)
            self.qpos_std = torch.ones(8, device=self.device)

        self.ensembler = TemporalEnsembler(self.cfg.CHUNK_SIZE, self.cfg.ACTION_DIM, self.device)

        self.transform = transforms.Compose([
            transforms.Resize(self.cfg.IMG_HEIGHT),
            transforms.CenterCrop(self.cfg.IMG_HEIGHT),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.bridge = CvBridge()
        
        # Subscribers
        # KUKA State 
        rospy.Subscriber('/lbr_state', LBRState, self.kuka_callback)
        # Hand-E State 
        rospy.Subscriber('/hande/joint_states', JointState, self.hande_callback)
        # Vision
        rospy.Subscriber('/camera_top/image_raw', Image, self.vis_top_callback)
        rospy.Subscriber('/camera_side/image_raw', Image, self.vis_side_callback)
        # Tactile
        rospy.Subscriber('/gelsight/image_raw', Image, self.tac_callback)
        
        # Publishers
        # KUKA Command 
        self.pub_kuka = rospy.Publisher('/lbr_command', LBRPositionCommand, queue_size=1)
        # Hand-E Command 
        self.pub_hande = rospy.Publisher('/hande/command', Float64MultiArray, queue_size=1)
        # Attention Visualization for debug
        self.pub_vis_debug = rospy.Publisher('/debug/attention_map', Image, queue_size=1)

        self.curr_kuka_qpos = None
        self.curr_hande_qpos = None
        self.curr_vis_top = None
        self.curr_vis_die = None
        self.curr_tac = None

        self.step_cnt = 0
        self.vis_interval = 20 # every 1sec for 20Hz

        # Control Loop (15Hz)
        self.rate = rospy.Rate(15)

    def kuka_callback(self, msg):
        self.curr_kuka_qpos = np.array(msg.measured_joint_position, dtype=np.float32)

    def hande_callback(self, msg):
        if len(msg.position) > 0:
            self.curr_hande_qpos = np.array([msg.position[0]], dtype=np.float32)

    def vis_top_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.curr_vis_top = PILImage.fromarray(cv_img)
        except Exception as e:
            pass

    def vis_side_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.curr_vis_side = PILImage.fromarray(cv_img)
        except Exception as e:
            pass

    def tac_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.curr_tac = PILImage.fromarray(cv_img)
        except Exception as e:
            pass

    def run(self):
        rospy.loginfo("Start Control Loop")
        while not rospy.is_shutdown():
            self.control_loop()
            self.rate.sleep()

    def control_loop(self):
        # Wait for all data
        if (self.curr_kuka_qpos is None or self.curr_hande_qpos is None or 
            self.curr_vis_top is None or self.curr_vis_side is None or self.curr_tac is None):
            return

        # Prepare Inputs
        # Combine KUKA(7) + HandE(1) = 8
        curr_qpos = np.concatenate([self.curr_kuka_qpos, self.curr_hande_qpos])
        
        qpos_tensor = torch.from_numpy(curr_qpos).to(self.device)
        qpos_norm = (qpos_tensor - self.qpos_mean) / self.qpos_std
        qpos_norm = qpos_norm.unsqueeze(0) 

        vis_top_tensor = self.transform(self.curr_vis_top).to(self.device).unsqueeze(0)
        vis_side_tensor = self.transform(self.curr_vis_side).to(self.device).unsqueeze(0)
        tac_tensor = self.transform(self.curr_tac).to(self.device).unsqueeze(0)

        # Inference
        with torch.no_grad():
            if self.cfg.MODEL_TYPE == 'ACT':
                pred_chunk, _, _ = self.model(qpos_norm, vis_top_tensor, vis_side_tensor, tac_tensor, is_training=False)
                raw_action_chunk = pred_chunk[0] 
                
                if (self.step_cnt % self.vis_interval == 0):
                    self.publish_attention_map(vis_top_tensor, vis_side_tensor, tac_tensor)
            else:
                pred_chunk = self.model.sample(qpos_norm, vis_tensor, tac_tensor) # TODO: modify Diffusion model to handle multi-view input
                raw_action_chunk = pred_chunk[0]

        # Update & Denormalize
        smooth_action_norm = self.ensembler.update(raw_action_chunk)
        next_action_real = (smooth_action_norm * self.qpos_std) + self.qpos_mean
        
        cmd_all = next_action_real.cpu().numpy()

        # KUKA 
        kuka_cmd = LBRPositionCommand()
        kuka_cmd.joint_position = cmd_all[:7].tolist()
        self.pub_kuka.publish(kuka_cmd)

        # Hand-E 
        hande_cmd = Float64MultiArray(data=cmd_all[7:].tolist())
        self.pub_hande.publish(hande_cmd)
        
        self.step_cnt += 1

    def publish_attention_map(self, vis_top_tensor, vis_side_tensor, tac_tensor):
        try:
            attn_weights = self.model.get_attention_weights()
            if attn_weights is None: return

            attn_t0 = attn_weights[0, 0, :].cpu()
            
            # Index (Style=1, State=1, Top=49, Side=49, Tac=49)
            start_top = 2
            end_top = 2 + 49
            start_side = end_top
            end_side = end_top + 49
            start_tac = end_side
            end_tac = end_side + 49
            
            attn_vis_top = attn_t0[start_top:end_top].reshape(7, 7).numpy()
            attn_vis_side = attn_t0[start_side:end_side].reshape(7, 7).numpy()
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

            vis_top_out = make_overlay(vis_top_tensor, attn_vis_top)
            vis_side_out = make_overlay(vis_side_tensor, attn_vis_side)
            tac_out = make_overlay(tac_tensor, attn_tac)
            
            combined_img = np.hstack([vis_top_out, vis_side_out, tac_out])
            
            msg = self.bridge.cv2_to_imgmsg(combined_img, encoding="bgr8")
            self.pub_vis_debug.publish(msg)
            
        except Exception as e:
            rospy.logwarn(f"Vis Error: {e}")

if __name__ == '__main__':
    try:
        node = InferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass