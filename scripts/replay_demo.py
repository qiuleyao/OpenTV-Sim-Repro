from isaacgym import gymapi
from isaacgym import gymutil

import math
import numpy as np
import matplotlib.pyplot as plt

from pytransform3d import rotations

from pathlib import Path
import h5py
from tqdm import tqdm
import time
import yaml
import torch
import cv2
from torchvision.transforms import v2
from sklearn.decomposition import PCA

class Player:
    def __init__(self, dt=1/60, record_path=None, record_fps=None):
        self.dt = dt
        self.head_mat = None
        self.left_wrist_mat = None
        self.right_wrist_mat = None
        self.left_hand_pos = None
        self.right_hand_pos = None

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = dt
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        robot_asset_root = "../assets"
        robot_asset_file = 'h1_inspire/urdf/h1_inspire.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, asset_options)
        dof = self.gym.get_asset_dof_count(robot_asset)

        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(17)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # robot
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.8, 0, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.robot_handle = self.gym.create_actor(self.env, robot_asset, pose, 'robot', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.robot_handle, np.zeros(dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # offscreen camera framing the upper body, used for video recording
        self.video_writer = None
        if record_path is not None:
            cam_props = gymapi.CameraProperties()
            cam_props.width = 640
            cam_props.height = 480
            self.cam_handle = self.gym.create_camera_sensor(self.env, cam_props)
            self.gym.set_camera_location(self.cam_handle, self.env,
                                         gymapi.Vec3(0.4, 0.5, 1.8), gymapi.Vec3(-0.8, 0, 1.3))
            Path(record_path).parent.mkdir(parents=True, exist_ok=True)
            fps = record_fps if record_fps is not None else 1 / dt
            self.video_writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (640 * 3, 480))
            print(f"Recording video to {record_path} at {fps:.0f} fps")

        plt.figure(figsize=(12, 6))
        plt.ion()

    def step(self, action, left_img, right_img):
        qpos = self.convert_h1_qpos(action)
        states = np.zeros(qpos.shape, dtype=gymapi.DofState.dtype)
        states['pos'] = qpos
        self.gym.set_actor_dof_states(self.env, self.robot_handle, states, gymapi.STATE_POS)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)

        left_img = left_img.transpose((1, 2, 0))
        right_img = right_img.transpose((1, 2, 0))
        img = np.concatenate((left_img, right_img), axis=1)

        plt.cla()
        plt.title('VisionPro View')
        plt.imshow(img, aspect='equal')
        plt.pause(0.001)

        if self.video_writer is not None:
            self.gym.render_all_camera_sensors(self.sim)
            sim_img = self.gym.get_camera_image(self.sim, self.env, self.cam_handle, gymapi.IMAGE_COLOR)
            sim_img = sim_img.reshape(480, 640, 4)[:, :, :3]
            frame = np.concatenate((sim_img, img), axis=1)
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def end(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        plt.close()

    def convert_h1_qpos(self, action):
        '''
        left_arm_indices = [13, 14, 15, 16, 17, 18, 19]
        right_arm_indices = [32, 33, 34, 35, 36, 37, 38]
        left_hand_indices = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        right_hand_indices = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        '''
        qpos = np.zeros(51)
        qpos[13:20] = action[0:7]

        # left hand actions
        qpos[20:22] = action[7]
        qpos[22:24] = action[8]
        qpos[24:26] = action[9]
        qpos[26:28] = action[10]
        qpos[28] = action[11]
        qpos[29:32] = action[12] * np.array([1, 1.6, 2.4])

        qpos[32:39] = action[13:20]

        # right hand actions
        qpos[39:41] = action[20]
        qpos[41:43] = action[21]
        qpos[43:45] = action[22]
        qpos[45:47] = action[23]
        qpos[47] = action[24]
        qpos[48:51] = action[25] * np.array([1, 1.6, 2.4])

        return qpos

if __name__ == '__main__':

    root = "../data/recordings/"
    # folder_name = "00-coke_can_insert-2024_05_26-23_50_58/processed"
    folder_name = "00-can-sorting/processed"
    episode_name = "processed_episode_0.hdf5"
    episode_path = Path(root) / folder_name / episode_name

    data = h5py.File(str(episode_path), 'r')
    actions = np.array(data['qpos_action'])[::2]
    left_imgs = np.array(data['observation.image.left'])[::2]  # 30hz
    right_imgs = np.array(data['observation.image.right'])[::2]
    data.close()

    timestamps = actions.shape[0]

    player = Player(1/30)
    
    try:
        for t in tqdm(range(timestamps)):
            player.step(actions[t], left_imgs[t, :], right_imgs[t, :])
    except KeyboardInterrupt:
        player.end()
        exit()
