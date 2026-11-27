from isaacgym import gymapi
from isaacgym import gymutil

import math
import numpy as np
import matplotlib.pyplot as plt

from replay_demo import Player

from pathlib import Path
import h5py
from tqdm import tqdm
import time
import yaml
import pickle
import torch
import cv2
from collections import deque
import argparse
import sys
sys.path.append("../")
from act.utils import parse_id
# from act.imitate_episodes import RECORD_DIR, DATA_DIR, LOG_DIR

from pathlib import Path
current_dir = Path(__file__).parent.resolve()
DATA_DIR = (current_dir.parent / 'data/').resolve()
RECORD_DIR = (DATA_DIR / 'recordings/').resolve()
LOG_DIR = (DATA_DIR / 'logs/').resolve()
# print(f"\nDATA dir: {DATA_DIR}")

def get_norm_stats(data_path):
    # norm_stats = {
    #     "action_mean": np.array([]), "action_std": np.array([]),
    #     "qpos_mean": np.array([]), "qpos_std": np.array([]),
    # }
    with open(data_path, "rb") as f:
        norm_stats = pickle.load(f)
    # tv_train may save with numpy 2.x; convert lists/arrays for tv (numpy 1.x)
    return {k: np.asarray(v, dtype=np.float32) for k, v in norm_stats.items()}


def load_policy(policy_path, device):
    policy = torch.jit.load(policy_path, map_location=device)
    return policy


def normalize_input(state, left_img, right_img, norm_stats, device, last_action_data=None):
    # import ipdb; ipdb.set_trace()
    # left_img = cv2.resize(left_img, (308, 224))
    # right_img = cv2.resize(right_img, (308, 224))
    image_data = torch.from_numpy(np.stack([left_img, right_img], axis=0)) / 255.0
    qpos_data = (torch.from_numpy(state) - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
    image_data = image_data.view((1, 2, 3, 480, 640)).to(device=device)
    qpos_data = qpos_data.view((1, 26)).to(device=device)

    if last_action_data is not None:
        last_action_data = torch.from_numpy(last_action_data).to(device=device).view((1, -1)).to(torch.float)
        qpos_data = torch.cat((qpos_data, last_action_data), dim=1)
    return (qpos_data, image_data)


def merge_act(actions_for_curr_step, k = 0.01):
    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]

    exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[0]))
    exp_weights = (exp_weights / exp_weights.sum()).reshape((-1, 1))
    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)

    return raw_action


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--taskid', action='store', type=str, help='task id', required=True)
    parser.add_argument('--exptid', action='store', type=str, help='experiment id', required=True)
    parser.add_argument('--resume_ckpt', action='store', type=str, help='resume checkpoint', required=True)
    parser.add_argument('--record', action='store_true', help='save sim render + camera views to an mp4 in the exp dir')
    args = vars(parser.parse_args())

    episode_name = "processed_episode_0.hdf5"
    task_dir, task_name = parse_id(RECORD_DIR, args['taskid'])
    episode_path = (Path(task_dir) / 'processed' / episode_name).resolve()
    exp_path, _ = parse_id((Path(LOG_DIR) / task_name).resolve(), args['exptid'])
    
    norm_stat_path = Path(exp_path) / "dataset_stats.pkl"
    policy_path = Path(exp_path) / f"traced_jit_{args['resume_ckpt']}.pt"
    
    temporal_agg = True
    action_dim = 28

    chunk_size = 60
    # tv env PyTorch cannot use RTX 5060 (sm_120); run policy on CPU
    device = "cpu"

    # images (~9GB per episode) stay on disk and are read per frame
    episode_file = h5py.File(str(episode_path), 'r')
    actions = np.array(episode_file['qpos_action'])
    left_imgs = episode_file['observation.image.left']
    right_imgs = episode_file['observation.image.right']
    states = np.array(episode_file['observation.state'])
    init_action = np.array(episode_file.attrs['init_action'])
    timestamps = states.shape[0]

    norm_stats = get_norm_stats(norm_stat_path)
    policy = load_policy(policy_path, device)
    policy.to(device)
    policy.eval()

    history_stack = 0
    if history_stack > 0:
        last_action_queue = deque(maxlen=history_stack)
        for i in range(history_stack):
            last_action_queue.append(actions[0])
    else:
        last_action_queue = None
        last_action_data = None
    # recordings are 60Hz, so a 60fps video plays back in real time
    record_path = Path(exp_path) / f"deploy_sim_{args['resume_ckpt']}.mp4" if args['record'] else None
    player = Player(dt=1/30, record_path=record_path, record_fps=60)

    if temporal_agg:
        # only the last chunk_size predictions overlap the current step, oldest first
        recent_chunks = deque(maxlen=chunk_size)
    else:
        num_actions_exe = chunk_size
    
    try:
        output = None
        act_index = 0
        for t in tqdm(range(timestamps)):
            if history_stack > 0:
                last_action_data = np.array(last_action_queue)

            left_img, right_img = left_imgs[t], right_imgs[t]
            data = normalize_input(states[t], left_img, right_img, norm_stats, device, last_action_data)

            if temporal_agg:
                output = policy(*data)[0].detach().cpu().numpy() # (chunk_size, action_dim)
                recent_chunks.append(output)
                n = len(recent_chunks)
                act = merge_act(np.stack([chunk[n - 1 - i] for i, chunk in enumerate(recent_chunks)]))
            else:
                if output is None or act_index == num_actions_exe-1:
                    print("Inference...")
                    output = policy(*data)[0].detach().cpu().numpy()
                    act_index = 0
                act = output[act_index]
                act_index += 1
            # import ipdb; ipdb.set_trace()
            if history_stack > 0:
                last_action_queue.append(act)
            act = act * norm_stats["action_std"] + norm_stats["action_mean"]
            player.step(act, left_img, right_img)
    except KeyboardInterrupt:
        pass
    finally:
        player.end()
        episode_file.close()