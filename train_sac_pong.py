import os
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py
import argparse
import torch.nn.functional as F
import random
from collections import deque

gym.register_envs(ale_py)

# ============================================================
# 1. Coordinate Environment (Same as before)
# ============================================================

class PongCoordinateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.history_length = 4
        self.coord_history = []
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.history_length * 3,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        coords = self.extract_coords(obs)
        self.coord_history = [coords] * self.history_length
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        coords = self.extract_coords(obs)
        self.coord_history.pop(0)
        self.coord_history.append(coords)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(self.coord_history).astype(np.float32)

    def extract_coords(self, frame):
        y_min, y_max = 34, 193
        paddle_mask = (frame[y_min:y_max, :, 0] < 100) & (frame[y_min:y_max, :, 1] > 150) & (frame[y_min:y_max, :, 2] < 100)
        paddle_indices = np.argwhere(paddle_mask)
        paddle_y = paddle_indices[:, 0].mean() + y_min if len(paddle_indices) > 0 else 0.0
        ball_mask = (frame[y_min:y_max, :, 0] > 230) & (frame[y_min:y_max, :, 1] > 230) & (frame[y_min:y_max, :, 2] > 230)
        ball_indices = np.argwhere(ball_mask)
        if len(ball_indices) > 0:
            ball_y, ball_x = ball_indices[:, 0].mean() + y_min, ball_indices[:, 1].mean()
        else:
            ball_x, ball_y = 0.0, 0.0
        h, w, _ = frame.shape
        return np.array([paddle_y / h, ball_x / w, ball_y / h])

def create_pong_env(env_name="PongNoFrameskip-v4", frame_skip=4):
    env = gym.make(env_name, render_mode=None, frameskip=frame_skip, repeat_action_probability=0.0)
    env = PongCoordinateWrapper(env)
    return env

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(state), torch.tensor(action), torch.tensor(reward, dtype=torch.float32),
                torch.stack(next_state), torch.tensor(done, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

# ============================================================
# 2. Actor-Critic Network
# ============================================================
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class DiscreteActor(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, n_actions), std=0.01)
        )

    def forward(self, state):
        return self.network(state)

    def get_action(self, state):
        logits = self.forward(state)
        log_pi = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        return probs, log_pi

class DiscreteCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.q1 = nn.Sequential(layer_init(nn.Linear(input_dim, 512)), nn.ReLU(), layer_init(nn.Linear(512, n_actions)))
        self.q2 = nn.Sequential(layer_init(nn.Linear(input_dim, 512)), nn.ReLU(), layer_init(nn.Linear(512, n_actions)))

    def forward(self, state):
        return self.q1(state), self.q2(state)

class SACAgent:
    def __init__(self, obs_dim, n_actions, cfg, device):
        self.device = device
        self.actor = DiscreteActor(obs_dim, n_actions).to(device)
        self.critic = DiscreteCritic(obs_dim, n_actions).to(device)
        self.critic_target = DiscreteCritic(obs_dim, n_actions).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr, eps=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr, eps=1e-4)

        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.lr, eps=1e-4)

        self.target_entropy = -0.89 * np.log(1.0 / n_actions)
        self.gamma, self.tau = cfg.gamma, cfg.tau

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.actor.get_action(obs)
            action = torch.argmax(probs, dim=-1) if deterministic else torch.distributions.Categorical(probs=probs).sample()
        return action.item()
    
    def update(self, batch):
        s, a, r, s_next, d = [x.to(self.device) for x in batch]
        alpha = self.log_alpha.exp()

        # Critic Update
        with torch.no_grad():
            next_pi, next_log_pi = self.actor.get_action(s_next)
            q1_t, q2_t = self.critic_target(s_next)
            min_q_t = torch.min(q1_t, q2_t)
            v_next = (next_pi * (min_q_t - alpha * next_log_pi)).sum(dim=1)
            target_q = r + (1 - d) * self.gamma * v_next

        q1, q2 = self.critic(s)
        q1_pred = q1.gather(1, a.unsqueeze(1).long()).squeeze(1)
        q2_pred = q2.gather(1, a.unsqueeze(1).long()).squeeze(1)
        
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor Update
        pi, log_pi = self.actor.get_action(s)
        with torch.no_grad():
            q1_curr, q2_curr = self.critic(s)
            min_q_curr = torch.min(q1_curr, q2_curr)
        
        actor_loss = (pi * (alpha * log_pi - min_q_curr)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha Update
        alpha_loss = (pi.detach() * (-self.log_alpha.exp() * (log_pi.detach() + self.target_entropy))).sum(1).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft Target Update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # Additional metrics calculation
        with torch.no_grad():
            entropy = -(pi * log_pi).sum(dim=1).mean()
            delta_q12 = torch.abs(q1_pred - q2_pred).mean()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha_value": alpha.item(),
            "entropy": entropy.item(),
            "delta_q12": delta_q12.item(),
            "target_q": target_q.mean().item(),
            "predicted_q": ((q1_pred + q2_pred) / 2).mean().item()
        }

def train_sac(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.save_dir, exist_ok=True)
    env = create_pong_env()
    agent = SACAgent(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n, cfg=args, device=device)
    replay_buffer = ReplayBuffer(args.buffer_size)
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Tracking variables
    global_step, episode_idx, update_step, best_reward = 0, 0, 0, -float("inf")
    # Buffer for average reward
    reward_window = deque(maxlen=100)
    history = {k: [] for k in ["reward_history", "critic_loss", "actor_loss", "alpha_loss", "alpha_value", "entropy", "delta_q12", "target_q", "predicted_q"]}

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        agent.actor.load_state_dict(ckpt["actor_state"])
        agent.critic.load_state_dict(ckpt["critic_state"])
        agent.critic_target.load_state_dict(ckpt["critic_target_state"])
        agent.actor_opt.load_state_dict(ckpt["actor_opt_state"])
        agent.critic_opt.load_state_dict(ckpt["critic_opt_state"])
        agent.alpha_opt.load_state_dict(ckpt["alpha_opt_state"])
        agent.log_alpha.data = ckpt["log_alpha"].data
        history, global_step, episode_idx, update_step = ckpt["history"], ckpt["global_step"], ckpt["episode_idx"], ckpt["update_step"]
        best_reward = ckpt.get("best_reward", -float("inf"))
        reward_window.extend(history["reward_history"][-100:])

    observation, _ = env.reset(seed=42)
    episode_reward, latest_update_metrics, last_saved_episode = 0.0, None, 0

    while global_step < args.total_steps:
        action = env.action_space.sample() if global_step < args.learning_starts else agent.select_action(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(torch.tensor(observation, dtype=torch.float32), action, reward, torch.tensor(next_observation, dtype=torch.float32), done)
        observation, global_step, episode_reward = next_observation, global_step + 1, episode_reward + reward

        if global_step >= args.learning_starts and len(replay_buffer) >= args.batch_size and global_step % args.train_freq == 0:
            for _ in range(args.gradient_steps):
                latest_update_metrics = agent.update(replay_buffer.sample(args.batch_size))
                update_step += 1
                for k, v in latest_update_metrics.items():
                    if k in history: history[k].append(v)

        if done:
            episode_idx += 1
            reward_window.append(episode_reward)
            history["reward_history"].append(episode_reward)
            best_reward = max(best_reward, episode_reward)

            if latest_update_metrics:
                avg_rew = np.mean(reward_window)
                print(f"Ep {episode_idx} | Avg100: {avg_rew:.2f} | Rew: {episode_reward:.1f} | "
                      f"Losses: [A: {latest_update_metrics['actor_loss']:.3f}, C: {latest_update_metrics['critic_loss']:.3f}, α: {latest_update_metrics['alpha_loss']:.3f}]")
            
            observation, _ = env.reset()
            episode_reward = 0.0

        if episode_idx % args.checkpoint_interval == 0 and episode_idx > 0 and episode_idx != last_saved_episode:
            ckpt_path = os.path.join(args.save_dir, f"sac_epoch_{episode_idx}.pt")
            last_saved_episode = episode_idx
            torch.save({
                "actor_state": agent.actor.state_dict(), "critic_state": agent.critic.state_dict(),
                "critic_target_state": agent.critic_target.state_dict(), "actor_opt_state": agent.actor_opt.state_dict(),
                "critic_opt_state": agent.critic_opt.state_dict(), "alpha_opt_state": agent.alpha_opt.state_dict(),
                "log_alpha": agent.log_alpha, "history": history, "global_step": global_step,
                "episode_idx": episode_idx, "update_step": update_step, "best_reward": best_reward
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--learning_starts", type=int, default=10000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--save_dir", type=str, default="results_sac")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    train_sac(args)