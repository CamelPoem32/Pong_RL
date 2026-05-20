import os
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation as FrameStack
import ale_py
import argparse

gym.register_envs(ale_py)

# ============================================================
# 1. Simplified Coordinate Environment
# ============================================================
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class PongCoordinateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.history_length = 4
        self.coord_history = []

        # observation = [paddle_y, ball_x, ball_y] * 4
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.history_length * 4,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        coords = self.extract_coords(obs)

        self.coord_history = [coords] * self.history_length
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        coords = self.extract_coords(obs)
        # print(coords*256)

        self.coord_history.pop(0)
        self.coord_history.append(coords)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(self.coord_history).astype(np.float32)

    def extract_coords(self, frame):
        """
        Extract paddle Y and ball X,Y from RGB frame.
        Normalize to [0,1].
        """
        # Field borders
        y_min = 34
        y_max = 193

        # Paddle (left side, greenish)
        paddle_mask = (
            (frame[y_min:y_max, :, 0] < 100) &
            (frame[y_min:y_max, :, 1] > 150) &
            (frame[y_min:y_max, :, 2] < 100)
        )

        paddle_indices = np.argwhere(paddle_mask)
        if len(paddle_indices) > 0:
            paddle_y = paddle_indices[:, 0].mean() + y_min
            paddle_x = paddle_indices[:, 1].mean()
        else:
            paddle_y = 0.0
            paddle_x = 0.0

        # Ball (white)
        ball_mask = (   
            # white mask
            (frame[y_min:y_max, :, 0] > 230) &
            (frame[y_min:y_max, :, 1] > 230) &
            (frame[y_min:y_max, :, 2] > 230) 
        )

        ball_indices = np.argwhere(ball_mask)
        if len(ball_indices) > 0:
            ball_y = ball_indices[:, 0].mean() + y_min
            ball_x = ball_indices[:, 1].mean()
        else:
            ball_x, ball_y = 0.0, 0.0

        h, w, _ = frame.shape

        return np.array([
            paddle_x / w,
            paddle_y / h,
            ball_x / w,
            ball_y / h
        ])

def create_pong_env(env_name="PongNoFrameskip-v4", frame_skip=4):
    env = gym.make(env_name, render_mode=None,
                   frameskip=frame_skip,
                   repeat_action_probability=0.0)
    env = PongCoordinateWrapper(env)
    return env

# ============================================================
# 2. MLP Policy
# ============================================================

@torch.compile
class MLPPolicy(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()

        self.shared = nn.Sequential(
            # nn.Linear(input_dim, 512),
            layer_init(nn.Linear(input_dim, 512)),
            nn.ReLU(),
            # nn.Linear(128, 512),
            # nn.ReLU(),
            # nn.Linear(512, 128),
            # nn.ReLU(),
        )

        # self.policy_head = nn.Linear(512, n_actions)
        self.policy_head = layer_init(nn.Linear(512, n_actions), std=0.01)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        return logits


# ============================================================
# 3. Episode rollout
# ============================================================

def run_episode(env, policy, device):

    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []

    state, _ = env.reset()
    done = False

    while not done:

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(device)

        logits = policy(state_tensor)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.squeeze(0))

        state = next_state

    return states, actions, log_probs, rewards, dones


# ============================================================
# 4. REINFORCE Update
# ============================================================

def reinforce_update(optimizer, log_probs, returns):
    """
    Performs REINFORCE update using episode trajectory.
    """

    # Since gamma = 1 and reward only at end,
    # total return is simply sum(rewards)
    loss = 0
    for log_prob, Gt in zip(log_probs, returns):
        loss += -log_prob * Gt

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================================================
# 5. Training Loop
# ============================================================

def train_reinforce(
    n_epochs=5000,
    episodes_per_epoch=10,
    gamma=1.0,
    lr=2.5e-4,
    save_dir="results_reinforce",
    log_interval=1000,
    device="",
    checkpoint_path=None,
    checkpoint_interval=100,
):
    if device == "": device="cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device)

    env = create_pong_env()
    n_actions = env.action_space.n

    input_dim = env.observation_space.shape[0]
    policy = MLPPolicy(input_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_history = []
    loss_history = []
    time_history = []
    start_epoch = 1

    # ============================================================
    # LOAD CHECKPOINT IF PROVIDED
    # ============================================================

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        policy.load_state_dict(checkpoint["policy_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        reward_history = checkpoint["reward_history"]
        loss_history = checkpoint["loss_history"]
        time_history = checkpoint["time_history"]
        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")

    # ============================================================

    t0 = time()
    t_start_epochs = time()
    for epoch in range(start_epoch, n_epochs + start_epoch):

        epoch_rewards = []

        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []

        t_start_episodes = time()
        for _ in range(episodes_per_epoch):

            states, actions, log_probs, rewards, dones = run_episode(env, policy, device)

            total_reward = sum(rewards)

            epoch_rewards.append(total_reward)

            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            all_states.extend(states)
            all_actions.extend(actions)
            all_log_probs.extend(log_probs)
            all_returns.extend(returns)

        t_start_update = time()

        states_tensor = torch.stack(all_states).to(device)
        actions_tensor = torch.stack(all_actions).to(device)
        log_probs_tensor = torch.stack(all_log_probs).to(device)
        returns_tensor = torch.tensor(all_returns).detach().to(device)

        # print(f"returns_batch = {returns_batch}")
        loss = reinforce_update(optimizer, log_probs_tensor, returns_tensor)
        avg_reward = np.mean(epoch_rewards)
        reward_history.append(avg_reward)
        loss_history.append(loss)
        time_history.append(time() - t_start_episodes)
        # print(f"t update = {time() - t_start_update}")

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}/{n_epochs+start_epoch-1} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Loss: {loss:.4f} | "
                  f"Time: {time() - t0:.2f}s | "
                  f"TPE: {(time() - t_start_epochs) / log_interval:.2f}s")
            t_start_epochs = time()

        # ========================================================
        # SAVE CHECKPOINT in runtime
        # ========================================================
        if epoch % checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "policy_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "reward_history": reward_history,
                "time_history": time_history,
                "loss_history": loss_history,
            }

            torch.save(
                checkpoint,
                os.path.join(save_dir, f"checkpoint_{epoch}epoch_{episodes_per_epoch}ep.pt")
            )

    # ========================================================
    # SAVE CHECKPOINT 
    # ========================================================
    checkpoint = {
        "epoch": epoch,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "reward_history": reward_history,
        "time_history": time_history,
        "loss_history": loss_history,
    }

    torch.save(
        checkpoint,
        os.path.join(save_dir, f"checkpoint_{epoch}epoch_{episodes_per_epoch}ep.pt")
    )

    # Save policy
    policy_path = os.path.join(save_dir, f"policy_reinforce_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(policy.state_dict(), policy_path)

    # Save rewards
    rewards_path = os.path.join(save_dir, f"reward_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(reward_history, rewards_path)
    loss_path = os.path.join(save_dir, f"loss_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(loss_history, loss_path)
    time_path = os.path.join(save_dir, f"time_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(time_history, time_path)

    print("Training finished.")
    print(f"Policy saved to: {policy_path}")
    print(f"Reward history saved to: {rewards_path}")

    env.close()

    return policy, reward_history


# ============================================================
# 6. Run Training
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO vs REINFORCE on Atari Pong")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs")
    parser.add_argument("--checkpoint_episodes", type=int, default=100, help="Checkpoint Episodes")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint Path")
    parser.add_argument("--log_interval", type=int, default=1, help="Log Interval")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    train_reinforce(
        n_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        save_dir="results_reinforce",
        log_interval=args.log_interval,
        device=args.device,
        checkpoint_interval=args.checkpoint_episodes,
        checkpoint_path=args.checkpoint_path,                   # "results_reinforce/checkpoint_700epoch_4ep.pt"
    )