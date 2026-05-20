import os
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import ale_py
import argparse

gym.register_envs(ale_py)

# ============================================================
# 1. Simplified Coordinate Environment
# ============================================================

class PongCoordinateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.history_length = 4
        self.coord_history = []

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.history_length * 3,),
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

        self.coord_history.pop(0)
        self.coord_history.append(coords)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(self.coord_history).astype(np.float32)

    def extract_coords(self, frame):

        y_min = 34
        y_max = 193

        paddle_mask = (
            (frame[y_min:y_max, :, 0] < 100) &
            (frame[y_min:y_max, :, 1] > 150) &
            (frame[y_min:y_max, :, 2] < 100)
        )

        paddle_indices = np.argwhere(paddle_mask)
        if len(paddle_indices) > 0:
            paddle_y = paddle_indices[:, 0].mean() + y_min
        else:
            paddle_y = 0.0

        ball_mask = (
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
# 2. Actor-Critic Network
# ============================================================
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

# @torch.compile
class PPOPolicy(nn.Module):
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

def trpo_kl(policy, old_policy, states):

    new_logits = policy(states)
    old_logits = old_policy(states).detach()

    new_dist = Categorical(logits=new_logits)
    old_dist = Categorical(logits=old_logits)

    kl = torch.distributions.kl_divergence(old_dist, new_dist)

    return kl.mean()

def flat_grad(grads):
    return torch.cat([g.reshape(-1) for g in grads])

def fisher_vector_product(policy, old_policy, states, v, damping=1e-2):

    kl = trpo_kl(policy, old_policy, states)

    grads = torch.autograd.grad(
        kl,
        policy.parameters(),
        create_graph=True
    )

    flat_grad_kl = flat_grad(grads)

    kl_v = (flat_grad_kl * v).sum()

    grads2 = torch.autograd.grad(
        kl_v,
        policy.parameters()
    )

    flat_grad2 = flat_grad(grads2)

    return flat_grad2 + damping * v

def conjugate_gradient(Avp, b, n_steps=10):

    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rsold = torch.dot(r, r)

    for _ in range(n_steps):

        Ap = Avp(p)
        denom = torch.dot(p, Ap)
        if denom.abs() < 1e-10:
            break
        alpha = rsold / denom

        x += alpha * p
        r -= alpha * Ap

        rsnew = torch.dot(r, r)

        if rsnew < 1e-10:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

def trpo_surrogate_objective(policy, states, actions,
                        old_log_probs, advantages):

    logits = policy(states)
    dist = torch.distributions.Categorical(logits=logits)

    log_probs = dist.log_prob(actions)

    ratio = torch.exp(log_probs - old_log_probs)
    # print(f"log_probs {log_probs.shape} old_log_probs = {old_log_probs.shape}, ratio {ratio.shape} advantages {advantages.shape}")

    return (ratio * advantages).mean()

def set_params(model, new_flat_params):
    index = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(new_flat_params[index:index+numel].view(p.size()))
        index += numel

def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

# ============================================================
# 4. TRPO Update
# ============================================================

def trpo_update(policy, old_policy, states, actions,
                old_log_probs, advantages,
                max_kl=1e-2, v=False):

    # Compute surrogate objective
    objective = trpo_surrogate_objective(
        policy, states, actions,
        old_log_probs, advantages
    )
    objective_value = objective.item()

    grads = torch.autograd.grad(objective, policy.parameters())
    objective_grad = flat_grad(grads).detach()
    objective_grad = torch.clamp(objective_grad, -10, 10)             # Prevents rare spikes causing NANs
    if objective_grad.norm() < 1e-8:
        return objective_value

    # Cache old logits ONCE
    with torch.no_grad():
        old_logits = policy(states).detach()

    def Avp(v):
        return fisher_vector_product(policy, old_policy, states, v)

    step_dir = conjugate_gradient(Avp, objective_grad)

    shs = 0.5 * (step_dir * Avp(step_dir)).sum()
    if shs <= 0:
        return objective_value
    step_size = torch.sqrt(max_kl / (shs + 1e-8))

    full_step = step_dir * step_size

    old_params = flat_params(policy)

    new_objective = objective_value
    for step_fraction in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        # print(f"step_fraction = {step_fraction:.4f}")
        new_params = old_params + step_fraction * full_step
        set_params(policy, new_params)

        logits_new = policy(states).detach()
        if torch.isnan(logits_new).any():
            if v:
                print(f"NaN logits at step fraction {step_fraction:.4f}, step_dir {step_dir.norm().item():.6f}, step_size {step_size}",
                    f"objective {objective_value:.6f} objective_grad {objective_grad.norm().item():.6f} shs {shs}")
            continue
        new_objective = trpo_surrogate_objective(policy, states, actions, old_log_probs, advantages).item()
        kl = trpo_kl(policy, old_policy, states)

        if kl <= max_kl and new_objective > objective_value:
            if v:
                print(f"Accepted step at fraction {step_fraction:.4f} with KL {kl:.6f} and objective improvement {new_objective - objective_value:.6f}")
            del logits_new, kl, new_params, old_params
            break

    else:
        # no acceptable step found
        set_params(policy, old_params)
        del old_params

    old_policy.load_state_dict(policy.state_dict())

    return new_objective - objective_value

# ============================================================
# 5. Training Loop
# ============================================================

def train_trpo(
    n_epochs=5000,
    episodes_per_epoch=10,
    gamma=1.0,
    lr=2.5e-4,
    save_dir="results_trpo",
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
    policy = PPOPolicy(input_dim, n_actions).to(device)
    old_policy = PPOPolicy(input_dim, n_actions).to(device)
    old_policy.load_state_dict(policy.state_dict())
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
        all_advantages = []
        all_returns = []

        t_start_episodes = time()
        for _ in range(episodes_per_epoch):

            states, actions, log_probs, rewards, dones = run_episode(env, policy, device) #, v=(_==0)

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
        # print(f"t episodes = {time() - t_start_episodes}")

        t_start_update = time()

        states_batch = torch.stack(all_states).detach()
        actions_batch = torch.stack(all_actions).squeeze().detach()
        old_log_probs_batch = torch.stack(all_log_probs).detach()
        returns_batch = torch.tensor(all_returns).detach().to(device)

        # normalize advantages (no critic yet)
        advantages = (returns_batch - returns_batch.mean())
        # print(states_batch.shape)
        # print(actions_batch.shape)
        # print(old_log_probs_batch.shape)
        # print(returns_batch.shape)
        # print(advantages.shape)


        loss = trpo_update(policy, old_policy, states_batch, actions_batch, old_log_probs_batch, advantages, max_kl=1e-2)      # , v=(epoch % log_interval == 0)
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
                "loss_history": loss_history,
                "time_history": time_history,
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
        "loss_history": loss_history,
        "time_history": time_history,
    }

    torch.save(
        checkpoint,
        os.path.join(save_dir, f"checkpoint_{epoch}epoch_{episodes_per_epoch}ep.pt")
    )

    # Save policy
    policy_path = os.path.join(save_dir, f"policy_trpo_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(policy.state_dict(), policy_path)

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

    train_trpo(
        n_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        save_dir="results_trpo",
        log_interval=args.log_interval,
        device=args.device,
        checkpoint_interval=args.checkpoint_episodes,
        checkpoint_path=args.checkpoint_path,                   # "results_ppo/checkpoint_700epoch_4ep.pt"
    )