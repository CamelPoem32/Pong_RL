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
# 1. Coordinate Environment (same as your TRPO)
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

@torch.compile
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
        # self.value_head = nn.Linear(512, 1)
        self.value_head = layer_init(nn.Linear(512, 1))

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value.squeeze(-1)

# ============================================================
# 3. Run episode
# ============================================================

def run_episode(env, policy, device):

    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    state, _ = env.reset()
    done = False

    while not done:

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(device)

        logits, value = policy(state_tensor)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state_tensor.squeeze(0))
        actions.append(action.squeeze(0))
        rewards.append(reward)
        dones.append(done)
        values.append(value.squeeze(0).detach())
        log_probs.append(log_prob.squeeze(0).detach())

        state = next_state

    return states, actions, log_probs, rewards, dones, values

# ============================================================
# 4. GAE Computation
# ============================================================

def compute_gae(rewards, values, dones, nextvalue, nextdone, gamma=0.99, lam=0.95):

    advantages = []
    gae = 0

    values = values + [nextvalue]
    dones = dones + [nextdone]

    for t in reversed(range(len(rewards))):

        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t+1]) - values[t]         # TD residual (temporal difference residual)
        gae = delta + gamma * lam * (1 - dones[t+1]) * gae

        advantages.insert(0, gae)
        # print(f"reward {rewards[t]} delta {float(delta):.4f} advantage {gae} value {values[t]} value_t+1 {values[t+1]} nextdone {dones[t+1]}")

    advantages = torch.stack(advantages).squeeze(-1)
    returns = advantages + torch.stack(values[:-1])

    return advantages.detach(), returns.detach()

# ============================================================
# 5. PPO Update
# ============================================================

# def ppo_update(
#     policy,
#     optimizer,
#     states,
#     actions,
#     old_log_probs,
#     advantages,
#     returns,
#     clip_eps=0.2,
#     value_coef=0.5,
#     entropy_coef=0.01,
#     ppo_epochs=4,
#     batch_size=64,
# ):

#     dataset_size = states.size(0)
#     n_updates = 0
#     total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0

#     for _ in range(ppo_epochs):

#         indices = torch.randperm(dataset_size)

#         for start in range(0, dataset_size, batch_size):

#             end = start + batch_size
#             batch_idx = indices[start:end]

#             batch_states = states[batch_idx]
#             batch_actions = actions[batch_idx]
#             batch_old_log_probs = old_log_probs[batch_idx]
#             batch_adv = advantages[batch_idx]
#             batch_returns = returns[batch_idx]

#             if torch.isnan(batch_adv.std()) or batch_adv.std() <= 0:
#                 batch_adv_norm = (batch_adv - batch_adv.mean())
#             else:
#                 batch_adv_norm = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

#             logits, values = policy(batch_states)
#             dist = Categorical(logits=logits)

#             new_log_probs = dist.log_prob(batch_actions)
#             entropy_loss = dist.entropy().mean()

#             ratio = torch.exp(new_log_probs - batch_old_log_probs)
#             # print(f"new_log_probs = {new_log_probs.mean()} +- {new_log_probs.std()}, batch_old_log_probs = {batch_old_log_probs.mean()} +- {batch_old_log_probs.std()}")

#             surr1 = ratio * batch_adv_norm
#             surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_adv_norm

#             policy_loss = -torch.min(surr1, surr2).mean()
#             value_loss = (batch_returns - values).pow(2).mean()
#             # print(f"ratio = {ratio.mean()} +- {ratio.std()}, advs = {batch_adv_norm.mean()} +- {batch_adv_norm.std()}")
#             # print(f"policy loss = {policy_loss}, surr1 = {surr1.mean()}, surr2 = {surr2.mean()}")
            
#             loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

#             # if _ < 2 and start == 0:

#             #     print(f"action {batch_actions[0]}, log_prob {batch_old_log_probs[0]}, value {batch_returns[0]}")
#             #     print(f"New action {batch_actions[0]}, log_prob {new_log_probs[0]}, value {values[0]}")
#             #     print(f"ratio {torch.exp(new_log_probs[0] - batch_old_log_probs[0])}")
#             #     print(f"ratios {torch.exp(new_log_probs - batch_old_log_probs)}")
#             #     print(f"policy_loss {policy_loss}, value_loss {value_loss}, entropy_loss {entropy_loss}")

#             # print("param before:", next(policy.parameters())[-1,-1].item())

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
#             optimizer.step()

#             # print("param after :", next(policy.parameters())[-1,-1].item())

#             total_policy_loss += policy_loss.item()
#             total_value_loss += value_loss.item()
#             total_entropy += entropy_loss.mean().item()
#             n_updates += 1

#     return {
#     "policy_loss": total_policy_loss / max(n_updates, 1),
#     "value_loss": total_value_loss / max(n_updates, 1),
#     "entropy": total_entropy / max(n_updates, 1),
#     }

def ppo_update(
    policy,
    optimizer,
    states,
    actions,
    old_log_probs,
    advantages,
    returns,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    ppo_epochs=4,
    batch_size=64,
):
    device = next(policy.parameters()).device
    dataset_size = states.size(0)
    n_updates = 0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    # Ensure types
    actions = actions.long().to(device)
    old_log_probs = old_log_probs.float().to(device)
    advantages = advantages.float().to(device)
    returns = returns.float().to(device)
    states = states.to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # Quick sanity prints
    with torch.no_grad():
        logits_check, _ = policy(states[:min(8, len(states))])
        dist_check = Categorical(logits=logits_check)
        new_lp_sample = dist_check.log_prob(actions[:min(8, len(actions))])
        # print("sanity: sample old_log_probs[0:5] =", old_log_probs[:5].cpu().numpy())
        # print("sanity: sample new_logprob(before update) [0:5] =", new_lp_sample[:5].cpu().numpy())
        # print("sanity: advantages mean/std =", float(advantages.mean().cpu()), float(advantages.std().cpu()))

    for epoch in range(ppo_epochs):
        indices = torch.randperm(dataset_size, device=device)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_lps = old_log_probs[mb_idx]
            mb_advs = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            mb_advs_norm = mb_advs
            # normalize advantages per minibatch
            # adv_mean = mb_advs.mean()
            # adv_std = mb_advs.std(unbiased=False)
            # if torch.isnan(adv_std) or adv_std <= 0:
            #     mb_advs_norm = mb_advs - adv_mean
            # else:
            #     mb_advs_norm = (mb_advs - adv_mean) / (adv_std + 1e-8)

            logits, values = policy(mb_states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # numerically stable ratio
            ratio = torch.exp(new_log_probs - mb_old_lps)
            # print(f"ratio {ratio.shape}")
            # print(f"mb_advs_norm {mb_advs_norm.shape}")

            surr1 = ratio * mb_advs_norm
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advs_norm
            # print(f"surr1 {surr1.shape}")
            # print(f"surr2 {surr2.shape}")
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_returns - values).pow(2).mean()
            entropy_loss = -entropy  # negative of mean entropy

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            # --- Diagnostic: log change in log_probs for this minibatch BEFORE step ---
            # if epoch < 2 and start == 0:
            #     with torch.no_grad():
            #         mean_abs_diff = (new_log_probs - mb_old_lps).abs().mean().item()
            #         print(f"DBG before step: mean(|new_log - old_log|) = {mean_abs_diff:.6e}")
            #         print(f"DBG ratios mean/std = {ratio.mean().item():.6e} / {ratio.std().item():.6e}")
            #         print(f"policy_loss {policy_loss:.4e} value_loss {value_loss:.4e} entropy_loss {entropy_loss:.4e}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            # for p in policy.parameters():
            #     if p.grad is not None:
            #         # print(p.grad.data)
            #         p.grad.data.mul_(100.0)
            optimizer.step()

            # --- Diagnostic: recompute new_log_probs AFTER step to see change ---
            # if epoch < 2 and start == 0:
            #     with torch.no_grad():
            #         logits_after, _ = policy(mb_states)
            #         new_log_probs_after = Categorical(logits=logits_after).log_prob(mb_actions)
            #         mean_abs_diff_after = (new_log_probs_after - mb_old_lps).abs().mean().item()
            #         print(f"DBG after step: mean(|new_log - old_log|) = {mean_abs_diff_after:.6e}")
            #         print(f"DBG logprob change due to step = {((new_log_probs_after - new_log_probs).abs().mean().item()):.6e}")

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += (-entropy).item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }

# ============================================================
# 6. Training Loop
# ============================================================

def train_ppo(
    n_epochs=5000,
    episodes_per_epoch=10,
    gamma=0.99,
    lam=0.95,
    lr=2.5e-4,
    save_dir="results_ppo",
    log_interval=100,
    device="",
    checkpoint_path=None,
    checkpoint_interval=100,
):

    if device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)

    env = create_pong_env()
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PPOPolicy(input_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_history = []
    policy_loss_history = []
    value_loss_history = []
    entropy_history = []
    time_history = []

    # ============================================================
    # LOAD CHECKPOINT IF PROVIDED
    # ============================================================
    start_epoch = 1

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        policy.load_state_dict(checkpoint["policy_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        reward_history = checkpoint["reward_history"]
        policy_loss_history = checkpoint["policy_loss_history"]
        value_loss_history = checkpoint["value_loss_history"]
        entropy_history = checkpoint["entropy_history"]
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
        all_dones = []

        t_start_episodes = time()
        for _ in range(episodes_per_epoch):

            states, actions, log_probs, rewards, dones, values = \
                run_episode(env, policy, device)

            total_reward = sum(rewards)
            epoch_rewards.append(total_reward)
            nextstate, _ = env.reset()
            nextstate_tensor = torch.tensor(nextstate, dtype=torch.float32).unsqueeze(0).to(device)
            _, nextvalue = policy(nextstate_tensor)
            nextdone = dones[-1]

            advantages, returns = compute_gae(
                rewards,
                values,
                dones,
                nextvalue,
                nextdone,
                gamma,
                lam
            )

            all_states.extend(states)
            all_actions.extend(actions)
            all_log_probs.extend(log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_dones.extend(dones)

        # Stack all collected data
        states_tensor = torch.stack(all_states).to(device)
        actions_tensor = torch.stack(all_actions).to(device)
        old_log_probs_tensor = torch.stack(all_log_probs).to(device)
        advantages_tensor = torch.stack(all_advantages).to(device)
        returns_tensor = torch.stack(all_returns).float().to(device).squeeze(-1)
        dones_tensor = torch.tensor(all_dones, dtype=torch.int64).to(device)
        # print(f"returns {epoch}")
        # print(returns_tensor[:1024].reshape(-1, 16)[:5])
        # print(returns_tensor[:1024].reshape(-1, 16)[5:10])
        # print(f"advantages {epoch}")
        # print(advantages_tensor[:1024].reshape(-1, 16)[:5])
        # print(advantages_tensor[:1024].reshape(-1, 16)[5:10])
        # print(f"dones {epoch}")
        # print(dones_tensor[:1024].reshape(-1, 16)[:5])
        # print(dones_tensor[:1024].reshape(-1, 16)[5:10])
        

        # PPO update step
        info = ppo_update(
            policy,
            optimizer,
            states_tensor,
            actions_tensor,
            old_log_probs_tensor,
            advantages_tensor,
            returns_tensor,
        )

        avg_reward = np.mean(epoch_rewards)
        reward_history.append(avg_reward)
        policy_loss_history.append(info["policy_loss"])
        value_loss_history.append(info["value_loss"])
        entropy_history.append(info["entropy"])
        time_history.append(time() - t_start_episodes)

        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch}/{n_epochs+start_epoch-1} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Policy loss: {info['policy_loss']:.4f} | "
                f"Value loss: {info['value_loss']:.4f} | "
                f"Time: {time() - t0:.2f}s | "
                f"TPE: {(time() - t_start_epochs)/log_interval:.2f}s"
            )
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
                "policy_loss_history": policy_loss_history,
                "value_loss_history": value_loss_history,
                "entropy_history": entropy_history,
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
        "policy_loss_history": policy_loss_history,
        "value_loss_history": value_loss_history,
        "entropy_history": entropy_history,
        "time_history": time_history,
    }

    # Save policy
    policy_path = os.path.join(save_dir, f"policy_ppo_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(policy.state_dict(), policy_path)

    rewards_path = os.path.join(save_dir, f"reward_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(reward_history, rewards_path)
    policy_loss_path = os.path.join(save_dir, f"policy_loss_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(policy_loss_history, policy_loss_path)
    value_loss_path = os.path.join(save_dir, f"value_loss_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(value_loss_history, value_loss_path)
    entropy_path = os.path.join(save_dir, f"entropy_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(entropy_history, entropy_path)
    time_path = os.path.join(save_dir, f"time_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(time_history, time_path)

    print("Training finished.")
    print(f"Policy saved to: {policy_path}")
    print(f"Reward history saved to: {rewards_path}")

    env.close()

    return policy, reward_history

# ============================================================
# 7. Run
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO vs REINFORCE on Atari Pong")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes")
    parser.add_argument("--epochs", type=int, default=1000, help="Epochs")
    parser.add_argument("--checkpoint_episodes", type=int, default=100, help="Checkpoint Episodes")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint Path")
    parser.add_argument("--log_interval", type=int, default=1, help="Log Interval")
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--lam", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    train_ppo(
        n_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.lr,
        save_dir="results_ppo",
        log_interval=args.log_interval,
        device=args.device,
        checkpoint_interval=args.checkpoint_episodes,
        checkpoint_path=args.checkpoint_path,                   # "results_ppo/checkpoint_700epoch_4ep.pt"
    )