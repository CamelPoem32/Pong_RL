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

# @torch.compile
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


def conjugate_gradient(Avp, b, nsteps=10, residual_tol=1e-10):       # 1e-10
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(nsteps):
        Avp_p = Avp(p)
        alpha = rdotr / (torch.dot(p, Avp_p) + 1e-8)
        x += alpha * p
        r -= alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        # print(f"new rdotr = {new_rdotr}")
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        # print(f"beta = {beta}")
        p = r + beta * p
        rdotr = new_rdotr
    return x

def flat_grad(grads):
    return torch.cat([g.reshape(-1) for g in grads])

def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, new_flat_params):
    index = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(new_flat_params[index:index+numel].view(p.size()))
        index += numel

def fisher_vector_product(policy, states, old_logits, vector, damping=1e-2):
    new_logits = torch.cat([policy(s) for s in states])

    old_dist = Categorical(logits=old_logits.detach())
    new_dist = Categorical(logits=new_logits)
    t0 = time()

    kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
    # print(f"t kl = {time() - t0}")
    t0 = time()
    # probs = torch.softmax(logits, dim=-1)
    # # KL divergence with itself (detached)
    # kl = torch.mean(
    #     torch.sum(
    #         probs * (torch.log(probs + 1e-8) - torch.log(probs.detach() + 1e-8)),
    #         dim=1
    #     )
    # )

    grads = torch.autograd.grad(kl, policy.parameters(), 
                                create_graph=True,
                                )
    flat_grad_kl = flat_grad(grads)
    # print(f"t grad = {time() - t0}")
    t0 = time()

    kl_v = (flat_grad_kl * vector).sum()

    grads2 = torch.autograd.grad(kl_v, policy.parameters())
    flat_grad2 = flat_grad(grads2)
    # print(f"t grad2 = {time() - t0}")

    return flat_grad2 + damping * vector

# ============================================================
# 4. NPG Update
# ============================================================

def natural_policy_gradient_update(policy, optimizer, states_batch, all_log_probs, all_returns, epsilon=1e-2):

    loss = 0
    t0 = time()
    # for log_probs, returns in zip(log_probs_batch, returns_batch):
    #     loss_i = 0
    #     for log_prob, return_val in zip(log_probs, returns):
    #         loss_i += -log_prob * return_val

    #     loss += loss_i / len(log_probs_batch)

    # all_log_probs = torch.cat([torch.stack(lp) for lp in log_probs_batch])
    # all_returns = torch.cat(returns_batch).to(all_log_probs.device)

    # all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

    loss = (all_log_probs * all_returns.detach()).mean()            # NOT MINUS LOSS!!!!!!!!!!!!!!!!!!!
    # Because here we update manually, so grad in formulas is grad of this loss, and we do ascend by theta + alpha*step_dir
    # Torch Adam does step as descend: theta - alpha*step_dir, so there we set loss as negative
    # print(f"t loss = {time() - t0}")
    t0 = time()

    grads = torch.autograd.grad(loss, policy.parameters())
    g = flat_grad(grads).detach()
    # print(f"loss = {loss.item()}")

    def Avp(v):
        return fisher_vector_product(policy, states_batch, old_logits, v)

    with torch.no_grad():
        old_logits = torch.cat([policy(s) for s in states_batch]).detach()

    step_direction = conjugate_gradient(Avp, g)
    # print(f"t stepdir = {time() - t0}")
    t0 = time()
    shs = 0.5 * torch.dot(step_direction, Avp(step_direction))
    step_size = torch.sqrt(2 * epsilon / (shs + 1e-8))

    old_params = flat_params(policy)
    new_params = old_params + step_size * step_direction
    # print(step_size, step_direction)
    # print(step_size * step_direction)
    set_params(policy, new_params)
    # print(f"t setparams = {time() - t0}")

    return loss.item(), step_size


# ============================================================
# 5. Training Loop
# ============================================================

def train_npg(
    n_epochs=5000,
    episodes_per_epoch=10,
    epsilon=1e-2,
    gamma=1,
    lr=2.5e-4,
    save_dir="results_npg",
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
    step_size_history = []
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
        step_size_history = checkpoint["step_size_history"]
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
        # print(f"t episodes = {time() - t_start_episodes}")

        t_start_update = time()
        states_tensor = torch.stack(all_states).to(device)
        actions_tensor = torch.stack(all_actions).to(device)
        log_probs_tensor = torch.stack(all_log_probs).to(device)
        returns_tensor = torch.tensor(all_returns).detach().to(device)
        # print(f"states_tensor {states_tensor.shape} log_probs_tensor {log_probs_tensor.shape} returns_tensor {returns_tensor.shape}")

        loss, step_size = natural_policy_gradient_update(policy, optimizer, states_tensor, log_probs_tensor, returns_tensor, epsilon=epsilon)
        # print(f"t update = {time() - t_start_update}")

        avg_reward = np.mean(epoch_rewards)
        reward_history.append(avg_reward)
        time_history.append(time() - t_start_episodes)
        loss_history.append(loss)

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}/{n_epochs+start_epoch-1} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Loss: {loss:.4f} | "
                  f"Step size: {step_size:.4f} | "
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
                "step_size_history": step_size_history,
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
        "step_size_history": step_size_history,
    }

    torch.save(
        checkpoint,
        os.path.join(save_dir, f"checkpoint_{epoch}epoch_{episodes_per_epoch}ep.pt")
    )

    # Save policy
    policy_path = os.path.join(save_dir, f"policy_npg_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(policy.state_dict(), policy_path)

    # Save rewards
    rewards_path = os.path.join(save_dir, f"reward_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(reward_history, rewards_path)
    loss_path = os.path.join(save_dir, f"loss_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(loss_history, loss_path)
    step_size_path = os.path.join(save_dir, f"step_size_history_{epoch}epoch_{episodes_per_epoch}ep.pt")
    torch.save(step_size_history, step_size_path)
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
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    train_npg(
        n_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        epsilon=args.epsilon,
        save_dir="results_npg",
        log_interval=args.log_interval,
        device=args.device,
        checkpoint_interval=args.checkpoint_episodes,
        checkpoint_path=args.checkpoint_path,                   # "results_npg/checkpoint_100epoch_5ep.pt"
    )