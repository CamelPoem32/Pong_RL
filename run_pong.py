

# pong_runner.py
import time
import numpy as np
import torch
import torch.nn as nn
import pygame
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation as FrameStack
from torch.distributions import Categorical
import ale_py
import matplotlib.pyplot as plt
import argparse

gym.register_envs(ale_py)

# ---------------------------
# 1) environment creation
# ---------------------------

def create_pong_env(env_name="PongNoFrameskip-v4", sticky_actions=False, render_mode="rgb_array"):
    """
    Create an Atari Pong environment, preprocessed to:
      - grayscale
      - resized to 84x84
      - frame-skip (default 4 inside AtariPreprocessing)
      - frame-stack of the last 4 frames

    Returns:
      env: wrapped env (observation shape: (4, 84, 84) as numpy array)
      action_meanings: list of action meaning strings for this ROM
    """
    # Make env with rgb_array render (we'll display frames with pygame)
    # env = gym.make(env_name, render_mode=render_mode)  # PongNoFrameskip-v4

    # # AtariPreprocessing does: grayscale, downsample to 84x84, frame_skip, terminal_on_life_loss (opt)
    # env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, screen_size=84)
    # # stack 4 frames
    # env = FrameStack(env, stack_size=4)

    env = gym.make(env_name, render_mode=render_mode)
    env = PongCoordinateWrapper(env)

    # Get action meanings from underlying unwrapped env to explain the 6 actions
    try:
        meanings = env.unwrapped.get_action_meanings()
    except Exception:
        # fallback
        meanings = [str(i) for i in range(env.action_space.n)]
    print("Environment created:", env_name)
    print("Action space size:", env.action_space.n)
    print("Action meanings:", meanings)
    return env, meanings

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

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
        # PPO
        # self.value_head = layer_init(nn.Linear(512, 1))

    # PPO
    # def forward(self, x):
    #     h = self.shared(x)
    #     logits = self.policy_head(h)
    #     value = self.value_head(h)
    #     return logits, value.squeeze(-1)

    # All but PPO
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        return logits


# ---------------------------
# 2) policy helper
# ---------------------------

def get_action_from_policy(policy, state, deterministic=False, device=None):
    """
    state: numpy array shape (stack, H, W) or torch tensor (1, stack, H, W)
    policy: a torch.nn.Module that returns logits or probs for discrete action space:
            - expected to return action logits (unnormalized) of shape (batch, n_actions)
            - or return a dict: {"logits": logits, "value": value} for actor-critic style
    deterministic: if True -> argmax, else sample
    returns: action (int), log_prob (torch scalar), info(dict)
    """
    if device is None:
        device = next(policy.parameters()).device if list(policy.parameters()) else torch.device("cpu")

    # ensure torch tensor batched
    if isinstance(state, np.ndarray):
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)  # (1, C, H, W)
    else:
        st = state.float().unsqueeze(0).to(device)

    # All but PPO
    policy_out = policy(st)
    # PPO
    # policy_out, value = policy(st)

    # policy may return logits directly or a dict
    if isinstance(policy_out, dict):
        logits = policy_out["logits"]
    else:
        logits = policy_out

    logits = logits.squeeze(0)   # shape (n_actions,)
    probs = torch.softmax(logits, dim=-1)
    dist = Categorical(probs=probs)

    if deterministic:
        action = int(torch.argmax(probs).item())
        log_prob = torch.log(probs[action] + 1e-8)
    else:
        action = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action, device=probs.device))

    return action, log_prob, {"probs": probs.detach().cpu().numpy()}


# ---------------------------
# 3) simple conv policy skeleton
# ---------------------------

class PongCoordinateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.history_length = 4
        self.coord_history = []

        # observation = [paddle_y, ball_x, ball_y] * 4
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
        else:
            paddle_y = 0.0

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
        # print(paddle_indices)
        # print(ball_indices)
        # print(paddle_y)
        # print(ball_x, ball_y)

        return np.array([
            paddle_y / h,
            ball_x / w,
            ball_y / h
        ])


# ------------------------------
# 4) GameRunner: display + controller
# ------------------------------

class GameRunner:
    """
    Run a Pong environment and visualize frames using pygame.
    Pass player_policy to control the agent (left or right paddle) automatically.
    If player_policy is None, human can press keys to control (UP/DOWN).
    """

    def __init__(self, env, action_meanings, player_policy=None, speed=1.0, device=torch.device("cpu"), fps=20):
        self.env = env
        self.meanings = action_meanings
        self.player_policy = player_policy
        self.device = device
        self.fps = fps * speed

        # basic pygame setup for display
        pygame.init()
        # We'll get frame shape dynamically after reset
        self.screen = None
        self.clock = pygame.time.Clock()

        # Map keyboard keys to action indices by querying action meanings automatically:
        # find indices for 'UP' and 'DOWN' in meanings, otherwise fall back to first noop/other mapping
        # action_meanings are strings like: ['NOOP', 'FIRE', 'UP', 'DOWN', ...]
        self.up_idx = None
        self.down_idx = None
        self.noop_idx = None
        for i, s in enumerate(self.meanings):
            s_up = s.upper()
            if "UP" in s_up and self.up_idx is None:
                self.up_idx = i
            if "DOWN" in s_up and self.down_idx is None:
                self.down_idx = i
            if "NOOP" in s.upper():
                self.noop_idx = i
        # fallback to common convention if not found
        if self.up_idx is None:
            self.up_idx = 2 if len(self.meanings) > 2 else 0
        if self.down_idx is None:
            self.down_idx = 3 if len(self.meanings) > 3 else 1

        print("Keyboard mapping: UP ->", self.up_idx, "DOWN ->", self.down_idx)

    def _init_display(self, frame):
        h, w, _ = frame.shape
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Pong Runner")

    def _frame_to_surface(self, frame):
        # frame is RGB array (H,W,3) from env.render()
        # pygame expects (W,H), so we transpose axes
        # convert to surface
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        return surf

    def run(self, n_episodes=1, deterministic=False, scale=2):
        """
        Run a few episodes. If player_policy is None -> human keys control the agent.
        Returns: nothing. Visualization only.
        """
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            if hasattr(self.env, "render"):
                frame = self.env.render()
            else:
                frame = None

            if frame is not None and self.screen is None:
                frame = np.repeat(frame, axis=0, repeats=scale)
                frame = np.repeat(frame, axis=1, repeats=scale)
                self._init_display(frame)

            total_reward = 0.0
            skip_i = 0
            skip_from = 1
            skip_n = 2

            while True:
                # Get action: policy or human keyboard
                if self.player_policy is not None:
                    # obs is stacked frames (FrameStack returns LazyFrame object convertible to np array)
                    if isinstance(obs, np.ndarray):
                        state = obs
                    else:
                        state = np.array(obs)  # try conversion
                    action, logp, infoa = get_action_from_policy(self.player_policy, state, deterministic=deterministic, device=self.device)
                else:
                    # human control by keyboard: use pygame events
                    action = self._human_action_from_pygame()
                    skip_i += 1
                    if skip_i > skip_from:
                        action = int(self.noop_idx)  # skip action to slow down for human control
                    if skip_i == skip_n: skip_i = 0

                # step env
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                done = terminated or truncated
                total_reward += float(reward)

                # render frame and display via pygame
                frame = self.env.render()
                # plt.figure()
                # plt.imshow(frame)
                frame = np.repeat(frame, axis=0, repeats=scale)
                frame = np.repeat(frame, axis=1, repeats=scale)
                if self.screen is None and frame is not None:
                    self._init_display(frame)
                if frame is not None:
                    surf = self._frame_to_surface(frame)
                    self.screen.blit(surf, (0, 0))

                    # small overlay text
                    font = pygame.font.SysFont("Arial", 20)
                    txt = font.render(f"Ep {ep+1}  Reward {total_reward:.2f}", True, (255, 255, 0))
                    self.screen.blit(txt, (10, 10))

                    pygame.display.flip()

                # handle pygame events (close window)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                obs = next_obs

                if done:
                    print(f"Episode {ep+1} finished, total reward: {total_reward}")
                    break

                self.clock.tick(self.fps)

        pygame.quit()

    def _human_action_from_pygame(self, skip_i = 1):
        """
        Read keyboard state and return an action index. If no key pressed => NOOP (or mid action).
        """
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:  # also allow W key for UP
            return int(self.up_idx)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:  # also allow S key for DOWN
            return int(self.down_idx)
        # default: look for NOOP or other fallback
        # try to return index of 'NOOP' if present
        for i, s in enumerate(self.meanings):
            return int(self.noop_idx)
        # else return 0
        return 0


# ------------------------------
# quick demo usage if run as main
# ------------------------------
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Atari Pong")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint Path")
    parser.add_argument("--speed", type=float, default=1.0, help="Game speed")
    args = parser.parse_args()
    env, meanings = create_pong_env()    
    n_actions = env.action_space.n
    input_dim = env.observation_space.shape[0]
    # instantiate a random policy for demo (just logits zeros -> uniform)
    # pol = ConvPolicy(n_actions=env.action_space.n)
    # Run demo with policy controlling the agent:
    player_policy = None
    if args.checkpoint_path is not None:
        policy = MLPPolicy(input_dim, n_actions).to("cpu")
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(checkpoint["policy_state"])
        player_policy = policy
    runner = GameRunner(env, meanings, player_policy=player_policy, speed=args.speed)
    # plt.figure()
    # runner.env.reset()
    # plt.imshow(runner.env.render())
    # plt.show()
    
    runner.run(n_episodes=1, deterministic=False, scale=4)