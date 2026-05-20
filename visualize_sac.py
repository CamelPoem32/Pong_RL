import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np

def plot_sac_history(checkpoint_path, window_size=10):
    # Load the checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        return

    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Extract history
    history = checkpoint.get("history", {})
    rewards = history.get("reward_history", [])
    losses = history.get("critic_loss", [])

    if not rewards:
        print("No reward history found in checkpoint.")
        return

    # Create DataFrames for easy smoothing
    df_rewards = pd.DataFrame(rewards, columns=['reward'])
    df_rewards['smoothed'] = df_rewards['reward'].rolling(window=window_size).mean()

    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"SAC Training Progress: {os.path.basename(checkpoint_path)}", fontsize=16)

    # Plot 1: Rewards
    ax1.plot(df_rewards['reward'], alpha=0.3, color='dodgerblue', label='Raw Reward')
    ax1.plot(df_rewards['smoothed'], color='blue', linewidth=2, label=f'Moving Avg (w={window_size})')
    ax1.set_ylabel("Episode Reward")
    ax1.set_xlabel("Episodes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Critic Loss
    if losses:
        smoothed_losses = np.convolve(losses[::1000], np.ones(window_size)/(window_size), mode='valid')
        ax2.plot(losses[::1000], alpha=0.3, color='pink', label='Raw Loss')
        ax2.plot(smoothed_losses, color='crimson', linewidth=2, label=f'Moving Avg (w={window_size})')
        ax2.set_yscale('log') # Losses vary wildly, log scale helps
        ax2.set_ylabel("Critic Loss (Log Scale)")
        ax2.set_xlabel("Update Steps (sampled per episode)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No Loss Data Available", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or Show
    save_path = checkpoint_path.replace(".pt", "_plot.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAC Training History")
    parser.add_argument("path", type=str, help="Path to the .pt checkpoint file")
    parser.add_argument("--window", type=int, default=20, help="Smoothing window size")
    args = parser.parse_args()

    plot_sac_history(args.path, window_size=args.window)