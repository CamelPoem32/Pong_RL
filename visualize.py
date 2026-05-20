import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_checkpoint_data(checkpoint_paths, solver_names=None):
    """
    Load multiple checkpoint files and extract histories.
    
    Args:
        checkpoint_paths: List of paths to .pt checkpoint files
        solver_names: List of names for each solver (if None, uses filename)
    
    Returns:
        dict: Dictionary with solver names as keys and dicts containing histories
    """
    if solver_names is None:
        solver_names = [Path(path).stem.split('_')[0] for path in checkpoint_paths]
    
    data = {}
    
    for path, name in zip(checkpoint_paths, solver_names):
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if name != "sac":
                data[name] = {
                    'epochs': checkpoint.get('epoch_history', 
                            list(range(len(checkpoint['reward_history'])))),
                    'reward_history': checkpoint['reward_history'],
                    'loss_history': checkpoint.get('loss_history', []),
                    'time_history': checkpoint.get('time_history', None),
                    'epoch': checkpoint['epoch']
                }
                if name == "ppo": 
                    data[name]['policy_loss_history'] = checkpoint.get('policy_loss_history', [])
                    data[name]['value_loss_history'] = checkpoint.get('value_loss_history', [])
            elif name == "sac": 
                history = checkpoint.get('history', [])
                data[name] = {}
                data[name]['reward_history'] = history['reward_history']
                data[name]['loss_history'] = np.interp(np.arange(len(history['reward_history'])), np.arange(len(history['critic_loss'])), history['critic_loss'])
                data[name]['epochs'] = list(range(len(history['reward_history'])))
                data[name]['epoch'] = checkpoint['episode_idx']
                data[name]['time_history'] = None
            print(f"Loaded {name}: {len(data[name]['reward_history'])} rewards, "
                  f"{len(data[name]['loss_history'])} losses")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return data


def correct_time_history(time_history, jump_threshold=3.0):
    """
    Correct time history by identifying and fixing jumps.
    
    Args:
        time_history: Array of timestamps
        jump_threshold: Multiplier threshold to identify jumps (e.g., 3x mean diff)
    
    Returns:
        corrected_time_history: Fixed timestamps
    """
    if time_history is None or len(time_history) < 2:
        return time_history
    
    time_history_init = time_history.copy()
    time_history = np.cumsum(time_history)
    time_diffs = np.diff(time_history)
    
    # Calculate mean of normal diffs (excluding outliers)
    mean_diff = np.array([min(time_history_init[i+1], time_history_init[i]) for i in range(len(time_history_init)-1)])
    std_diff = np.std(time_diffs)
    
    # Identify jumps (diffs > threshold * mean_diff)
    jump_indices = np.where(time_diffs > jump_threshold * mean_diff)[0]
    
    if len(jump_indices) == 0:
        print("No significant jumps detected in time history")
        return time_history
    
    print(f"Detected {len(jump_indices)} jumps in time history")
    
    # Create corrected time array
    corrected_time = time_history.copy()
    
    for idx in jump_indices:
        # This diff from idx to idx+1 is too large
        # Replace with mean of surrounding diffs
        if idx > 0 and idx < len(time_diffs) - 1:
            # Use mean of previous and next diffs
            new_diff = (time_diffs[idx-1] + time_diffs[idx+1]) / 2
        elif idx == 0 and len(time_diffs) > 1:
            # First diff, use next diff
            new_diff = time_diffs[1]
        elif idx == len(time_diffs) - 1 and len(time_diffs) > 1:
            # Last diff, use previous diff
            new_diff = time_diffs[-2]
        else:
            # Fallback to global mean
            new_diff = mean_diff
        
        # Adjust all subsequent times
        shift = time_diffs[idx] - new_diff
        corrected_time[idx+1:] -= shift
    
    return corrected_time


def plot_checkpoints_by_epoch(checkpoint_paths, solver_names=None, save_path=None):
    """
    Plot reward and loss histories from checkpoints over epochs.
    
    Args:
        checkpoint_paths: List of paths to .pt checkpoint files
        solver_names: List of names for each solver (REINFORCE, NPG, TRPO, PPO)
        save_path: Optional path to save the figure
    """
    # Load data
    data = load_checkpoint_data(checkpoint_paths, solver_names)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Colors for different solvers
    colors = {'reinforce': 'red', 'npg': 'blue', 'trpo': 'green', 'ppo': 'purple'}
    
    # Plot rewards
    for name, solver_data in data.items():
        color = colors.get(name.lower(), None)
        epochs = solver_data['epochs']
        rewards = solver_data['reward_history']
        
        ax1.plot(epochs, rewards, label=name.upper(), linewidth=0.5, color=color)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Rewards over Episodes', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot losses
    for name, solver_data in data.items():
        color = colors.get(name.lower(), None)
        if len(solver_data['loss_history']) > 0:
            # Losses might be recorded at different frequency
            # Create x-axis that matches loss recording frequency
            if len(solver_data['loss_history']) == len(solver_data['epochs']):
                loss_epochs = np.array(solver_data['epochs'])*5
            else:
                # Assume losses are recorded per update, not per epoch
                # Use indices scaled to epoch range
                loss_epochs = np.linspace(0, np.array(solver_data['epochs'])*5, 
                                          len(solver_data['loss_history']))
            plus_label = ""
            # if name == "reinforce": plus_label = " x 1e4"
            ax2.plot(loss_epochs, -np.log10(np.abs(solver_data['loss_history'])), 
                    label=f'{name.upper()} Loss{plus_label}', linewidth=0.5, color=color, alpha=0.7)
        elif name == "ppo":
            loss_epochs = solver_data['epochs']
            ax2.plot(loss_epochs, -np.log10(np.abs(solver_data['policy_loss_history'])), 
                    label=f'{name.upper()} Policy Loss{plus_label}', linewidth=0.5, color=color, alpha=0.7)
            ax2.plot(loss_epochs, -np.log10(np.abs(solver_data['value_loss_history'])), 
                    label=f'{name.upper()} Value Loss{plus_label}', linewidth=0.5, color=color, marker="o", markersize=8, markevery=500, alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Log Loss', fontsize=12)
    ax2.set_title('Training Losses over Epochs', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.suptitle('Policy Gradient Methods Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return data


def plot_checkpoints_by_time(checkpoint_paths, solver_names=None, 
                            jump_threshold=3.0, save_path=None):
    """
    Plot reward and loss histories from checkpoints over corrected time.
    
    Args:
        checkpoint_paths: List of paths to .pt checkpoint files
        solver_names: List of names for each solver
        jump_threshold: Threshold multiplier for detecting time jumps
        save_path: Optional path to save the figure
    """
    # Load data
    data = load_checkpoint_data(checkpoint_paths, solver_names)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Colors for different solvers
    colors = {'reinforce': 'red', 'npg': 'blue', 'trpo': 'green', 'ppo': 'purple'}
    
    # Plot rewards over corrected time
    for name, solver_data in data.items():
        color = colors.get(name.lower(), None)
        
        if solver_data['time_history'] is not None:
            # Correct time history
            corrected_time = correct_time_history(solver_data['time_history'], 
                                                  jump_threshold)
            
            # Rewards are recorded per epoch, time might be per step
            # Align time with rewards (take time at end of each epoch)
            if len(corrected_time) >= len(solver_data['reward_history']):
                # Take time at reward recording points
                time_points = corrected_time[:len(solver_data['reward_history'])]
            else:
                time_points = np.linspace(corrected_time[0], corrected_time[-1], 
                                         len(solver_data['reward_history']))
            
            ax1.plot(time_points, solver_data['reward_history'], 
                    label=name.upper(), linewidth=0.5, color=color)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Rewards over Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot losses over corrected time
    for name, solver_data in data.items():
        color = colors.get(name.lower(), None)
        
        if solver_data['time_history'] is not None and len(solver_data['loss_history']) > 0:
            corrected_time = correct_time_history(solver_data['time_history'], 
                                                  jump_threshold)
            
            # Losses might be recorded at different frequency
            if len(solver_data['loss_history']) == len(corrected_time):
                loss_time = corrected_time
            else:
                # Assume losses recorded at each update
                loss_time = np.linspace(corrected_time[0], corrected_time[-1], 
                                       len(solver_data['loss_history']))
            
            plus_label = ""
            # if name == "reinforce": plus_label = " x 1e4"
            ax2.plot(loss_time, -np.log10(np.abs(solver_data['loss_history'])), 
                    label=f'{name.upper()} Loss{plus_label}', linewidth=0.5, color=color, alpha=0.7)
        elif name == "ppo":
            corrected_time = correct_time_history(solver_data['time_history'], 
                                                  jump_threshold)
            loss_time = corrected_time
            ax2.plot(loss_time, -np.log10(np.abs(solver_data['policy_loss_history'])), 
                    label=f'{name.upper()} Policy Loss{plus_label}', linewidth=0.5, color=color, alpha=0.7)
            ax2.plot(loss_time, -np.log10(np.abs(solver_data['value_loss_history'])), 
                    label=f'{name.upper()} Value Loss{plus_label}', linewidth=0.5, color=color, marker="o", markersize=8, markevery=500, alpha=0.7)
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Log Loss', fontsize=12)
    ax2.set_title('Training Losses over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.suptitle('Policy Gradient Methods Comparison (Time-Corrected)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return data


def plot_smoothed_rewards(checkpoint_paths, solver_names=None, 
                         window=20, by_time=False, jump_threshold=3.0, save_path=None):
    """
    Plot smoothed rewards with confidence intervals.
    
    Args:
        checkpoint_paths: List of paths to .pt checkpoint files
        solver_names: List of names for each solver
        window: Smoothing window size
        by_time: If True, plot over time instead of epochs
        jump_threshold: Threshold for time correction
        save_path: Optional path to save the figure
    """
    data = load_checkpoint_data(checkpoint_paths, solver_names)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = {'reinforce': 'red', 'npg': 'blue', 'trpo': 'green', 'ppo': 'purple'}
    
    for name, solver_data in data.items():
        color = colors.get(name.lower(), None)
        rewards = np.array(solver_data['reward_history'])
        
        # Smooth rewards
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # Calculate confidence interval (standard deviation of smoothing window)
        std = np.array([np.std(rewards[max(0, i-window):i+1]) 
                       for i in range(window-1, len(rewards))])
        
        # Determine x-axis
        if by_time and solver_data['time_history'] is not None:
            corrected_time = correct_time_history(solver_data['time_history'], 
                                                  jump_threshold)
            x = corrected_time[:len(smoothed)]
            xlabel = 'Time (seconds)'
        else:
            if len(solver_data['epochs']) == len(rewards):
                x = solver_data['epochs'][window-1:]
            else:
                x = np.arange(window-1, len(rewards))
            xlabel = 'Epoch'
        
        ax.plot(x, smoothed, label=name.upper(), linewidth=1, color=color)
        ax.fill_between(x, smoothed - std, smoothed + std, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'Smoothed Rewards (window={window})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """
    Main function to demonstrate plotting with your specific checkpoints.
    """
    # Your checkpoint paths
    checkpoint_paths = [
        "results_reinforce/checkpoint_10000epoch_5ep.pt",
        "results_npg/checkpoint_1000epoch_5ep.pt",
        "results_trpo/checkpoint_6000epoch_5ep.pt",
        "results_ppo/checkpoint_5000epoch_5ep.pt",
        "results_sac/sac_epoch_150_gr10.pt",
    ]
    
    # Solver names (in same order)
    solver_names = ["reinforce", "npg", "trpo", "ppo", "sac"]
    
    # Check which files exist
    existing_paths = []
    existing_names = []
    for path, name in zip(checkpoint_paths, solver_names):
        if os.path.exists(path):
            existing_paths.append(path)
            existing_names.append(name)
            print(f"Found: {name} at {path}")
        else:
            print(f"Warning: {path} not found")
    
    if not existing_paths:
        print("No checkpoint files found!")
        return
    
    # Plot 1: Rewards and losses over epochs
    print("\n=== Plotting over epochs ===")
    data_epoch = plot_checkpoints_by_epoch(
        existing_paths, 
        existing_names,
        # save_path="comparison_by_epoch.png"
    )
    
    # Plot 2: Rewards and losses over corrected time
    print("\n=== Plotting over corrected time ===")
    data_time = plot_checkpoints_by_time(
        existing_paths,
        existing_names,
        jump_threshold=3.0,  # Detect jumps > 3x mean diff
        # save_path="comparison_by_time.png"
    )
    
    # Bonus: Smoothed rewards plot
    print("\n=== Plotting smoothed rewards ===")
    plot_smoothed_rewards(
        existing_paths,
        existing_names,
        window=20,
        # save_path="smoothed_rewards.png"
    )


if __name__ == "__main__":
    main()