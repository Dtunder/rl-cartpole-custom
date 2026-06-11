import gymnasium as gym
from custom_cartpole import WindCartPoleEnv

# Register the environment
gym.register(
    id='WindCartPole-v0',
    entry_point='custom_cartpole:WindCartPoleEnv',
    max_episode_steps=500,
)

def run_simulation(env_id='WindCartPole-v0', episodes=5, render_mode=None, **env_kwargs):
    if not isinstance(episodes, int):
        raise TypeError(f"episodes must be an integer, got {type(episodes).__name__}")
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")

    print(f"Creating environment: {env_id} with kwargs: {env_kwargs}")
    try:
        env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return

    try:
        for episode in range(episodes):
            success = _run_single_episode(env, episode)
            if not success:
                break
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"Error while closing environment: {e}")


def _run_single_episode(env, episode_index):
    """
    Helper function to run a single episode.
    Extracts the inner simulation loop to reduce nesting.
    """
    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        return False
        
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    while not done and not truncated:
        # Simple random agent
        action = env.action_space.sample()
        
        try:
            obs, reward, done, truncated, info = env.step(action)
        except Exception as e:
            print(f"Error during step: {e}")
            break
            
        total_reward += reward
        step_count += 1
        
    print(f"Episode {episode_index + 1}: Total Reward = {total_reward}, Steps = {step_count}")
    return True

def main():
    print("Running with random wind...")
    run_simulation(wind_mode='random', wind_intensity=5.0)
    
    print("\nRunning with sinusoidal wind...")
    run_simulation(wind_mode='sinusoidal', wind_intensity=10.0)

if __name__ == "__main__":
    main()
