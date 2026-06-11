import gymnasium as gym
from custom_cartpole import WindCartPoleEnv

# Register the environment
gym.register(
    id='WindCartPole-v0',
    entry_point='custom_cartpole:WindCartPoleEnv',
    max_episode_steps=500,
)

def run_simulation(env_id='WindCartPole-v0', episodes=5, render_mode=None, **env_kwargs):
    print(f"Creating environment: {env_id} with kwargs: {env_kwargs}")
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not done and not truncated:
            # Simple random agent
            action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step_count}")

    env.close()

def main():
    print("Running with random wind...")
    run_simulation(wind_mode='random', wind_intensity=5.0)

    print("\nRunning with sinusoidal wind...")
    run_simulation(wind_mode='sinusoidal', wind_intensity=10.0)

if __name__ == "__main__":
    main()
