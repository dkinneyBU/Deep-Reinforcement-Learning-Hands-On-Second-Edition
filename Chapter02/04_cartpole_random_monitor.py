import gymnasium as gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    img = plt.imshow(env.render()) # only call this once

    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        img.set_data(env.render())  # Add this line to render the environment

        if done:
            break

    print(f"Episode done in {total_steps} steps, total reward {total_reward:.2f}")
    env.close()
    env.env.close()