# test.py
from utils import load_trained_model
from utils import plot_df, load_trained_model
import gym
import numpy as np
import config
import pickle
import pandas as pd
import wandb

def test_already_trained_model(trained_model, env, num_episodes=100):
    test_rewards = []
    for episode in range(num_episodes):
        initial_state = env.reset()
        state = initial_state[0] if isinstance(initial_state, tuple) else initial_state
        state_flattened = state.flatten()
        state = np.reshape(state_flattened, [1, np.prod(env.observation_space.shape)])

        total_reward = 0
        done = False
        while not done:
            action = np.argmax(trained_model.predict(state)[0])
            step_result = env.step(action)
            next_state = step_result[0]
            reward = step_result[1]
            done = step_result[2]

            next_state_flattened = next_state.flatten()
            state = np.reshape(next_state_flattened, [1, np.prod(env.observation_space.shape)])

            total_reward += reward
        test_rewards.append(total_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        wandb.log({'Test Episode': episode, 'Total Reward': total_reward})
    return test_rewards

def test_model():
    # WandB yapılandırması
    wandb.init(project='RKkungfumaster', entity='fth123bng', job_type="testing")

    # Gym ortamını başlat
    env = gym.make('ALE/KungFuMaster-v5', render_mode="rgb_array")

    # Eğitilmiş modeli yükle
    save_dir = "saved_models/"
    model_path = save_dir + "trained_model.h5"
    trained_model = load_trained_model(model_path)

    # Modeli test et
    test_rewards = test_already_trained_model(trained_model, env)
    pickle.dump(test_rewards, open(save_dir + "test_rewards.p", "wb"))
    test_rewards_df = pd.DataFrame(test_rewards)
    plot_df(test_rewards_df, save_dir + "testing_rewards.png", "Testing Rewards per Episode", "Episode", "Reward")
    wandb.log({"Testing Rewards": wandb.Image(save_dir + "testing_rewards.png")})

    print("Testing Completed!")

if __name__ == "__main__":
    test_model()