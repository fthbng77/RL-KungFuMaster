from dqn_agent import DQN
from utils import plot_df, save_model, load_trained_model
import gym
import numpy as np
import config 
import pickle
import pandas as pd

def main():
    # Gym ortamını başlat
    env = gym.make('ALE/KungFuMaster-v5', render_mode="rgb_array")
    env.action_space.seed(42)
    np.random.seed(21)

    # DQN modelini başlat
    model = DQN(env, config.lr, config.gamma, config.epsilon, config.epsilon_decay)
    
    print("Starting training for DQN model...")
    model.train(config.training_episodes)

    # Modeli kaydet
    save_dir = "saved_models/"
    save_model(model, save_dir + "trained_model.h5")

    # Eğitim ödüllerini kaydet ve görselleştir
    pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
    rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))
    reward_df = pd.DataFrame(rewards_list)
    plot_df(reward_df, save_dir + "training_rewards.png", "Training Rewards per Episode", "Episode", "Reward")

    # Eğitilmiş modeli test et
    trained_model = load_trained_model(save_dir + "trained_model.h5")
    test_rewards = test_already_trained_model(trained_model, env)
    pickle.dump(test_rewards, open(save_dir + "test_rewards.p", "wb"))
    test_rewards = pickle.load(open(save_dir + "test_rewards.p", "rb"))
    plot_df(pd.DataFrame(test_rewards), save_dir + "testing_rewards.png", "Testing Rewards per Episode", "Episode", "Reward")

    print("Training and Testing Completed!")

def test_already_trained_model(trained_model, env, num_episodes=100):
    test_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(trained_model.predict(state.reshape(1, -1))[0])
            state, reward, done, _ = env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)
        print(f"Episode: {episode}, Total Reward: {total_reward}")
    return test_rewards

if __name__ == "__main__":
    main()
