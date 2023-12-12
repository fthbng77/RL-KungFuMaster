import wandb
from dqn_agent import DQN
from utils import plot_df, save_model, load_trained_model
import gym
import numpy as np
import config
import pickle
import pandas as pd

def main():
    # WandB yapılandırması
    wandb.init(project='your-project-name', entity='your-user-name')

    # Gym ortamını başlat
    env = gym.make('ALE/KungFuMaster-v5', render_mode="rgb_array")
    env.action_space.seed(42)
    np.random.seed(21)

    # DQN modelini başlat
    model = DQN(env, config.lr, config.gamma, config.epsilon, config.epsilon_decay)
    
    print("Starting training for DQN model...")
    training_rewards = model.train(config.training_episodes)

    # Modeli kaydet ve WandB'a yükle
    save_dir = "saved_models/"
    model_path = save_dir + "trained_model.h5"
    save_model(model.model, model_path)
    wandb.save(model_path)

    # Eğitim ödüllerini kaydet ve görselleştir
    pickle.dump(training_rewards, open(save_dir + "train_rewards_list.p", "wb"))
    reward_df = pd.DataFrame(training_rewards)
    plot_df(reward_df, save_dir + "training_rewards.png", "Training Rewards per Episode", "Episode", "Reward")
    wandb.log({"Training Rewards": wandb.Image(save_dir + "training_rewards.png")})

    # Eğitilmiş modeli test et
    trained_model = load_trained_model(model_path)
    test_rewards = test_already_trained_model(trained_model, env)
    pickle.dump(test_rewards, open(save_dir + "test_rewards.p", "wb"))
    test_rewards_df = pd.DataFrame(test_rewards)
    plot_df(test_rewards_df, save_dir + "testing_rewards.png", "Testing Rewards per Episode", "Episode", "Reward")
    wandb.log({"Testing Rewards": wandb.Image(save_dir + "testing_rewards.png")})

    print("Training and Testing Completed!")

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

if __name__ == "__main__":
    main()
