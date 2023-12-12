# train.py
import wandb
from dqn_agent import DQN
from utils import plot_df, save_model
import gym
import numpy as np
import config
import pickle
import pandas as pd

def main():
    # WandB yapılandırması
    wandb.init(project='RKkungfumaster', entity='fth123bng')

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

    print("Training Completed!")

if __name__ == "__main__":
    main()