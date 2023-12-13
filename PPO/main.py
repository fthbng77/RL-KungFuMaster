# train.py
import wandb
from dqn_agent import DQN
from utils import plot_df, save_model
import gym
import numpy as np
import config
import pickle
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    def _init():
        env = gym.make('ALE/KungFuMaster-v5', render_mode="rgb_array")
        env.action_space.seed(42)
        return env
    return _init

def main():
    # WandB yapılandırması
    wandb.init(project='RLkungfumaster', entity='fth123bng')

    # Gym ortamını başlat
    envs = [make_env() for _ in range(config.num_envs)]
    vec_env = DummyVecEnv(envs)

    # DQN modelini başlat
    model = DQN(vec_env, config.lr, config.gamma, config.epsilon, config.epsilon_decay)
    
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