import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
import config
import wandb 
class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.num_envs = env.num_envs
        self.num_action_space = env.action_space.n
        self.num_observation_space = np.prod(env.observation_space.shape)
        self.model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.num_observation_space, activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(self.num_action_space, activation=linear))
        model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=self.lr))
        return model

    def get_action(self, states):
        actions = []
        for state in states:
            if np.random.rand() < self.epsilon:
                actions.append(random.randrange(self.num_action_space))
            else:
                predicted_actions = self.model.predict(state)
                actions.append(np.argmax(predicted_actions[0]))
        return actions

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def learn_and_update_weights_by_reply(self):
        if len(self.replay_memory_buffer) < self.batch_size:
            return

        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        history = self.model.fit(states, target_vec, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        wandb.log({'Loss': loss})

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return states, actions, rewards, next_states, done_list

    def get_random_sample_from_replay_mem(self):
        return random.sample(self.replay_memory_buffer, self.batch_size)
    
    def save_training_progress(self, rewards_list, episode, epsilon):
        with open('training_progress.txt', 'a') as file:
            file.write(f'Episode: {episode}, Average Reward: {sum(rewards_list)/len(rewards_list)}, Epsilon: {epsilon}\n')
    
    def train(self, num_episodes, can_stop=True):
        rewards_list = [[] for _ in range(self.num_envs)]
        for episode in range(num_episodes):
            states = self.env.reset()  # Tüm ortamları sıfırla
            states = np.array([np.reshape(state.flatten(), [1, self.num_observation_space]) for state in states])

            total_rewards = [0 for _ in range(self.num_envs)]
            dones = [False for _ in range(self.num_envs)]

            step = 0
            while not all(dones):
                actions = self.get_action(states)  # Her ortam için ayrı bir eylem seç
                next_states, rewards, dones, _ = self.env.step(actions)

                # Her ortam için verileri işle
                for i in range(self.num_envs):
                    state = states[i]
                    action = actions[i]
                    reward = rewards[i]
                    next_state = next_states[i]
                    done = dones[i]

                    next_state = np.reshape(next_state.flatten(), [1, self.num_observation_space])

                    self.add_to_replay_memory(state, action, reward, next_state, done)
                    total_rewards[i] += reward

                    if step % 100 == 0:
                        wandb.log({'Episode': episode, 'Env': i, 'Step': step, 'Total Reward (Step)': total_rewards[i]})

                states = np.array([np.reshape(state.flatten(), [1, self.num_observation_space]) for state in next_states])
                self.learn_and_update_weights_by_reply()
                step += 1

                if step % 100 == 0:
                    for i in range(self.num_envs):
                        wandb.log({'Episode': episode, 'Env': i, 'Total Reward (Episode)': total_rewards[i]})

            for i in range(self.num_envs):
                rewards_list[i].append(total_rewards[i])
                wandb.log({'Episode': episode, 'Env': i, 'Total Reward (Episode)': total_rewards[i]})

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # Epsilon güncelleme

            # Erken durma koşulu
            if can_stop and all(np.mean(rewards[-100:]) > config.some_threshold for rewards in rewards_list):
                print(f"Erken durma koşulu {episode} bölümünde karşılandı.")
                break

            if episode % 100 == 0 or episode == num_episodes - 1:
                self.save_training_progress(rewards_list, episode, self.epsilon)

        return rewards_list