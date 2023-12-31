import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
import config
import wandb 

if tf.test.is_gpu_available(cuda_only=True):
    print("CUDA destekli GPU kullanılabilir.")
else:
    print("CUDA destekli GPU kullanılamıyor, CPU üzerinde çalışılacak.")


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
        self.num_action_space = env.action_space.n
        self.num_observation_space = np.prod(env.observation_space.shape)
        self.model = self.initialize_model()

    def initialize_model(self):
        with tf.device('/GPU:0'):
            model = Sequential()
            model.add(Dense(512, input_dim=self.num_observation_space, activation=relu))
            model.add(Dense(256, activation=relu))
            model.add(Dense(self.num_action_space, activation=linear))
            model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=self.lr))
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.num_action_space)
        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def learn_and_update_weights_by_reply(self):
        if len(self.replay_memory_buffer) < self.batch_size:
            return

        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)

        targets = np.zeros(self.batch_size)
        for idx in range(self.batch_size):
            Q_future_max = np.amax(self.model.predict_on_batch(next_states[idx:idx+1]))
            targets[idx] = rewards[idx] if done_list[idx] else rewards[idx] + self.gamma * Q_future_max

        target_vec = self.model.predict_on_batch(states)
        for idx, action in enumerate(actions):
            target_vec[idx, action] = targets[idx]

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
        rewards_list = []
        for episode in range(num_episodes):
            initial_state = self.env.reset()
            state = initial_state[0] if isinstance(initial_state, tuple) else initial_state
            state_flattened = state.flatten()
            state = np.reshape(state_flattened, [1, self.num_observation_space])

            total_reward = 0
            done = False
            step = 0
            while not done:
                action = self.get_action(state)
                step_result = self.env.step([action])

                next_state = step_result[0]
                reward = step_result[1]
                done = step_result[2]

                next_state_flattened = next_state.flatten()
                next_state = np.reshape(next_state_flattened, [1, self.num_observation_space])

                self.add_to_replay_memory(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.learn_and_update_weights_by_reply()

                step += 1
                if step % 100 == 0:
                    wandb.log({'Episode': episode, 'Step': step, 'Total Reward (Step)': total_reward})

                if done:
                    break

            rewards_list.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            wandb.log({'Episode': episode, 'Total Reward (Episode)': total_reward, 'Epsilon': self.epsilon})
        # Erken durma koşulu
            if can_stop and np.mean(rewards_list[-100:]) > config.some_threshold:
                print(f"Erken durma koşulu {episode} bölümünde karşılandı.")
                break

            if episode % 100 == 0 or episode == num_episodes - 1:
                self.save_training_progress(rewards_list, episode, self.epsilon)

        return rewards_list
