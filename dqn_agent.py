import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

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
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets
        self.model.fit(states, target_vec, epochs=1, verbose=0)

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

    def train(self, num_episodes, can_stop=True):
        rewards_list = []
        for episode in range(num_episodes):
            initial_state = self.env.reset()
            state = initial_state[0] if isinstance(initial_state, tuple) else initial_state
            state_flattened = state.flatten()
            state = np.reshape(state_flattened, [1, self.num_observation_space])

            print(f"Initial state shape after flattening and reshaping: {state.shape}")
            total_reward = 0
            done = False
            while not done:
                action = self.get_action(state)
                step_result = self.env.step(action)

                # step_result'dan gerekli öğeleri al
                next_state = step_result[0]
                reward = step_result[1]
                done = step_result[2]

                # next_state'i düzleştir ve işle
                next_state_flattened = next_state.flatten()
                next_state = np.reshape(next_state_flattened, [1, self.num_observation_space])

                self.add_to_replay_memory(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.learn_and_update_weights_by_reply()
                if done:
                    break
            rewards_list.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
        return rewards_list
