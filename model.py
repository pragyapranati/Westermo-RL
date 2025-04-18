import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Load and fill missing
df = pd.read_csv("merged_data.csv")
df = df.ffill()

# Convert server-up to binary: 0 = normal, 1 = anomaly
df['server-up'] = df['server-up'].replace({2: 0, 1: 1})

# Balance classes
normal = df[df['server-up'] == 0]
anomalies = df[df['server-up'] == 1]
normal_sampled = normal.sample(n=len(anomalies), random_state=42)
df_balanced = pd.concat([normal_sampled, anomalies]).sample(frac=1, random_state=42)

# Feature selection
selected_features = ['sys-mem-swap-total', 'sys-mem-total', 'sys-mem-swap-free',
                     'sys-context-switch-rate', 'sys-mem-cache', 'cpu-system', 'cpu-user']
print("Selected features for anomaly detection:", selected_features)

X = df_balanced[selected_features]
y = df_balanced['server-up']

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# RL Environment
class AnomalyEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.index = 0

    def reset(self):
        self.index = 0
        return self.X[self.index]

    def step(self, action):
        label = self.y[self.index]
        # Reward shaping
        reward = 2 if action == label and label == 1 else 1 if action == label else -2
        self.index += 1
        done = self.index >= len(self.X)
        next_state = self.X[self.index] if not done else self.X[-1]
        return next_state, reward, done

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Training
def train_dqn(env, input_dim, episodes=10, max_steps=2000):
    model = DQN(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    gamma = 0.9

    for ep in range(episodes):
        epsilon = max(0.05, 1.0 - (ep / episodes))  # ε-decay
        state = env.reset()
        total_reward = 0
        steps = 0

        while steps < max_steps:
            state_tensor = torch.FloatTensor(state)

            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(2)
            else:
                with torch.no_grad():
                    action = model(state_tensor).argmax().item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            next_state_tensor = torch.FloatTensor(next_state)
            target = reward + (gamma * model(next_state_tensor).max().item() if not done else 0)

            output = model(state_tensor)[action]
            loss = criterion(output, torch.tensor(target).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            steps += 1
            if done:
                break

        print(f"Episode {ep + 1} - Total Reward: {total_reward}")

    torch.save(model.state_dict(), "rl_anomaly_model.pth")
    return model

# Train
env = AnomalyEnv(X_scaled, y.values)
model = train_dqn(env, input_dim=X_scaled.shape[1])
