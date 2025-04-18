import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Define selected features and explanations
selected_features = ['sys-mem-swap-total', 'sys-mem-total', 'sys-mem-swap-free',
                     'sys-context-switch-rate', 'sys-mem-cache', 'cpu-system', 'cpu-user']

explanations = {
    'sys-mem-swap-total': "Total size of swap memory is unusual — possible misconfigured swap partition.",
    'sys-mem-total': "System memory total is abnormal — potential hardware or config issue.",
    'sys-mem-swap-free': "Free swap memory is low/high — indicates memory pressure or idle resources.",
    'sys-context-switch-rate': "High context switching — potential CPU contention or overloaded scheduler.",
    'sys-mem-cache': "Memory cache is off-normal — could suggest disk I/O bottlenecks or memory mismanagement.",
    'cpu-system': "CPU system usage is off — may suggest high kernel-level processing.",
    'cpu-user': "Unusual user-space CPU activity — possible high user process load or faulty app."
}

# Load scaler and model
scaler = MinMaxScaler()
df = pd.read_csv("merged_data.csv").fillna(method='ffill')
df = df[selected_features]
scaler.fit(df)

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

# Load trained model
model = DQN(len(selected_features))
model.load_state_dict(torch.load("rl_anomaly_model.pth", map_location=torch.device('cpu')))
model.eval()

# App UI
st.title("🔍 RL-based Anomaly Detection for Westermo Metrics")

st.markdown("Enter system performance values below:")

user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"{feature}", step=0.1)

if st.button("Detect Anomaly"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.FloatTensor(input_scaled[0])

    with torch.no_grad():
        prediction = model(input_tensor).argmax().item()

    st.subheader("Prediction:")
    if prediction == 1:
        st.success("✅ System working normally.")
    else:
        st.error("⚠️ Anomaly detected!")

        # Explain why
        st.subheader("Contributing Features:")
        means = df.mean()
        stds = df.std()

        z_scores = np.abs((input_df.iloc[0] - means) / stds)
        abnormal_features = z_scores[z_scores > 2].index.tolist()

        if abnormal_features:
            for feat in abnormal_features:
                st.markdown(f"**{feat}**: {explanations.get(feat, 'No explanation available.')}")
        else:
            st.info("No single feature strongly deviated (z-score < 2).")
