import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
import plotly.express as px
import scipy.cluster.hierarchy as sch
import torch
import torch.nn as nn

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

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("merged_data.csv")
    df = df.ffill()
    df = df.sample(n=10000, random_state=42)
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
options = [
    "Overview",
    "Data Visualization",
    "Statistics",
    "Feature Analysis",
    "Anomaly Detection"
]
choice = st.sidebar.radio("Go to", options)

if choice == "Overview":
    st.title("Westermo System Performance Dashboard")
    st.markdown("<medium>- Collected from Westermo industrial network devices (routers, switches).</medium>", unsafe_allow_html=True)
    st.markdown("<medium>- Data sourced via telemetry logs, SNMP, and device diagnostics.</medium>", unsafe_allow_html=True)
    st.markdown("<medium>- Includes metrics like CPU load, memory usage, throughput, errors, and uptime.</medium>", unsafe_allow_html=True)
    st.markdown("<medium>- Used for monitoring, diagnostics, and predictive maintenance.</medium>", unsafe_allow_html=True)

    st.write("### Sample Data")
    st.dataframe(df.head())

elif choice == "Data Visualization":
    st.title("Data Visualization")

    df_vis = df.head(200).copy()
    df_vis["server-up"] = df_vis["server-up"].astype(str)

    st.subheader("Line Chart (first 100 rows)")
    st.line_chart(df[selected_features].head(100))

    st.subheader("Bar Chart - Top 10 Context Switch Rate")
    top_context = df.nlargest(10, 'sys-context-switch-rate')
    st.bar_chart(top_context[['sys-context-switch-rate']])

    st.subheader("Pie Chart - Server Status Distribution")
    status_counts = df_vis['server-up'].value_counts()
    fig_pie = px.pie(values=status_counts, names=status_counts.index, title='Server-Up Distribution')
    st.plotly_chart(fig_pie)

    st.subheader("Histogram - CPU System")
    filtered_cpu_system = df_vis['cpu-system'][df_vis['cpu-system'] > 0]
    fig, ax = plt.subplots()
    ax.hist(filtered_cpu_system.dropna(), bins=20, color='skyblue', edgecolor='black')
    st.pyplot(fig)

    st.subheader("Bubble Chart")
    fig_bubble = px.scatter(
        df_vis,
        x="cpu-user",
        y="sys-mem-cache",
        size="sys-context-switch-rate",
        color="server-up",
        title="Bubble Chart"
    )
    st.plotly_chart(fig_bubble)

    st.subheader("Box Plot - CPU Usage")
    fig_box = px.box(df_vis, y=["cpu-system", "cpu-user"])
    st.plotly_chart(fig_box)

    st.subheader("Tree Map - Memory Stats")
    fig_tree = px.treemap(
        df_vis,
        path=["server-up"],
        values="sys-mem-cache",
        color="sys-context-switch-rate"
    )
    st.plotly_chart(fig_tree)

    st.subheader("Dendrogram - Feature Clustering")
    linkage = sch.linkage(df_vis[selected_features].dropna(), method='ward')
    fig, ax = plt.subplots(figsize=(10, 6))
    sch.dendrogram(linkage, ax=ax)
    st.pyplot(fig)

    st.subheader("Scatter Plot")
    fig_scatter = px.scatter(df_vis, x="sys-mem-cache", y="cpu-system", color="server-up")
    st.plotly_chart(fig_scatter)

    st.subheader("Heatmap - Correlation")
    corr = df_vis.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, ax=ax, cmap="coolwarm")
    st.pyplot(fig)

elif choice == "Feature Analysis":
    st.title("Feature Engineering")
    st.write("Selected Features:", selected_features)

    X = df[selected_features]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("PCA for Dimensionality Reduction")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], color=df['server-up'].astype(str))
    st.plotly_chart(fig)

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(X_scaled, columns=selected_features).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
elif choice == "Statistics":
    st.title("Descriptive & Inferential Statistics")
    st.subheader("Central Tendency")
    st.write("Mean:", df.mean(numeric_only=True))
    st.write("Median:", df.median(numeric_only=True))
    st.write("Mode:", df.mode(numeric_only=True).iloc[0])

    st.subheader("Dispersion")
    st.write("Variance:", df.var(numeric_only=True))
    st.write("Range:", df.max(numeric_only=True) - df.min(numeric_only=True))
    st.write("IQR:", df.quantile(0.75, numeric_only=True) - df.quantile(0.25, numeric_only=True))

    st.subheader("Distribution Shape")
    st.write("Skewness:", df.skew(numeric_only=True))
    st.write("Kurtosis:", df.kurtosis(numeric_only=True))

    st.subheader("Hypothesis Test (t-test)")
    stat, pval = stats.ttest_ind(df[df['server-up'] == 1]['cpu-system'],
                                 df[df['server-up'] == 2]['cpu-system'],
                                 equal_var=False)
    st.write(f"T-test statistic: {stat}, P-value: {pval}")
    if pval < 0.05:
        st.success("Reject null hypothesis: significant difference in CPU system usage.")
    else:
        st.info("Fail to reject null hypothesis: no significant difference in CPU system usage.")
    st.subheader("ANOVA Test")
    f_stat, pval_anova = stats.f_oneway(
        df[df['server-up'] == 1]['cpu-system'],
        df[df['server-up'] == 2]['cpu-system'],
        df[df['server-up'] == 3]['cpu-system']
    )

    st.write(f"ANOVA F-statistic: {f_stat}, P-value: {pval_anova}")
    if pval_anova < 0.05:
        st.success("Reject null hypothesis: significant difference in CPU system usage across groups.")
    else:
        st.info("Fail to reject null hypothesis: no significant difference in CPU system usage across groups.")
    st.subheader("Chi-Squared Test")
    contingency_table = pd.crosstab(df['server-up'], df['cpu-system'] > 0)
    chi2, pval_chi2, _, _ = stats.chi2_contingency(contingency_table)
    st.write(f"Chi-squared statistic: {chi2}, P-value: {pval_chi2}")
    if pval_chi2 < 0.05:
        st.success("Reject null hypothesis: significant association between server status and CPU system usage.")
    else:
        st.info("Fail to reject null hypothesis: no significant association between server status and CPU system usage.")

    st.subheader("Correlation Coefficient")
    corr_matrix = df.corr(numeric_only=True)
    st.write("Correlation Coefficient Matrix:")
    st.write(corr_matrix)
    st.subheader("Feature Importance")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    X = df[selected_features]
    y = df['server-up']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.subheader("Feature Importance Plot")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [selected_features[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    st.pyplot(plt)
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    
elif choice == "Anomaly Detection":
    st.title("Anomaly Detection (RL Model)")

    X = df[selected_features]
    scaler = MinMaxScaler()
    scaler.fit(X)

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

    model = DQN(len(selected_features))
    model.load_state_dict(torch.load("rl_anomaly_model.pth", map_location=torch.device('cpu')))
    model.eval()

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

            st.subheader("Contributing Features:")
            means = df[selected_features].mean()
            stds = df[selected_features].std()
            z_scores = np.abs((input_df.iloc[0] - means) / stds)
            abnormal_features = z_scores[z_scores > 2].index.tolist()

            if abnormal_features:
                for feat in abnormal_features:
                    st.markdown(f"**{feat}**: {explanations.get(feat, 'No explanation available.')}")
            else:
                st.info("No single feature strongly deviated (z-score < 2).")
