# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# st.set_page_config(
#     page_title="Customer Segmentation App",
#     layout="wide"
# )
# # Title
# st.title("Customer Segmentation App")

# st.write("Customer Segmentation using K-Means Clustering")

# # Load Dataset
# df = pd.read_csv("data/Mall_Customers.csv")

# # Show Dataset
# st.subheader("Dataset")

# st.dataframe(df.head())
# st.subheader("Dataset Information")

# col1, col2, col3 = st.columns(3)

# col1.metric("Total Customers", df.shape[0])

# col2.metric("Average Income", round(df['Annual Income (k$)'].mean(), 2))

# col3.metric("Average Spending Score", round(df['Spending Score (1-100)'].mean(), 2))

# # Select Features
# X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# # Train Model
# kmeans = KMeans(n_clusters=5, random_state=42)

# y_kmeans = kmeans.fit_predict(X)

# # Create Plot
# fig, ax = plt.subplots(figsize=(8,6))

# scatter = ax.scatter(
#     X.iloc[:,0],
#     X.iloc[:,1],
#     c=y_kmeans,
#     cmap='rainbow'
# )

# # Cluster Centers
# ax.scatter(
#     kmeans.cluster_centers_[:,0],
#     kmeans.cluster_centers_[:,1],
#     s=300,
#     c='black',
#     label='Centroids'
# )

# ax.set_title("Customer Segments")

# ax.set_xlabel("Annual Income")

# ax.set_ylabel("Spending Score")

# ax.legend()

# # Show Plot in Streamlit
# st.pyplot(fig)



# st.subheader("Predict Customer Segment")

# st.sidebar.header("Customer Input")

# income = st.sidebar.slider(
#     "Annual Income",
#     0,
#     150,
#     50
# )

# score = st.sidebar.slider(
#     "Spending Score",
#     0,
#     100,
#     50
# )

# # Prediction
# new_data = pd.DataFrame({
#     'Annual Income (k$)': [income],
#     'Spending Score (1-100)': [score]
# })

# prediction = kmeans.predict(new_data)

# cluster_names = {
#     0: "Standard Customers",
#     1: "Premium Customers",
#     2: "Careful Customers",
#     3: "Budget Customers",
#     4: "High Spending Customers"
# }

# st.success(
#     f"Predicted Segment: {cluster_names[prediction[0]]}"
# )


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Segmentation App",
    layout="centered"
)

# =========================
# UI STYLING (CLEAN LOOK)
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 800px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🎯 Customer Segmentation App")
st.caption("K-Means Clustering based ML Project")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/Mall_Customers.csv")

# =========================
# DATA PREVIEW
# =========================
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================
# METRICS
# =========================
st.subheader("📌 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", df.shape[0])
col2.metric("Avg Income", round(df['Annual Income (k$)'].mean(), 2))
col3.metric("Avg Spending Score", round(df['Spending Score (1-100)'].mean(), 2))

# =========================
# FEATURE SELECTION
# =========================
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# =========================
# MODEL TRAINING
# =========================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# =========================
# CLUSTER VISUALIZATION
# =========================
st.subheader("📍 Customer Segments Visualization")

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y_kmeans,
    cmap='rainbow',
    alpha=0.7
)

# Centroids
ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=250,
    c='black',
    marker='X',
    label='Centroids'
)

ax.set_title("Customer Segments")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.legend()

st.pyplot(fig)

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.title("🎛️ Customer Input")
st.sidebar.markdown("Adjust values to predict segment")

income = st.sidebar.slider(
    "💰 Annual Income (k$)",
    0, 150, 50, step=5
)

score = st.sidebar.slider(
    "🛍️ Spending Score",
    0, 100, 50, step=1
)

# =========================
# PREDICTION
# =========================
new_data = pd.DataFrame({
    'Annual Income (k$)': [income],
    'Spending Score (1-100)': [score]
})

prediction = kmeans.predict(new_data)

cluster_names = {
    0: "Standard Customers",
    1: "Premium Customers",
    2: "Careful Customers",
    3: "Budget Customers",
    4: "High Spending Customers"
}

# =========================
# OUTPUT
# =========================
st.subheader("🎯 Prediction Result")

st.success(
    f"Customer belongs to: **{cluster_names[prediction[0]]}**"
)