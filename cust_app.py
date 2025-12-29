# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
import joblib
import streamlit as st

# Load the data
data = pd.read_csv("CC GENERAL.csv")

# Load the model
model_kmeans = joblib.load("Cust_seg_kmeans.pkl")

# Feature Selection
features = ['BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','CREDIT_LIMIT','PAYMENTS','PRC_FULL_PAYMENT','TENURE']
data_feat = data[features]

# Scaling the data
def scaling_data(data):
    # Log transformation
    data_log = np.log1p(data)
    # Standard Scaler
    scaler = StandardScaler()
    data_scale = scaler.fit_transform(data_log)
    # Turning to df
    scaled_df = pd.DataFrame(
        data_scale,
        columns=data_log.columns,
        index=data_log.index
    )
    scaled_df = scaled_df.dropna()
    return scaled_df

 # Clustering

scaled_data = scaling_data(data_feat)
cluster_labels = model_kmeans.predict(scaled_data)

data_cl = data_feat.dropna(subset=['CREDIT_LIMIT'])
data_cl["Cluster"] = cluster_labels 

# Building the interface
st.title('Customer Segmentation Model')
st.write('customer segmentation model using credit card usage behavior in order to support data-driven marketing strategies.')

#Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Segmentation", "Distributions"]
)

# Overview
if page == "Segmentation":
    # Show data sample
    st.subheader('Data Overview')
    st.dataframe(data_feat.head())

    # Button for segmentation
    if st.button('Run Segmentation'):
        #PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(scaled_data)

        data_cl['PC1'] = X_pca[:, 0]
        data_cl['PC2'] = X_pca[:, 1]
        
        # Layout
        #col1, col2 = st.columns([3, 1])
        # PCA plot
        st.subheader("PCA Cluster Visualisation")

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            data_cl["PC1"],
            data_cl["PC2"],
            c=data_cl["Cluster"],
            alpha=0.7
        )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Customer Segments (PCA)")
        plt.colorbar(scatter, ax=ax, label="Cluster")

        st.pyplot(fig)
            
        # Cluster summary
        st.subheader("Cluster Summary( Mean)")
        cluster_profile = data_cl.groupby('Cluster')[features].mean()
        st.dataframe(cluster_profile)
        # Bar graph for cluster Ditribution
        st.subheader("Cluster Distribution")
        cluster_counts = data_cl['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)

# Distribution
elif page == "Distributions":
    numeric_features = data_cl.select_dtypes("number").columns.drop("Cluster")
    feat_opt = ["All"] + numeric_features

    clusters = ["All"] + sorted(data_cl["Cluster"].unique().tolist())
    
    # Adding drop downs
    selector_col1, selector_col2 = st.columns([2, 1])

    with selector_col1:
        selected_feature = st.selectbox(
            "Select Feature",
            numeric_features
        )

    with selector_col2:
        selected_cluster = st.selectbox(
            "Select Cluster",
            clusters
        )
        
    if selected_cluster == "All":
        filtered_df = data_cl
    else:
        filtered_df = data_cl[data_cl["Cluster"] == selected_cluster]
       
    # Visualizations 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution")
        fig, ax = plt.subplots()
        sns.histplot(
            filtered_df[selected_feature],
            kde=True,
            ax=ax
        )
        ax.set_title(f"{selected_feature} | Cluster: {selected_cluster}")
        st.pyplot(fig)

    with col2:
        st.subheader("Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(
            x=filtered_df[selected_feature],
            ax=ax
        )
        st.pyplot(fig)
        
    # Summary Statistics
    st.subheader("Summary Statistics")

    stats = (
        filtered_df[selected_feature]
        .describe()[["mean", "50%", "std", "min", "max"]]
        .round(2)
    )

    st.dataframe(stats.rename({
        "mean": "Mean",
        "50%": "Median",
        "std": "Std Dev",
        "min": "Min",
        "max": "Max"
    }))




