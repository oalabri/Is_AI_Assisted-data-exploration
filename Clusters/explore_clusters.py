from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import plotly.express as px




df = pd.read_csv(os.path.join("data", "AI_writing.csv"))

features = [
    "Word_Count",
    "Average_Sentence_Length",
    "Grammar_Errors",
    "Vocabulary_Richness",
    "Passive_Voice_Ratio",
    "Reading_Level",
    "Editing_Time",
    "Revision_Count",
    "Typing_Speed",
    "Punctuation_Density",
    "Sentence_Complexity"
]


x = df[features]

scaler = StandardScaler()
imputer = SimpleImputer(strategy="median")
x_imputed = imputer.fit_transform(x)
x_scaled = scaler.fit_transform(x_imputed)


inertias = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=12, n_init=10)
    kmeans.fit(x_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(2, 9), inertias, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.savefig("where_elbow.png", dpi=300, bbox_inches="tight")

#I will use k = 5 

kmeans = KMeans(n_clusters=5, random_state=12, n_init=10)

df["Cluster"] = kmeans.fit_predict(x_scaled)


pca = PCA(n_components=3) # 3d :0

x_pca = pca.fit_transform(x_scaled)

df["PC1"] = x_pca[:, 0]
df["PC2"] = x_pca[:, 1]
df["PC3"] = x_pca[:, 2]

#plotting 


fig = px.scatter_3d(
    df,
    x="PC1",
    y="PC2",
    z="PC3",
    color="Cluster",
    hover_data=[
        "Word_Count",
        "Grammar_Errors",
        "Vocabulary_Richness",
        "Editing_Time",
        "Revision_Count",
        "Typing_Speed",
        "Is_AI_Assisted"
    ]
)

#fig.show()


fig = px.scatter_3d(
    df,
    x="PC1",
    y="PC2",
    z="PC3",
    color="Is_AI_Assisted",
    hover_data=[
        "Cluster",
        "Word_Count",
        "Grammar_Errors",
        "Vocabulary_Richness",
        "Editing_Time",
        "Revision_Count",
        "Typing_Speed"
    ]
)

fig.show()

cluster_ai_rate = (
    df.groupby("Cluster")["Is_AI_Assisted"]
    .mean()
    .sort_index()
)
print(cluster_ai_rate)


#-------------Explore the columns
cluster_profile = df.groupby("Cluster")[features].mean()
print(cluster_profile.round(2))




scaled_df = pd.DataFrame(
    x_scaled,
    columns=features,
    index=df.index
)

scaled_df["Cluster"] = df["Cluster"]

cluster_profile_scaled = scaled_df.groupby("Cluster").mean()

fig, axes = plt.subplots(
    nrows=len(cluster_profile_scaled),
    ncols=1,
    figsize=(14, 4 * len(cluster_profile_scaled))
)

for i, (cluster, values) in enumerate(cluster_profile_scaled.iterrows()):

    axes[i].bar(values.index, values.values)

    axes[i].axhline(0, linewidth=1)

    axes[i].set_title(f"Cluster {cluster}")
    axes[i].set_ylabel("Standardized Averages")
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("clusters_features.png", dpi=300, bbox_inches="tight")



