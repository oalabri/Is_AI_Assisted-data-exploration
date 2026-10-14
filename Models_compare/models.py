import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



df = pd.read_csv(os.path.join("data", "AI_writing.csv"))

Behavioral_columns = [
    "Editing_Time",
    "Typing_Speed",
    "Revision_Count"
]

Linguistic_columns = [
    "Word_Count",
    "Average_Sentence_Length",
    "Grammar_Errors",
    "Vocabulary_Richness",
    "Passive_Voice_Ratio",
    "Reading_Level",
    "Punctuation_Density",
    "Sentence_Complexity"
]

x_combined = Behavioral_columns + Linguistic_columns
X = df[x_combined]
y = df["Is_AI_Assisted"]

#-------------------------
#The three models

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=12,
    stratify=y
)

Xb_train = x_train[Behavioral_columns]
Xb_test = x_test[Behavioral_columns]

Xl_train = x_train[Linguistic_columns]
Xl_test = x_test[Linguistic_columns]

Xc_train = x_train[x_combined]
Xc_test = x_test[x_combined]

b_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classification", LogisticRegression(class_weight="balanced"))

])

l_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classification", LogisticRegression(class_weight="balanced"))
    
])

c_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classification", LogisticRegression(class_weight="balanced"))

])


#-----------------
b_model.fit(Xb_train, y_train)
l_model.fit(Xl_train, y_train)
c_model.fit(Xc_train, y_train)

b_pred = b_model.predict(Xb_test)
l_pred = l_model.predict(Xl_test)
c_pred = c_model.predict(Xc_test)

b_prob = b_model.predict_proba(Xb_test)[:, 1]
l_prob = l_model.predict_proba(Xl_test)[:, 1]
c_prob = c_model.predict_proba(Xc_test)[:, 1]

#--------------------------
#Compare metrics, F1, ROC, PR
#We care about both false positives and false negatives here.
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

results = {
    "Behavioral": [
        f1_score(y_test,b_pred),
        roc_auc_score(y_test,b_prob),
        average_precision_score(y_test,b_prob)
    ],

    "Linguistic": [
        f1_score(y_test,l_pred),
        roc_auc_score(y_test,l_prob),
        average_precision_score(y_test,l_prob)
    ],

    "Combined": [
        f1_score(y_test,c_pred),
        roc_auc_score(y_test,c_prob),
        average_precision_score(y_test,c_prob)
    ]
}







results_df = pd.DataFrame(
    results,
    index=["F1", "ROC-AUC", "PR-AUC"]
)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["F1", "ROC-AUC", "PR-AUC"]

for ax, metric in zip(axes, metrics):
    scores = results_df.loc[metric]

    bars = ax.bar(scores.index, scores.values)

    ax.set_title(metric)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")

    for bar, value in zip(bars, scores.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center"
        )

fig.suptitle(
    "Behavioral vs. Linguistic Models",
    fontsize=16
)

plt.tight_layout()
plt.savefig("Models_metrics.png", dpi=300, bbox_inches="tight")






df_plot = df.copy()
df_plot["Is_AI_Assisted"] = df_plot["Is_AI_Assisted"].map({
    0: "No",
    1: "Yes"
})

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(x_combined):

    df_plot.boxplot(
        column=feature,
        by="Is_AI_Assisted",
        ax=axes[i]
    )

    axes[i].set_title(feature)
    axes[i].set_xlabel("Is AI Assisted")
    axes[i].set_ylabel(feature)

fig.delaxes(axes[-1])

plt.suptitle("Explore their box plots", fontsize=16)
plt.tight_layout()

plt.savefig("all_feature_boxplots.png", dpi=300, bbox_inches="tight")

