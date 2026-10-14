import pandas as pd
import os
import scipy
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

#cd AI_detection_models

df = pd.read_csv(os.path.join("data", "AI_writing.csv")) 
df = df.drop_duplicates().reset_index(drop=True)

x = df.drop(columns=["Is_AI_Assisted"])
y = df["Is_AI_Assisted"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=22, stratify=y
)

#The columns:
cata_columns = ["Submission_Type",
                "Academic_Level",
                "Primary_Language",
                "Font_Family"]

numerical_columns = ["Grammar_Errors",
                    "Word_Count",
                    "Revision_Count",
                    "Citation_Count",
                    "Submission_Hour"
                    ]

cont_columns = ["Average_Sentence_Length",
                "Vocabulary_Richness",
                "Passive_Voice_Ratio",
                "Reading_Level",
                "Editing_Time",
                "Typing_Speed",
                "Plagiarism_Score",
                "Punctuation_Density",
                "Sentence_Complexity",
                "time_per_revision",
                "error_speed_ratio"]


def create_ratio_columns(x):
    x = x.copy()
    x["time_per_revision"] = x["Editing_Time"] / (x["Revision_Count"] +1)
    x["error_speed_ratio"] = x["Typing_Speed"] / (x["Grammar_Errors"] +1)
    return x

ratio_transformer = FunctionTransformer(create_ratio_columns, validate=False)
cont_columns = cont_columns + ["time_per_revision", "error_speed_ratio"]


cata_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median") ),
    ("scaler", StandardScaler())
])

cont_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#1------------------------
preproccesor = ColumnTransformer(
    transformers=[
        ("cata", cata_pipe, cata_columns),
        ("num", num_pipe, numerical_columns),
        ("cont", cont_pipe, cont_columns)
    ]
)



#2-------------------------------

model = Pipeline([
    ("ratio_creator", ratio_transformer),
    ("preprocessor", preproccesor),
    ("classifier", LogisticRegression(max_iter= 2000))
])



model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(pd.Series(y_pred).value_counts())
#Very imbalanced , 0:1710 1:290



x_train_model, x_val, y_train_model, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=34,
    stratify=y_train
)

model.fit(x_train_model, y_train_model)
y_val_prob = model.predict_proba(x_val)[:, 1]


#---------------------
#Trying different thresholds

precisions = []
recalls = []
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
for threshold in thresholds:
    
    y_val_pred = (y_val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )
    precisions.append(precision)
    recalls.append(recall)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, label="Precision", marker="o", color="blue")
plt.plot(thresholds, recalls, label="Recall", marker="s", color="orange")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.savefig("thresholds.png", dpi=300, bbox_inches="tight")


#0.8 is the best threshold :|
#Highest precision, high recall

#===================================

model.fit(x_train, y_train)

y_test_prob = model.predict_proba(x_test)[:, 1]

chosen_threshold = 0.7
y_test_pred = (y_test_prob >= chosen_threshold).astype(int)

final_precision = precision_score(y_test, y_test_pred, zero_division=0)
final_recall = recall_score(y_test, y_test_pred, zero_division=0)

print("TEST RESULTS (Threshold = 0.8): ")
print(f"Precision: {final_precision:.4f}")
print(f"Recall:    {final_recall:.4f}")

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nFalse Positives (Innocent Students Flagged): {fp}")
print(f"True Positives (AI Submissions Caught):     {tp}")

# matrix 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
disp.plot(cmap="Blues")
plt.title(f"Test Set Confusion Matrix (Threshold = {chosen_threshold})")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
#=======================================



