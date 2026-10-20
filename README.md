 Is AI-Assisted? Data Exploration
 
Exploring a dataset of ~10,000 student writing submissions to see whether writing behavior can reveal AI assistance.
 
**Key finding:** the data separates *too* perfectly. Clustering and simple classifiers predict `Is_AI_Assisted` almost flawlessly, which points to a synthetic dataset with built-in bias rather than a real-world signal.
 
## Dataset
 
`data/AI_writing.csv` — one row per submission, with:
 
- **Behavioral features:** editing time, typing speed, revision count
- **Linguistic features:** word count, sentence length, grammar errors, vocabulary richness, reading level, etc.
- **Context:** submission type, academic level, primary language
- **Target:** `Is_AI_Assisted` (0 / 1)
## What's inside
 
| Folder | What it does |
|---|---|
| `Clusters/` | K-Means (k = 5) on scaled features + PCA to visualize writing-behavior profiles |
| `Logistic_Regression/` | Full pipeline (encoding, imputation, engineered ratio features) with confusion matrix and threshold analysis |
| `Models_compare/` | Compares models trained on behavioral vs. linguistic vs. combined features |
 
## Results
 
Writing-behavior clusters (purple = AI-assisted):
 
![Clusters](Clusters/student-writing-behavior-profiles.png)
 
The near-perfect separation is explained by a few features with extreme differences between classes:
 
![Feature boxplots](Models_compare/all_feature_boxplots.png)
 
 
## Takeaway
 
Perfect accuracy is a red flag. Here it exposed an unrealistic dataset.
