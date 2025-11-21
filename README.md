# EDA_EXP_6

**Aim**

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

**Algorithm**

1. Import pandas, numpy, seaborn, matplotlib, and sklearn libraries.

2. Load the Wine Quality dataset and perform basic EDA.

3. Detect and remove outliers using the IQR method.

4. Train and evaluate Logistic Regression before outlier removal.

5. Train and evaluate Logistic Regression after outlier removal and compare results.

**Program and Output**

***NAME : Rishab p doshi***

***REG NO : 212224240134***

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")
REMOVE_MODE = 'all'

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
print("Loaded rows,cols:", df.shape)
```
<img width="296" height="36" alt="image" src="https://github.com/user-attachments/assets/07f5865a-d327-44ea-a426-f1e671ccdd57" />

```
df['good_wine'] = (df['quality'] >= 7).astype(int)
print("\n--- Quick head & shape ---")
display(df.head())
print("Shape:", df.shape)
```
<img width="1729" height="335" alt="image" src="https://github.com/user-attachments/assets/7e3144a3-8c8d-4c04-ba04-2740ad3679be" />

```
# Univariate
brief_feats = ['alcohol', 'volatile acidity', 'pH']
print("\n--- Quick stats (mean,median,std) for key features ---")
print(df[brief_feats].agg(['mean','median','std']).round(3))
```

<img width="650" height="119" alt="image" src="https://github.com/user-attachments/assets/c42cb23b-856f-4671-aaa9-69ec3526a083" />

```
plt.figure(figsize=(9,3))
for i, col in enumerate(brief_feats, 1):
    plt.subplot(1,3,i)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(col)
plt.tight_layout()
plt.show()
```

<img width="748" height="236" alt="image" src="https://github.com/user-attachments/assets/ac77cd3a-5fa8-4684-a00e-694a7ad359ce" />

```
# Bivariate
plt.figure(figsize=(8,3))
for i, col in enumerate(brief_feats[:2], 1):
    plt.subplot(1,2,i)
    sns.boxplot(x='quality', y=col, data=df)
    plt.title(f"{col} vs quality")
plt.tight_layout()
plt.show()
```
<img width="726" height="275" alt="image" src="https://github.com/user-attachments/assets/50e7e114-8172-45c0-86c7-e0d4f30912a2" />

```
# Multivariate
print("\n--- Correlation (selected features + quality) ---")
print(df[brief_feats + ['quality']].corr().round(3))
```
<img width="667" height="165" alt="image" src="https://github.com/user-attachments/assets/983a743d-7120-4f96-a658-f2271374c9db" />

```
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='alcohol', y='volatile acidity', hue='quality', palette='viridis', s=25)
plt.title('alcohol vs volatile acidity (color=quality)')
plt.show()
```
<img width="637" height="399" alt="image" src="https://github.com/user-attachments/assets/dd8b0694-b5a3-434f-939d-178be0a1a13a" />


```
# FULL OUTLIER DETECTION (IQR) - ALL numeric features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# exclude target columns from detection
exclude = ['quality','good_wine']
num_features = [c for c in num_cols if c not in exclude]

print("\nRunning IQR outlier detection on every numeric feature...")
outlier_summary = []
```
<img width="644" height="48" alt="image" src="https://github.com/user-attachments/assets/4dbdbab0-bb1e-4814-a374-ae0730c5aced" />

```
for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    cnt = int(mask.sum())
    pct = cnt / df.shape[0] * 100
    outlier_summary.append({
        'feature': col,
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'lower': lower, 'upper': upper,
        'outlier_count': cnt,
        'outlier_pct': pct
    })

out_df = pd.DataFrame(outlier_summary).sort_values(by='outlier_count', ascending=False)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
print("\n--- Outlier summary (feature, count, %): ---")
display(out_df[['feature','outlier_count','outlier_pct']])

```
<img width="655" height="528" alt="image" src="https://github.com/user-attachments/assets/c1ce1918-47d0-4c7b-bea5-5dd3e5c5f22d" />

```
# Top features by outlier count
top_k = min(6, len(out_df))
top_features = out_df.head(top_k)['feature'].tolist()
print(f"\nTop {top_k} features with most outliers: {top_features}")
```
<img width="1340" height="60" alt="image" src="https://github.com/user-attachments/assets/5b553aa8-6944-4e2e-9cc2-2a1ea0a187a2" />

```
#VISUALIZE OUTLIERS (boxplots for top features)
plt.figure(figsize=(12, 2*len(top_features)))
for i, col in enumerate(top_features, 1):
    plt.subplot(len(top_features), 1, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot — {col} (outliers beyond whiskers)")
plt.tight_layout()
plt.show()
```
<img width="803" height="624" alt="image" src="https://github.com/user-attachments/assets/b7f9a049-cf0e-4de1-9598-5098ca774416" />

```
print("\nExample outlier rows for top features (up to 5 rows each):")
for col in top_features:
    row = out_df.loc[out_df['feature'] == col].iloc[0]
    lower, upper = float(row['lower']), float(row['upper'])
    mask = (df[col] < lower) | (df[col] > upper)
    if mask.sum() > 0:
        print(f"\n--- {col} : {mask.sum()} outliers (showing up to 5) ---")
        display(df[mask].head(5))
    else:
        print(f"\n--- {col} : no outliers by IQR rule ---")
```
<img width="947" height="625" alt="image" src="https://github.com/user-attachments/assets/4ffaaacb-0ad4-4896-9ac8-894b1d512346" />

```
# MODEL: BEFORE OUTLIER REMOVAL

selected_features = ['alcohol','volatile acidity','pH']  # features used for modeling comparison
X = df[selected_features]
y = df['good_wine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_before = LogisticRegression(max_iter=1000)
model_before.fit(X_train, y_train)
pred_before = model_before.predict(X_test)

acc_before = accuracy_score(y_test, pred_before)
print("\n--- Model BEFORE outlier removal ---")
print("Rows used:", df.shape[0])
print("Accuracy:", round(acc_before,4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_before))
print(classification_report(y_test, pred_before, digits=4))
```
<img width="601" height="306" alt="image" src="https://github.com/user-attachments/assets/dfd5c266-fcef-4385-88ac-0b96710eef7b" />

```
def remove_outliers_any_feature(df_in, features_list):
    df_tmp = df_in.copy()
    remove_mask = np.zeros(len(df_tmp), dtype=bool)
    for col in features_list:
        row = out_df[out_df['feature'] == col]
        if row.empty:
            continue
        lower = float(row['lower']); upper = float(row['upper'])
        remove_mask = remove_mask | ((df_tmp[col] < lower) | (df_tmp[col] > upper))
    return df_tmp.loc[~remove_mask].reset_index(drop=True), int(remove_mask.sum())

# attempt removal according to REMOVE_MODE
if REMOVE_MODE == 'all':
    df_removed, removed_count = remove_outliers_any_feature(df, num_features)
    print(f"\nAttempting aggressive removal on ALL numeric features -> rows removed: {removed_count}")
    # If too many removed or classes broken, fallback
    if df_removed.shape[0] < 100 or df_removed['good_wine'].nunique() < 2:
        print("Warning: aggressive removal removed too many rows or left only one class. Falling back to 'focused' removal.")
        df_removed, removed_count = remove_outliers_any_feature(df, selected_features)
        print(f"Focused removal on {selected_features} -> rows removed: {removed_count}")
        removal_mode_used = 'focused (fallback from all)'
    else:
        removal_mode_used = 'all'
else:
    df_removed, removed_count = remove_outliers_any_feature(df, selected_features)
    print(f"\nFocused removal on {selected_features} -> rows removed: {removed_count}")
    removal_mode_used = 'focused'

print("Rows after removal:", df_removed.shape[0])
```
<img width="637" height="399" alt="image" src="https://github.com/user-attachments/assets/0a79a17a-2794-4526-a525-8bc1e55a5a0c" />


```
#QUICK BEFORE vs AFTER CHECK (compact)
plt.figure(figsize=(9,6))
for i, col in enumerate(selected_features,1):
    plt.subplot(3,2,2*i-1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"{col} - original")
    plt.subplot(3,2,2*i)
    sns.histplot(df_removed[col], kde=True, bins=20)
    plt.title(f"{col} - after removal")
plt.tight_layout()
plt.show()
```
<img width="930" height="593" alt="image" src="https://github.com/user-attachments/assets/884d4889-4698-4577-8fee-17f87eb6301c" />

```
means_before = df[selected_features].mean().round(3)
means_after  = df_removed[selected_features].mean().round(3)
print("\nFeature means before vs after (selected features):")
display(pd.DataFrame({'before_mean': means_before, 'after_mean': means_after}))
```
<img width="641" height="251" alt="image" src="https://github.com/user-attachments/assets/caecfed3-c5d5-47f4-b419-88a7f1978f5a" />

```
#MODEL: AFTER OUTLIER REMOVAL
X2 = df_removed[selected_features]
y2 = df_removed['good_wine']

if y2.nunique() < 2 or df_removed.shape[0] < 30:
    print("\nAfter removal dataset is too small or single-class — cannot train reliable model. Stopping after showing removal info.")
else:
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
    model_after = LogisticRegression(max_iter=1000)
    model_after.fit(X_train2, y_train2)
    pred_after = model_after.predict(X_test2)

    acc_after = accuracy_score(y_test2, pred_after)
    print("\n--- Model AFTER outlier removal ---")
    print("Removal mode used:", removal_mode_used)
    print("Rows used:", df_removed.shape[0])
    print("Accuracy:", round(acc_after,4))
    print("Confusion Matrix:\n", confusion_matrix(y_test2, pred_after))
    print(classification_report(y_test2, pred_after, digits=4))
```
<img width="781" height="478" alt="image" src="https://github.com/user-attachments/assets/598fdd56-548d-4a25-bda7-ea004f728b01" />

```
# COMPARISON SUMMARY
print(f"Accuracy BEFORE: {acc_before:.4f}")
print(f"Accuracy AFTER : {acc_after:.4f}")
if acc_after > acc_before:
    print("-> Accuracy improved after outlier removal.")
elif acc_after < acc_before:
    print("-> Accuracy decreased after outlier removal.")
else:
    print("-> Accuracy unchanged after outlier removal.")
```
<img width="772" height="118" alt="image" src="https://github.com/user-attachments/assets/47c871c3-bf71-4f6e-a5a7-3317dbced769" />



**Result**

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset was successful.
