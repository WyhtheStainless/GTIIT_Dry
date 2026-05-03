import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载模型与标准化器
scaler = joblib.load("scaler.pkl")
model = joblib.load("svm_protein_binder.pkl")

# 读取新提取的嵌入（假设你已生成 peptide_single.csv）
df = pd.read_csv("protein_embeddings_test.csv")
embed_cols = [c for c in df.columns if c.startswith("embed_")]
X_new = df[embed_cols].values
X_scaled = scaler.transform(X_new)

pred = model.predict(X_scaled)[0]
score = model.decision_function(X_scaled)[0]
print(f"预测类别: {pred} (1=结合, 0=不结合)")
print(f"决策分数: {score:.4f} (正值越大越倾向阳性)")