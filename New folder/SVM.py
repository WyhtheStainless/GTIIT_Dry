import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="用 SVM 对蛋白质嵌入进行二分类")
    parser.add_argument("--input", default="protein_embeddings.csv", help="包含嵌入和标签的CSV文件")
    parser.add_argument("--output_model", default="svm_protein_binder.pkl", help="保存模型的路径")
    parser.add_argument("--output_scaler", default="scaler.pkl", help="保存标准化器的路径")
    parser.add_argument("--predict", default=None, help="新数据CSV文件路径（用于预测，需包含相同格式的嵌入列）")
    parser.add_argument("--kernel", default="rbf", help="SVM核函数，建议 linear 或 rbf")
    parser.add_argument("--C", type=float, default=1.0, help="正则化参数")
    parser.add_argument("--gamma", default="scale", help="RBF核系数")
    parser.add_argument("--class_weight", default="balanced", help="类别权重处理不平衡")
    parser.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--no_train", action="store_true", help="跳过训练，仅用已有模型预测")
    args = parser.parse_args()

    # 1. 读取数据
    print(f"从 {args.input} 读取数据...")
    df = pd.read_csv(args.input)
    # 自动识别特征列
    embed_cols = [c for c in df.columns if c.startswith("embed_")]
    if not embed_cols:
        raise ValueError("未找到以 'embed_' 开头的特征列，请检查输入文件。")
    X = df[embed_cols].values
    y = df["label"].values
    print(f"数据形状: X={X.shape}, y={y.shape}, 正样本数: {np.sum(y==1)}, 负样本数: {np.sum(y==0)}")

    # ================= 预测模式 =================
    if args.no_train:
        if not args.predict:
            raise ValueError("预测模式需要提供 --predict 参数指定新数据文件。")
        print(f"加载模型: {args.output_model} 和标准化器: {args.output_scaler}")
        model = joblib.load(args.output_model)
        scaler = joblib.load(args.output_scaler)
        new_df = pd.read_csv(args.predict)
        X_new = new_df[embed_cols].values
        X_new_scaled = scaler.transform(X_new)
        preds = model.predict(X_new_scaled)
        probs = model.decision_function(X_new_scaled)
        # 输出预测结果
        results = new_df.copy()
        results["prediction"] = preds
        results["decision_score"] = probs
        out_name = args.predict.replace(".csv", "_predicted.csv")
        results.to_csv(out_name, index=False)
        print(f"预测结果已保存到 {out_name}")
        return

    # 2. 数据标准化（SVM对尺度敏感）
    print("数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 定义 SVM 模型
    svm = SVC(kernel=args.kernel, C=args.C, gamma=args.gamma,
              class_weight=args.class_weight, probability=True, random_state=42)

    # 4. 分层 K 折交叉验证评估
    print(f"\n正在进行 {args.cv} 折分层交叉验证...")
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    scores_accuracy = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
    scores_roc_auc = cross_val_score(svm, X_scaled, y, cv=cv, scoring='roc_auc')
    print(f"交叉验证准确率: {scores_accuracy.mean():.4f} ± {scores_accuracy.std():.4f}")
    print(f"交叉验证 ROC AUC: {scores_roc_auc.mean():.4f} ± {scores_roc_auc.std():.4f}")

    # 5. （可选）超参数网格搜索，优化模型
    print("\n进行简单网格搜索，优化 C 和 gamma...")
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    grid = GridSearchCV(SVC(class_weight=args.class_weight, probability=True, random_state=42),
                        param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_scaled, y)
    print(f"最佳参数: {grid.best_params_}, 最佳交叉验证 ROC AUC: {grid.best_score_:.4f}")

    # 6. 用最佳模型评估详细指标
    best_model = grid.best_estimator_
    # 在全部训练数据上做一次简单的留出验证（保持后续训练用全量数据）
    # 这里用交叉验证的结果已足够有说服力，但也可以展示一个分类报告（用训练集预测评估会有偏，但仅作参考）
    y_pred = best_model.predict(X_scaled)
    print("\n在训练集上的表现（仅供参考，不代表泛化能力）：")
    print(classification_report(y, y_pred, target_names=["阴性", "阳性"]))
    print("混淆矩阵:")
    print(confusion_matrix(y, y_pred))

    # 7. 在全部数据上训练最终模型并保存
    print(f"\n在全部数据上训练最终模型并保存到 {args.output_model} ...")
    final_model = SVC(**grid.best_params_, class_weight=args.class_weight,
                      probability=True, random_state=42)
    final_model.fit(X_scaled, y)
    joblib.dump(final_model, args.output_model)
    joblib.dump(scaler, args.output_scaler)
    print("模型与标准化器已保存。")

    # 8. 如果提供了新数据文件，直接预测
    if args.predict:
        print(f"对新数据 {args.predict} 进行预测...")
        new_df = pd.read_csv(args.predict)
        X_new = new_df[embed_cols].values
        X_new_scaled = scaler.transform(X_new)
        preds = final_model.predict(X_new_scaled)
        probs = final_model.decision_function(X_new_scaled)
        results = new_df.copy()
        results["prediction"] = preds
        results["decision_score"] = probs
        out_name = args.predict.replace(".csv", "_predicted.csv")
        results.to_csv(out_name, index=False)
        print(f"预测结果已保存到 {out_name}")


if __name__ == "__main__":
    main()