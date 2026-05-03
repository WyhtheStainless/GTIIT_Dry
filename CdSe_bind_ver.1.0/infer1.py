import os
os.environ["PYTHONUTF8"] = "1"

import argparse
import pandas as pd
import numpy as np
import torch
from transformers import EsmTokenizer, EsmModel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="用ESM-2提取蛋白质嵌入，合并正负样本输出为CSV")
    parser.add_argument("--pos_file", default="positive.csv", help="阳性样本CSV文件路径")
    parser.add_argument("--neg_file", default="negative.csv", help="阴性样本CSV文件路径")
    parser.add_argument("--seq_col", default="Sequence", help="CSV中蛋白质序列的列名（默认: Sequence）")
    parser.add_argument("--id_col", default=None, help="CSV中蛋白质ID/名称的列名（如不指定则自动生成序号）")
    parser.add_argument("--out_file", default="protein_embeddings.csv", help="输出CSV文件路径")
    parser.add_argument("--batch_size", type=int, default=8, help="推理时的批量大小（根据显存调整）")
    parser.add_argument("--model_name", default="esm2_t36_3B_UR50D", 
                        help="ESM-2模型名称（可选650M/150M/3B等）")
    args = parser.parse_args()

    # 1. 读取数据并打标签
    print("正在读取阳性样本...")
    df_pos = pd.read_csv(args.pos_file)
    df_pos["label"] = 1   # 阳性标记为1
    
    print("正在读取阴性样本...")
    df_neg = pd.read_csv(args.neg_file)
    df_neg["label"] = 0   # 阴性标记为0

    # 检查序列列是否存在
    if args.seq_col not in df_pos.columns:
        raise ValueError(f"阳性文件中未找到序列列 '{args.seq_col}'，可用列: {list(df_pos.columns)}")
    if args.seq_col not in df_neg.columns:
        raise ValueError(f"阴性文件中未找到序列列 '{args.seq_col}'，可用列: {list(df_neg.columns)}")

    # 合并数据
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    print(f"共合并 {len(df)} 条蛋白质序列（阳性: {len(df_pos)}，阴性: {len(df_neg)}）")

    # 处理ID列
    if args.id_col and args.id_col in df.columns:
        protein_ids = df[args.id_col].tolist()
    else:
        protein_ids = [f"protein_{i}" for i in range(len(df))]
        print("未指定有效ID列，已自动生成 protein_0, protein_1, ... 作为标识")

    sequences = df[args.seq_col].tolist()
    labels = df["label"].tolist()

    # 2. 加载模型
    print(f"正在加载模型 {args.model_name} ...")
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    model = EsmModel.from_pretrained(args.model_name)

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"模型已加载至 {device}")

    # 3. 批量提取嵌入（整蛋白的平均池化）
    embeddings = []
    print("开始提取嵌入向量...")
    for i in tqdm(range(0, len(sequences), args.batch_size)):
        batch_seqs = sequences[i:i+args.batch_size]
        # Tokenize，注意ESM-2最大长度为1024，超出部分截断（也可用max_length=None查看警告）
        inputs = tokenizer(
            batch_seqs, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024, 
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # 取最后一层隐藏状态，在序列长度维度上求均值（去除padding的影响）
            # attention_mask用于正确处理padding
            hidden = outputs.last_hidden_state  # shape: (batch, seq_len, hidden_dim)
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
            masked_hidden = hidden * mask
            # 均值池化，得到每条序列的整蛋白嵌入
            mean_emb = masked_hidden.sum(dim=1) / mask.sum(dim=1)
            embeddings.append(mean_emb.cpu().numpy())

    # 4. 拼接所有嵌入
    embedding_matrix = np.concatenate(embeddings, axis=0)  # shape: (num_samples, 1280)
    embed_dim = embedding_matrix.shape[1]
    print(f"嵌入提取完成，维度: {embedding_matrix.shape}")

    # 5. 构建输出DataFrame
    # 将嵌入向量拆成独立列：embed_0, embed_1, ..., embed_1279
    embed_df = pd.DataFrame(
        embedding_matrix,
        columns=[f"embed_{i}" for i in range(embed_dim)]
    )
    # 加入ID、序列和标签
    result_df = pd.DataFrame({
        "protein_id": protein_ids,
        "sequence": sequences,
        "label": labels
    })
    result_df = pd.concat([result_df, embed_df], axis=1)

    # 6. 保存到CSV
    result_df.to_csv(args.out_file, index=False)
    print(f"结果已保存至 {args.out_file}")
    print(f"输出文件形状: {result_df.shape[0]} 行 × {result_df.shape[1]} 列")

if __name__ == "__main__":
    main()