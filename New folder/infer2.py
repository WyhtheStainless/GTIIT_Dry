import argparse
import pandas as pd
import numpy as np
import torch
import esm
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="使用 fair-esm 提取蛋白质嵌入 (ESM-2)")
    parser.add_argument("--pos_file", default="positive.csv", help="阳性样本CSV文件路径")
    parser.add_argument("--neg_file", default="negative.csv", help="阴性样本CSV文件路径")
    parser.add_argument("--seq_col", default="Sequence")
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--out_file", default="protein_embeddings.csv")
    parser.add_argument("--batch_size", type=int, default=2, help="3B模型非常大，建议1-2")
    parser.add_argument("--model_path", default="esm2_t36_3B_UR50D.pt", help="本地模型文件路径")
    parser.add_argument("--repr_layer", type=int, default=36, help="提取哪一层的表示，默认最后一层")
    args = parser.parse_args()

    # 1. 读取并合并数据
    df_pos = pd.read_csv(args.pos_file)
    df_pos["label"] = 1
    df_neg = pd.read_csv(args.neg_file)
    df_neg["label"] = 0
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    sequences = df[args.seq_col].tolist()
    labels = df["label"].tolist()
    if args.id_col and args.id_col in df.columns:
        ids = df[args.id_col].tolist()
    else:
        ids = [f"protein_{i}" for i in range(len(df))]

    # 2. 加载模型（加载本地 .pt 文件）
    print(f"正在加载模型 {args.model_path} ...")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.model_path)
    model = model.eval()
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model = model.to(device)
    print(f"模型已加载至 {device}")

    batch_converter = alphabet.get_batch_converter()

    # 3. 批量提取嵌入
    embeddings = []
    print("正在提取嵌入向量...")
    for i in tqdm(range(0, len(sequences), args.batch_size)):
        batch_seqs = sequences[i:i+args.batch_size]
        # fair-esm 要求输入为 [(id, seq), ...]
        batch_data = [(f"{j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[args.repr_layer])
        # 提取指定层的残基表示
        token_embeddings = results["representations"][args.repr_layer]  # (batch, seq_len+2, dim)
        # 去掉 BOS/EOS 标记，取实际序列部分
        seq_embeddings = token_embeddings[:, 1:-1, :]  # (batch, seq_len, dim)
        # 整蛋白平均池化
        mean_emb = seq_embeddings.mean(dim=1).cpu().numpy()
        embeddings.append(mean_emb)

    embedding_matrix = np.concatenate(embeddings, axis=0)
    embed_dim = embedding_matrix.shape[1]
    print(f"嵌入提取完成，维度: {embedding_matrix.shape}")

    # 4. 输出为 CSV
    embed_df = pd.DataFrame(embedding_matrix, columns=[f"embed_{i}" for i in range(embed_dim)])
    result_df = pd.DataFrame({"protein_id": ids, "sequence": sequences, "label": labels})
    result_df = pd.concat([result_df, embed_df], axis=1)
    result_df.to_csv(args.out_file, index=False)
    print(f"结果已保存至 {args.out_file}")

if __name__ == "__main__":
    main()