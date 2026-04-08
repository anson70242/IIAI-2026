import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from sentence_transformers import SentenceTransformer

# 1. 定义 Domain 字典
domain_dict = {
    "Algriculture": ["US_RetailBroilerComposite_Month.csv", [6,8,10,12], 8, [2, 4, 6, 8]], 
    "Climate": ["US_precipitation_month.csv", [6,8,10,12], 8, [2, 4, 6, 8]], 
    "Economy": ["US_TradeBalance_Month.csv", [6,8,10,12], 8, [2, 4, 6, 8]],
    "Energy": ["US_GasolinePrice_Week.csv", [12, 24, 36, 48], 36, [6, 12, 24, 36]], 
    "Environment": ["NewYork_AQI_Day.csv", [48, 96, 192, 336], 96, [12, 24, 48, 96]], 
    "Public_Health": ["US_FLURATIO_Week.csv", [12, 24, 36, 48], 36, [6, 12, 24, 36]], 
    "Security": ["US_FEMAGrant_Month.csv", [6,8,10,12], 8, [2, 4, 6, 8]],
    "SocialGood": ["Unadj_UnemploymentRate_ALL_processed_fixed.csv", [6,8,10,12], 8, [2, 4, 6, 8]],
    "Traffic": ["US_VMT_Month.csv", [6,8,10,12], 8, [2, 4, 6, 8]],
}

# 2. 初始化模型与 Tokenizer，并确认设备
model_name = 'BAAI/bge-m3'
print(f"Loading Model & Tokenizer for {model_name}...")

device = "cpu"
print(f"Loading Model & Tokenizer for {model_name} on {device.upper()}...")
model = SentenceTransformer(model_name, device=device)
tokenizer = model.tokenizer

print("\n" + "="*40)
print(" [Model Configuration Info] ")
print("="*40)
print(f" >> Current Device: cpu (Mapped to physical GPU 1 due to env var)")
print(f" >> Max Input Tokens (Sequence Length): {model.max_seq_length}")
print(f" >> Output Embedding Dimension: {model.get_sentence_embedding_dimension()}")

# 提取并打印 Pooling 策略信息
pooling_layer = model._modules.get('1') # SentenceTransformer 的第 1 层通常是 Pooling
if pooling_layer is not None:
    print(f" >> Pooling Strategy: {pooling_layer}")
print("="*40 + "\n")


# 3. 统计变量
stats = {
    "Domain": [],
    "Min": [],
    "Max": [],
    "Mean": []
}

data_dir = "data" 
text_column = "Final_Search_4" # ⚠️ 请替换为你 CSV 中实际存在的文本列名，或在下方实现拼接逻辑

# 4. 遍历提取 Token 长度信息
print("Processing domains and calculating token lengths...")
for domain, values in domain_dict.items():
    csv_file = values[0]
    file_path = os.path.join(data_dir, domain, csv_file)
    
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件: {file_path}，跳过该 Domain ({domain})")
        continue
        
    try:
        df = pd.read_csv(file_path)
        
        # --- 文本获取逻辑 ---
        if text_column in df.columns:
            text_list = df[text_column].astype(str).tolist()
        else:
            # ⚠️ 在此添加你将纯数值转为文本的逻辑
            # text_list = [f"Value: {val}" for val in df.iloc[:, 1]]
            print(f"⚠️ 在 {csv_file} 中找不到列 '{text_column}'，请根据你的数据结构调整代码！")
            continue
            
        # 计算每个文本的 token 长度
        token_lengths = [len(tokenizer.encode(t)) for t in text_list]
        
        if len(token_lengths) == 0:
            print(f"Domain {domain} 的文本列表为空。")
            continue
            
        min_len = np.min(token_lengths)
        max_len = np.max(token_lengths)
        mean_len = np.mean(token_lengths)
        
        stats["Domain"].append(domain)
        stats["Min"].append(min_len)
        stats["Max"].append(max_len)
        stats["Mean"].append(mean_len)
        
        print(f"[{domain}] - Min: {min_len}, Max: {max_len}, Mean: {mean_len:.2f}")
        
    except Exception as e:
        print(f"处理 {domain} 时发生错误: {e}")

# 5. 绘图 (Plot)
if len(stats["Domain"]) > 0:
    x = np.arange(len(stats["Domain"]))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width, stats["Min"], width, label='Min', color='skyblue')
    rects2 = ax.bar(x, stats["Mean"], width, label='Average', color='orange')
    rects3 = ax.bar(x + width, stats["Max"], width, label='Max', color='tomato')

    ax.set_ylabel('Token Length')
    ax.set_xlabel('Domains')
    ax.set_title(f'Token Length Distribution')
    ax.set_xticks(x)
    
    # 🌟 修改了这里：移除了 rotation=45 和 ha="right"，让文字水平居中显示
    ax.set_xticklabels(stats["Domain"], rotation=0, ha="center") 
    
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.0f')
    ax.bar_label(rects2, padding=3, fmt='%.0f')
    ax.bar_label(rects3, padding=3, fmt='%.0f')

    fig.tight_layout()
    plt.savefig("token_distribution_plot.png", dpi=300)
    print("\n✅ 图表已保存为 'token_distribution_plot.png'")
else:
    print("没有收集到足够的数据来绘制图表。")