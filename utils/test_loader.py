import torch
from torch.utils.data import DataLoader

import sys
import os

# 将当前脚本所在目录添加到搜索路径，确保能找到同级目录的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 假设你的 dataset 代码文件名叫 data_loader.py
from data_loader import Dataset_Custom

# 1. 定义一个简单的类来模拟 args 参数
class MockArgs:
    def __init__(self):
        self.use_closedllm = 0  # 0 表示使用 csv 中的 Final_Search_X 列
        self.text_len = 4       # 设置为 2，对应 CSV 中的 'Final_Search_2' 列
        # 如果你的 CSV 里想用 Final_Search_4，这里就改成 4

def main():
    # ================= 配置参数 =================
    args = MockArgs()
    root_path = 'data/Algriculture/'
    data_path = 'US_RetailBroilerComposite_Month.csv'
    
    # 设定序列长度：比如过去 24 个时间步，预测未来 12 个时间步
    # size = [seq_len, label_len, pred_len]
    seq_len = 24
    label_len = 12
    pred_len = 12
    size = [seq_len, label_len, pred_len]
    
    batch_size = 2  # 测试时用小一点的 batch size 方便观察
    
    print(f"Loading data from {root_path}{data_path} ...")

    # ================= 实例化 Dataset =================
    # 注意：features='S' (单变量) 或 'M' (多变量)，根据你的需求调整
    # timeenc=0 使用内部 pandas 处理时间，避免依赖外部 utils.timefeatures 报错
    dataset = Dataset_Custom(
        args=args,
        root_path=root_path,
        data_path=data_path,
        flag='train',        # 测试训练集模式
        size=size,
        features='S',        # 假设是单变量预测 (只预测 OT)
        target='OT',         # 你的 CSV 中必须有 'OT' 这一列
        scale=True,          # 进行标准化
        timeenc=0,           # 使用简单的 pandas 时间编码
        freq='m'             # 时间频率，虽然是月度数据，填 'h' 也没关系，主要影响 time_features
    )

    print(f"Dataset length: {len(dataset)}")

    # ================= 实例化 DataLoader =================
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,      # Windows 下建议设为 0，Linux 可设为 4
        drop_last=True
    )

    # ================= 迭代测试 =================
    # ================= Iteration Test =================
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(data_loader):
        
        print(f"\n====== Batch {i} Analysis ======")
        
        # 1. Shape Check
        print("--- Shape Info ---")
        print(f"Input X shape: {batch_x.shape}") 
        print(f"Target Y shape: {batch_y.shape}")
        
        # 2. Data Difference Check (Crucial for verifying shuffle)
        # We subtract sample 1 from sample 0. If result is 0, they are identical.
        diff = (batch_x[0] - batch_x[1]).abs().sum().item()
        print(f"\n--- Data Content Comparison (Sample 0 vs Sample 1) ---")
        print(f"Total numerical difference: {diff:.4f}")
        
        if diff != 0:
            print("Status: ✅ PASS (Samples contain different data)")
        else:
            print("Status: ❌ WARNING (Samples are identical! Check shuffle or index logic)")

        # 3. Text Feature Check
        print("\n--- Text Feature Comparison ---")
        text_batch = dataset.get_text(index)
        
        # Print the first 100 characters to verify content is different
        # Flatten [0] in case shape is (1,) to get the string
        txt_0 = str(text_batch[0]) 
        txt_1 = str(text_batch[1])
        
        print(f"Sample 0 Text: {txt_0[:100]}...") 
        print(f"Sample 1 Text: {txt_1[:100]}...")

        if txt_0 != txt_1:
             print("Status: ✅ PASS (Texts are different)")
        else:
             print("Status: ❌ WARNING (Texts are identical!)")

        # 4. Date Check
        start_dates, end_dates = dataset.get_date(index)
        print("\n--- Date Comparison (Last step in sequence) ---")
        
        # Compare the last date in the history window
        date_0 = start_dates[0][-1]
        date_1 = start_dates[1][-1]
        
        print(f"Sample 0 Date: {date_0}") 
        print(f"Sample 1 Date: {date_1}")
        
        if str(date_0) != str(date_1):
             print("Status: ✅ PASS (Dates are different)")
        else:
             print("Status: ❌ WARNING (Dates are identical!)")

        break # Stop after checking the first batch

if __name__ == '__main__':
    main()