import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from types import SimpleNamespace

# 导入你的模块
from utils.data_loader import Dataset_Custom
from models.LeMoLE import Model as LeMoLEModel
from trainer import LeMoLETrainerWithES

# Set random seed
seed = 3407
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

# ==========================================
# 0. 参数冻结控制函数
# ==========================================
def set_training_stage(model, stage="stage1"):
    """
    针对 LeMoLE 模型设计的两阶段参数冻结控制
    """
    if stage == "stage1":
        print("--> [STAGE 1] Activating: Numerical Backbone | Freezing: Projectors, Mixer, SBERT")
        for name, param in model.named_parameters():
            if "numerical_backbone" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    elif stage == "stage2":
        print("--> [STAGE 2] Activating: Projectors, Mixer | Freezing: Numerical Backbone, SBERT")
        for name, param in model.named_parameters():
            if "text_encoder" in name or "numerical_backbone" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

# ==========================================
# 1. DataLoader 包装器
# ==========================================
class TextDataLoaderWrapper:
    def __init__(self, dataloader, dataset):
        self.dataloader = dataloader
        self.dataset = dataset

    def __iter__(self):
        for batch_x, batch_y, batch_x_mark, batch_y_mark, index in self.dataloader:
            dynamic_text = self.dataset.get_text(index).flatten().tolist()
            dynamic_text = [str(t) for t in dynamic_text]
            
            batch_size = batch_x.size(0)
            static_text = [""] * batch_size 
            
            yield batch_x, batch_y, static_text, dynamic_text

    def __len__(self):
        return len(self.dataloader)

# ==========================================
# 2. 测试集结果储存函数
# ==========================================
def save_model_predictions(model, test_loader, device, pred_len, save_dir, stage_name, timemmd=False):
    """
    运行测试集并将预测结果和真实值保存为 .npy 文件，用于后续可视化
    """
    model.eval()
    preds = []
    trues = []
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with torch.no_grad():
        for batch_x, batch_y, static_text, dynamic_text in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x, static_text, dynamic_text, timemmd=timemmd)
            
            # 截取预测长度部分
            outputs = outputs[:, -pred_len:, :]
            batch_y = batch_y[:, -pred_len:, :]
            
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            
    # 拼接所有 batch 的结果
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 保存为 numpy 数组
    np.save(os.path.join(save_dir, f'{stage_name}_preds.npy'), preds)
    np.save(os.path.join(save_dir, f'{stage_name}_trues.npy'), trues)
    print(f"--> [Visual Data] Saved {stage_name} predictions to {save_dir}/ (Shape: {preds.shape})")

# ==========================================
# 3. 主函数
# ==========================================
def main():
    # 建立 output_table 文件夹
    output_dir = "./output_table"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 包含所有领域的配置
    domain_dict = {
        # Domain: [csv_file, [pred_len_list], seq_len, window_list]
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

    # 基础配置，部分内容会在循环中被动态覆盖
    base_config = {
        "task_name": "long_term_forecast",
        "label_len": 0,
        "enc_in": 1,               
        "moving_avg": 25,          
        "smoothing": False,        
        "features": "S", 
        "target": "OT",
        "use_closedllm": 0,        
        "text_len": 4,             
        "batch_size": 16,
        "learning_rate": 5e-4,
        "epochs": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    # 遍历所有 Domain
    for domain_name, domain_info in domain_dict.items():
        csv_file, pred_len_list, seq_len, window_list = domain_info
        
        print("\n" + "*"*80)
        print(f"*** STARTING DOMAIN: {domain_name} ***")
        print(f"*** DATA: {csv_file} ***")
        print("*"*80)
        
        results = {}
        
        # 针对当前 Domain 动态修改 config
        domain_config = base_config.copy()
        domain_config["root_path"] = f"./data/{domain_name}/"  # 假设你的文件夹结构是 ./data/Domain/
        domain_config["data_path"] = csv_file
        domain_config["seq_len"] = seq_len
        domain_config["window_sizes"] = window_list

        # 遍历当前 Domain 下的所有预测长度
        for p_len in pred_len_list:
            print("\n" + "="*60)
            print(f" Start Training - Domain: {domain_name} | Pred Len: {p_len} ")
            print("="*60)
            
            current_config = domain_config.copy()
            current_config["pred_len"] = p_len
            args = SimpleNamespace(**current_config)
            
            # 定义两个阶段的保存路径 (加入了 domain_name)
            save_path_s1 = f"./checkpoints/LeMoLE_{domain_name}_pl{p_len}_Stage1"
            save_path_s2 = f"./checkpoints/LeMoLE_{domain_name}_pl{p_len}_Stage2"
            
            size = [args.seq_len, args.label_len, args.pred_len]
            
            # --- 准备数据 ---
            train_dataset = Dataset_Custom(args=args, root_path=args.root_path, data_path=args.data_path, flag='train', size=size, features='S', target='OT', timeenc=0)
            val_dataset = Dataset_Custom(args=args, root_path=args.root_path, data_path=args.data_path, flag='val', size=size, features='S', target='OT', timeenc=0)
            test_dataset = Dataset_Custom(args=args, root_path=args.root_path, data_path=args.data_path, flag='test', size=size, features='S', target='OT', timeenc=0)

            train_loader_raw = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
            val_loader_raw = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
            test_loader_raw = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

            train_loader = TextDataLoaderWrapper(train_loader_raw, train_dataset)
            val_loader = TextDataLoaderWrapper(val_loader_raw, val_dataset)
            test_loader = TextDataLoaderWrapper(test_loader_raw, test_dataset)

            # 初始化模型与 Scaler
            model = LeMoLEModel(args)
            scaler = train_dataset.scaler

            # ==========================================================
            # STAGE 1: 训练数值模型 (Numerical Backbone)
            # ==========================================================
            print(f"\n--- [STAGE 1] Training Numerical Backbone for {domain_name} (pl={p_len}) ---")
            set_training_stage(model, stage="stage1")
            
            trainer_s1 = LeMoLETrainerWithES(
                model=model, 
                config=args, 
                device=args.device, 
                learning_rate=args.learning_rate, 
                scaler=scaler
            )
            trainer_s1.skip_fusion = True  
            trainer_s1.build_optimizer()   
            
            best_t_loss1, best_v_loss1, te_loss1 = trainer_s1.train_with_es(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                patience=10,
                save_path=save_path_s1
            )
            
            # 保存 Stage 1 的最优模型测试集结果
            print("--> Extracting Stage 1 visual data...")
            model.load_state_dict(torch.load(f"{save_path_s1}/lemole_best.pth"))
            save_model_predictions(model, test_loader, args.device, p_len, save_path_s1, stage_name="stage1")

            # ==========================================================
            # STAGE 2: 训练融合层 (Fusion Layer)
            # ==========================================================
            print(f"\n--- [STAGE 2] Training Fusion Layer for {domain_name} (pl={p_len}) ---")
            print(f"Loading best Stage 1 model from {save_path_s1}...")
            model.load_state_dict(torch.load(f"{save_path_s1}/lemole_best.pth"))
            
            set_training_stage(model, stage="stage2")
            
            lr_stage2 = args.learning_rate * 0.5
            trainer_s2 = LeMoLETrainerWithES(
                model=model, 
                config=args, 
                device=args.device, 
                learning_rate=lr_stage2, 
                scaler=scaler
            )
            trainer_s2.skip_fusion = False 
            trainer_s2.timemmd = False
            trainer_s2.build_optimizer()   
            
            best_t_loss2, best_v_loss2, te_loss2 = trainer_s2.train_with_es(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                patience=10,
                save_path=save_path_s2
            )
            
            # 保存 Stage 2 的最优模型测试集结果
            print("--> Extracting Stage 2 visual data...")
            model.load_state_dict(torch.load(f"{save_path_s2}/lemole_best.pth"))
            save_model_predictions(model, test_loader, args.device, p_len, save_path_s2, stage_name="stage2")
            
            # 记录当前长度的结果
            results[p_len] = {
                'stage1': {'train': best_t_loss1, 'val': best_v_loss1, 'test': te_loss1},
                'stage2': {'train': best_t_loss2, 'val': best_v_loss2, 'test': te_loss2}
            }

        # ==========================================================
        # 生成并保存单个 Domain 的最终报告
        # ==========================================================
        report_lines = []
        report_lines.append("#"*90)
        report_lines.append(f" FINAL REPORT FOR DOMAIN: {domain_name} (STAGE 1 & STAGE 2)".center(90))
        report_lines.append("#"*90)
        report_lines.append(f"{'Pred_Len':<10} | {'S1 Train':<10} | {'S1 Val':<10} | {'S1 Test':<10} || {'S2 Train':<10} | {'S2 Val':<10} | {'S2 Test':<10}")
        report_lines.append("-" * 90)
        
        s1_train_losses, s1_val_losses, s1_test_losses = [], [], []
        s2_train_losses, s2_val_losses, s2_test_losses = [], [], []
        
        for p_len in pred_len_list:
            r1 = results[p_len]['stage1']
            r2 = results[p_len]['stage2']
            
            s1_train_losses.append(r1['train'])
            s1_val_losses.append(r1['val'])
            s1_test_losses.append(r1['test'])
            
            s2_train_losses.append(r2['train'])
            s2_val_losses.append(r2['val'])
            s2_test_losses.append(r2['test'])
            
            report_lines.append(f"{p_len:<10} | {r1['train']:<10.4f} | {r1['val']:<10.4f} | {r1['test']:<10.4f} || {r2['train']:<10.4f} | {r2['val']:<10.4f} | {r2['test']:<10.4f}")
            
        report_lines.append("-" * 90)
        avg_s1_train, avg_s1_val, avg_s1_test = np.mean(s1_train_losses), np.mean(s1_val_losses), np.mean(s1_test_losses)
        avg_s2_train, avg_s2_val, avg_s2_test = np.mean(s2_train_losses), np.mean(s2_val_losses), np.mean(s2_test_losses)
        
        report_lines.append(f"{'AVERAGE':<10} | {avg_s1_train:<10.4f} | {avg_s1_val:<10.4f} | {avg_s1_test:<10.4f} || {avg_s2_train:<10.4f} | {avg_s2_val:<10.4f} | {avg_s2_test:<10.4f}")
        report_lines.append("#"*90 + "\n")

        # 将报告合并为字符串
        report_str = "\n".join(report_lines)
        
        # 打印在终端
        print("\n" + report_str)
        
        # 写入 txt 文件
        output_file_path = os.path.join(output_dir, f"{domain_name}.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"--> Saved report for {domain_name} to {output_file_path}\n")

if __name__ == "__main__":
    main()