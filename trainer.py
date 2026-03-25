import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class LeMoLETrainer:
    def __init__(self, model, config, device, learning_rate=1e-3):
        self.model = model
        self.config = config
        self.device = device
        
        # 将模型移动到指定的设备 (CPU/GPU)
        self.model.to(self.device)
        
        # 定义损失函数 (时间序列预测通常使用 MSE)
        self.criterion = nn.MSELoss()
        
        # 定义优化器 (AdamW 通常表现较好)
        # 注意：由于 SBERT 被冻结，优化器只会更新主干网络和 Projector 的参数
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=learning_rate
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        # 使用 tqdm 显示训练进度条
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_x, batch_y, static_text, dynamic_text in progress_bar:
            # 1. 将数值数据移动到 GPU
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            # 注意：文本列表 (static_text, dynamic_text) 保持原样，
            # SbertTextEncoder 内部会自动处理 tokenization 并将其移动到 GPU
            
            # 2. 梯度清零
            self.optimizer.zero_grad()
            
            # 3. 前向传播
            # 你的模型接受 x_enc 以及文本 prompts
            outputs = self.model(
                x_enc=batch_x, 
                static_text=static_text, 
                dynamic_text=dynamic_text
            )
            
            # 4. 计算损失
            # 确保 output 和 target 形状一致 [Batch, Pred_Len, Channels]
            loss = self.criterion(outputs, batch_y)
            
            # 5. 反向传播与优化
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y, static_text, dynamic_text in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(
                    x_enc=batch_x, 
                    static_text=static_text, 
                    dynamic_text=dynamic_text
                )
                
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader=None, epochs=10, save_path="checkpoints"):
        print(f"Starting training on {self.device} for {epochs} epochs...")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), f"{save_path}/lemole_best.pth")
                    print("--> Saved new best model!")
            else:
                print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}")
                
        print("Training complete.")