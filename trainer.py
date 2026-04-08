import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class LeMoLETrainer:
    def __init__(self, model, config, device, learning_rate=1e-3):
        self.model = model
        self.config = config
        self.device = device
        self.learning_rate = learning_rate
        
        self.model.to(self.device)
        self.criterion = nn.SmoothL1Loss()
        
        # 控制是否跳过融合层的开关
        self.skip_fusion = False 
        self.timemmd = False

        # 初始化构建优化器
        self.build_optimizer()

    def build_optimizer(self, weight_decay=0.005):
        """根据当前 parameters 的 requires_grad 状态动态构建优化器"""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.AdamW(
            trainable_params, 
            lr=self.learning_rate,
            weight_decay=weight_decay
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_x, batch_y, static_text, dynamic_text in progress_bar:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                x_enc=batch_x, 
                static_text=static_text, 
                dynamic_text=dynamic_text,
                skip_fusion=self.skip_fusion,
                timemmd=self.timemmd
            )
            
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion=None):
        self.model.eval()
        total_loss = 0.0
        
        # 如果没有传入特定的 loss 函数，就用默认的 Smooth L1
        if criterion is None:
            criterion = self.criterion
        
        with torch.no_grad():
            for batch_x, batch_y, static_text, dynamic_text in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(
                    x_enc=batch_x, 
                    static_text=static_text, 
                    dynamic_text=dynamic_text,
                    skip_fusion=self.skip_fusion,
                    timemmd=self.timemmd
                )
                
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)


class LeMoLETrainerWithES(LeMoLETrainer):
    """继承原有的 LeMoLETrainer，加入 Early Stopping、Test 流程和可视化"""
    def __init__(self, model, config, device, learning_rate, scaler=None):
        super().__init__(model, config, device, learning_rate)
        self.scaler = scaler
    
    def visualize_predictions(self, dataloader, save_path, pred_len, num_samples=3):
        self.model.eval()
        
        with torch.no_grad():
            for batch_x, batch_y, static_text, dynamic_text in dataloader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(
                    x_enc=batch_x, 
                    static_text=static_text, 
                    dynamic_text=dynamic_text,
                    skip_fusion=self.skip_fusion,
                    timemmd=self.timemmd
                )
                
                f_dim = -1 if self.config.features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:]
                
                preds = outputs.detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()
                histories = batch_x.detach().cpu().numpy()
                break

        if self.scaler is not None:
            histories = self.scaler.inverse_transform(histories.reshape(-1, 1)).reshape(histories.shape)
            preds = self.scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
            trues = self.scaler.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)
            y_label_text = "Value (Original)"
        else:
            y_label_text = "Value (Scaled)"

        plt.figure(figsize=(12, 4 * num_samples))
        for i in range(min(num_samples, preds.shape[0])):
            plt.subplot(num_samples, 1, i + 1)
            
            history_len = histories.shape[1]
            x_history = range(history_len)
            plt.plot(x_history, histories[i, :, -1], label='History (Input)', color='gray', linestyle='--')
            
            x_pred = range(history_len, history_len + pred_len)
            plt.plot(x_pred, trues[i, :, -1], label='Ground Truth', color='blue', marker='o')
            plt.plot(x_pred, preds[i, :, -1], label='Prediction', color='red', marker='x')
            
            plt.title(f"Test Sample {i+1} | Prediction Length: {pred_len}")
            plt.xlabel("Time Steps")
            plt.ylabel(y_label_text)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        img_path = os.path.join(save_path, f"pred_vs_truth_pl{pred_len}.png")
        plt.savefig(img_path)
        plt.close()
        print(f"--> [Visualizer] Saved prediction plot to: {img_path}")
        
    def train_with_es(self, train_loader, val_loader, test_loader, epochs=100, patience=20, save_path="checkpoints"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end=" ")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                torch.save(self.model.state_dict(), f"{save_path}/lemole_best.pth")
                print("--> Saved new best model!")
                early_stopping_counter = 0  
            else:
                early_stopping_counter += 1
                print(f"--> EarlyStopping counter: {early_stopping_counter}/{patience}")
                if early_stopping_counter >= patience:
                    print("Early stopping triggered. Training stopped.")
                    break
                    
        print("\nLoading best model for testing and visualization...")
        self.model.load_state_dict(torch.load(f"{save_path}/lemole_best.pth"))
        
        mse_criterion = nn.MSELoss()
        test_loss = self.validate(test_loader, criterion=mse_criterion)
        print(f"Final Test Loss: {test_loss:.4f}")
        
        self.visualize_predictions(test_loader, save_path, self.config.pred_len)
        
        return best_train_loss, best_val_loss, test_loss