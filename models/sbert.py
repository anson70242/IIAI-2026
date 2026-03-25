import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class SbertTextEncoder(nn.Module):
    def __init__(self, model_name='BAAI/bge-m3', freeze=True):
        super(SbertTextEncoder, self).__init__()
        
        print(f"Loading Sentence-BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.sbert_dim = self.model.get_sentence_embedding_dimension()
        self.freeze = freeze
        self.debug_printed = False  # 控制只打印一次调试信息

        # 显式设置梯度状态
        if freeze:
            print("Status: BERT parameters are FROZEN (requires_grad=False).")
            self.model.eval() # 冻结时设为 eval 模式
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("Status: BERT parameters are TRAINABLE (Fine-tuning).")
            self.model.train() # 训练时必须设为 train 模式！
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, text_list):
        # 0. 确保模型处于正确的模式 (防止被外部意外改成 eval)
        if not self.freeze:
            self.model.train()

        # 1. Tokenize
        # SentenceTransformer 的 tokenize 默认在 CPU，我们需要手动搬运到 GPU
        device = next(self.model.parameters()).device
        features = self.model.tokenize(text_list)
        
        # 将所有输入移动到正确的设备 (GPU)
        features = {key: value.to(device) for key, value in features.items()}

        # 2. Forward Pass
        # 这里调用 transformer 的 forward，这会建立计算图
        output = self.model(features)
        
        # 3. Extract Embeddings (Pooling 后的结果)
        embeddings = output['sentence_embedding']

        # === 调试信息 (只在第一个 Batch 打印) ===
        if not self.debug_printed:
            print("\n" + "="*40)
            print(" [SBERT Debug Info - First Batch] ")
            print("="*40)
            print(f" >> Input Text Count: {len(text_list)}")
            print(f" >> Input Sample: '{text_list[0]}'")
            print(f" >> Model Device: {device}")
            print(f" >> Input Features Device: {features['input_ids'].device}")
            print(f" >> Embeddings Shape: {embeddings.shape}")
            print(f" >> Embeddings Requires Grad: {embeddings.requires_grad}")
            print(f" >> Model Training Mode: {self.model.training}")
            
            if embeddings.requires_grad and self.model.training:
                print(" ✅ Status: Gradients are ON. Backprop should work.")
            else:
                print(" ❌ Status: Gradients are OFF! Check freeze settings.")
            print("="*40 + "\n")
            self.debug_printed = True

        return embeddings