import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# 定义数据集类
class ChnsentiCorpDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]['text_a']
        label = self.data.iloc[index]['label']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 训练函数
def train_model(model, train_dataloader, val_dataloader, epochs, optimizer, scheduler, device):
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f"\n{'='*20} Epoch {epoch+1}/{epochs} {'='*20}")
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 清除之前的梯度
            model.zero_grad()
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                
                val_preds.extend(preds.cpu().tolist())
                val_true.extend(labels.cpu().tolist())
        
        val_accuracy = accuracy_score(val_true, val_preds)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(val_true, val_preds, target_names=['Negative', 'Positive']))
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"Saving best model with accuracy: {best_accuracy:.4f}")
            model_save_path = os.path.join("D:/BERT/model_output", "best_model")
            os.makedirs(model_save_path, exist_ok=True)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)

# 主函数
def main():
    # 设置参数
    bert_model_path = "D:/BERT/model_load/bert-base-chinese"
    train_path = "D:/BERT/dataset/train.tsv"
    dev_path = "D:/BERT/dataset/dev.tsv"
    test_path = "D:/BERT/dataset/test.tsv"
    batch_size = 16
    epochs = 3
    max_length = 128
    learning_rate = 2e-5
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集
    print("Loading datasets...")
    try:
        train_data = pd.read_csv(train_path, sep='\t', encoding='utf-8')
        dev_data = pd.read_csv(dev_path, sep='\t', encoding='utf-8')
        test_data = pd.read_csv(test_path, sep='\t', encoding='utf-8')
    except UnicodeDecodeError:
        # 如果utf-8编码失败，尝试使用其他编码
        encodings = ['gbk', 'gb2312', 'gb18030', 'latin1']
        for encoding in encodings:
            try:
                train_data = pd.read_csv(train_path, sep='\t', encoding=encoding)
                dev_data = pd.read_csv(dev_path, sep='\t', encoding=encoding)
                test_data = pd.read_csv(test_path, sep='\t', encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(dev_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # 加载分词器和模型
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        bert_model_path, 
        num_labels=2,  # 二分类任务
        local_files_only=True
    )
    model.to(device)
    
    # 创建数据集和数据加载器
    print("Creating datasets and dataloaders...")
    train_dataset = ChnsentiCorpDataset(train_data, tokenizer, max_length)
    val_dataset = ChnsentiCorpDataset(dev_data, tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练模型
    print("Starting training...")
    train_model(model, train_dataloader, val_dataloader, epochs, optimizer, scheduler, device)
    
    # 在测试集上评估最佳模型
    print("\nEvaluating on test set...")
    best_model_path = os.path.join("D:/BERT/model_output", "best_model")
    
    if os.path.exists(best_model_path):
        best_model = BertForSequenceClassification.from_pretrained(best_model_path)
        best_model.to(device)
        
        test_dataset = ChnsentiCorpDataset(test_data, tokenizer, max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        best_model.eval()
        test_preds = []
        test_true = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = best_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                
                test_preds.extend(preds.cpu().tolist())
                test_true.extend(labels.cpu().tolist())
        
        test_accuracy = accuracy_score(test_true, test_preds)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        print("\nTest Classification Report:")
        print(classification_report(test_true, test_preds, target_names=['Negative', 'Positive']))
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main() 