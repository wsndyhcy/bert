import torch
import argparse
from transformers import BertTokenizer, BertForSequenceClassification

def predict_sentiment(text, model_path, max_length=128):
    """
    使用微调后的BERT模型预测文本的情感
    
    Args:
        text (str): 要分析的中文文本
        model_path (str): 微调模型的路径
        max_length (int): 最大序列长度
    
    Returns:
        tuple: (预测标签, 预测概率, 标签名称)
    """
    # 检查使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 对输入文本进行编码
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # 将编码后的数据移动到指定设备
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
    # 获取预测结果
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    pred_probs, pred_label = torch.max(probabilities, dim=1)
    
    # 解释标签 (0表示负面，1表示正面)
    sentiment_label = "正面" if pred_label.item() == 1 else "负面"
    
    return pred_label.item(), pred_probs.item(), sentiment_label

def main():
    parser = argparse.ArgumentParser(description='预测中文文本的情感')
    parser.add_argument('--model_path', type=str, default='D:/BERT/model_output/best_model',
                        help='微调后的模型路径')
    parser.add_argument('--text', type=str, required=False,
                        help='要分析的中文文本')
    args = parser.parse_args()
    
    # 如果未提供文本，使用示例文本
    if args.text is None:
        print("未提供测试文本，使用示例文本进行测试。\n")
        test_texts = [
            "这家餐厅的菜品非常美味，服务态度也很好，下次一定还会再来。",
            "酒店环境很差，房间又小又脏，价格还贵，绝对不会再来了。",
            "这部电影剧情还行，但是演员的表演太差了，感觉很尴尬。",
            "这款手机性价比很高，运行速度快，电池续航也不错。"
        ]
        
        print("开始预测情感...\n")
        for text in test_texts:
            label, prob, sentiment = predict_sentiment(text, args.model_path)
            print(f"文本: {text}")
            print(f"预测情感: {sentiment}")
            print(f"概率: {prob:.4f}")
            print("-" * 50)
    else:
        # 预测用户提供的文本
        label, prob, sentiment = predict_sentiment(args.text, args.model_path)
        print(f"文本: {args.text}")
        print(f"预测情感: {sentiment}")
        print(f"概率: {prob:.4f}")
        
if __name__ == "__main__":
    main() 