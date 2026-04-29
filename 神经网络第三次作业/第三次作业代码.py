import json
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== 配置参数 =====================
# 数据路径
DATA_DIR = "./poetry_data"
DATA_FILES = ["poet.song.40k.json", "poet.song.41k.json", 
              "poet.song.42k.json", "poet.song.43k.json"]

# 生成格式配置（七言绝句：4句，每句7字）
POEM_TYPE = "七言绝句"
SENTENCE_NUM = 4    # 诗句总数量
SENTENCE_LEN = 7    # 每句字数
TOTAL_LEN = SENTENCE_NUM * SENTENCE_LEN  # 全诗总字数

# 模型超参数
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成配置
START_WORDS = "明月"  # 起始词
GENERATE_TEMPERATURE = 0.7  # 温度系数，越小越保守，越大越随机

# ===================== 1. 修正后的数据预处理 =====================
def load_poems():
    """加载并筛选符合格式的古诗（修复中文过滤和诗句拆分问题）"""
    poems = []
    # 正则表达式：只保留中文字符，去掉所有标点、空格、数字等
    chinese_only = re.compile(r'[^\u4e00-\u9fa5]')
    
    for file in DATA_FILES:
        file_path = os.path.join(DATA_DIR, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                paragraphs = item["paragraphs"]
                all_sentences = []
                valid = True
                
                for para in paragraphs:
                    # 只保留中文字符
                    clean_para = chinese_only.sub('', para)
                    # 每个段落必须是2句×7字=14字（数据集格式）
                    if len(clean_para) != 2 * SENTENCE_LEN:
                        valid = False
                        break
                    # 拆分成两句
                    sentence1 = clean_para[:SENTENCE_LEN]
                    sentence2 = clean_para[SENTENCE_LEN:]
                    all_sentences.append(sentence1)
                    all_sentences.append(sentence2)
                
                # 筛选：总句数正好等于要求的数量
                if valid and len(all_sentences) == SENTENCE_NUM:
                    full_poem = "".join(all_sentences)
                    poems.append(full_poem)
    
    print(f"共加载符合{POEM_TYPE}格式的古诗：{len(poems)}首")
    return poems

def build_vocab(poems):
    """构建词汇表"""
    all_chars = set()
    for poem in poems:
        all_chars.update(poem)
    # 添加特殊字符
    all_chars = sorted(list(all_chars))
    char2idx = {char: idx+1 for idx, char in enumerate(all_chars)}  # 0留给padding
    char2idx["<PAD>"] = 0
    idx2char = {idx: char for char, idx in char2idx.items()}
    vocab_size = len(char2idx)
    print(f"词汇表大小：{vocab_size}")
    return char2idx, idx2char, vocab_size

class PoetryDataset(Dataset):
    """古诗数据集类"""
    def __init__(self, poems, char2idx, seq_len):
        self.poems = poems
        self.char2idx = char2idx
        self.seq_len = seq_len
        self.data = self._prepare_data()
    
    def _prepare_data(self):
        """将古诗转换为数字序列"""
        sequences = []
        for poem in self.poems:
            # 转换为索引
            seq = [self.char2idx[c] for c in poem]
            # 生成输入和标签（输入：前n-1字，标签：后n-1字）
            for i in range(1, len(seq)):
                input_seq = seq[:i]
                target = seq[i]
                # 填充到固定长度
                input_seq = [0]*(self.seq_len - len(input_seq)) + input_seq
                sequences.append((input_seq, target))
        return sequences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# ===================== 2. LSTM模型构建 =====================
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embed = self.dropout(embed)
        
        if hidden is None:
            output, hidden = self.lstm(embed)
        else:
            output, hidden = self.lstm(embed, hidden)
        
        # 取最后一个时间步的输出
        output = output[:, -1, :]  # [batch_size, hidden_dim]
        output = self.dropout(output)
        logits = self.fc(output)  # [batch_size, vocab_size]
        return logits, hidden

# ===================== 3. 模型训练 =====================
def train_model(model, dataloader, criterion, optimizer, epochs, device):
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            # 每200步打印一次loss（和作业示例格式一致）
            if (batch_idx + 1) % 200 == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # 计算平均loss
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"\n==== Epoch {epoch+1} Average Loss: {avg_loss:.4f} ====")
        
        # 每个epoch结束后生成一首示例古诗
        print("【生成演示】:")
        generate_poem(model, START_WORDS, char2idx, idx2char, device)
        print("-"*50)
    
    return loss_history

# ===================== 4. 古诗生成 =====================
def generate_poem(model, start_words, char2idx, idx2char, device, temperature=GENERATE_TEMPERATURE):
    model.eval()
    generated = list(start_words)
    
    with torch.no_grad():
        # 初始化隐藏状态
        hidden = None
        
        # 先输入起始词
        for char in start_words:
            input_seq = torch.tensor([[char2idx[char]]], dtype=torch.long).to(device)
            logits, hidden = model(input_seq, hidden)
        
        # 生成剩余字符
        for _ in range(TOTAL_LEN - len(start_words)):
            # 取最后一个字符作为输入
            last_char = generated[-1]
            input_seq = torch.tensor([[char2idx[last_char]]], dtype=torch.long).to(device)
            logits, hidden = model(input_seq, hidden)
            
            # 温度采样
            logits = logits / temperature
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            next_idx = np.random.choice(len(probs), p=probs)
            next_char = idx2char[next_idx]
            
            generated.append(next_char)
    
    # 按格式输出（每句7字，共4句）
    poem = "".join(generated)
    for i in range(SENTENCE_NUM):
        print(poem[i*SENTENCE_LEN : (i+1)*SENTENCE_LEN])

# ===================== 5. Loss曲线绘制 =====================
def plot_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history)+1), loss_history, 'b-', linewidth=2)
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 1. 加载并预处理数据
    print("正在加载数据...")
    poems = load_poems()
    if len(poems) == 0:
        raise ValueError("未找到符合格式的古诗，请检查数据文件或格式配置")
    
    char2idx, idx2char, vocab_size = build_vocab(poems)
    dataset = PoetryDataset(poems, char2idx, seq_len=TOTAL_LEN-1)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 初始化模型
    print("正在初始化模型...")
    model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. 训练模型
    print("开始训练...")
    loss_history = train_model(model, dataloader, criterion, optimizer, EPOCHS, DEVICE)
    
    # 4. 绘制Loss曲线
    print("绘制训练Loss曲线...")
    plot_loss(loss_history)
    
    # 5. 保存模型
    torch.save(model.state_dict(), "poetry_lstm_model.pth")
    print("模型已保存为 poetry_lstm_model.pth")
    
    # 6. 最终生成示例
    print("\n" + "="*30 + " 最终生成结果 " + "="*30)
    generate_poem(model, START_WORDS, char2idx, idx2char, DEVICE)
