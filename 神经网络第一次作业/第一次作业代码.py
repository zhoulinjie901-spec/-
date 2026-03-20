import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)
# 绘图设置
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# 读取数据
df = pd.read_csv('Concrete_Data_Yeh.csv')
print("数据集形状:", df.shape)
df.head()
# 数据基本信息
df.info()
# 描述性统计
df.describe()
# 检查缺失值
df.isnull().sum()
# 计算所有特征与目标变量的相关系数
corr_with_target = df.corr()['csMPa'].sort_values(ascending=False)
print("各特征与抗压强度的相关系数：\n", corr_with_target)
# 绘制相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('热力图')
plt.show()
# 分离特征和标签
X = df.drop('csMPa', axis=1).values
y = df['csMPa'].values.reshape(-1, 1)

# 划分训练集和测试集
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")
# 数据标准化（Z-score）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
# 定义网络结构
class ConcreteNN(nn.Module):
    def __init__(self, input_dim):
        super(ConcreteNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  
        )
    
    def forward(self, x):
        return self.net(x)

# 初始化模型
input_dim = X_train.shape[1]
model = ConcreteNN(input_dim)
print(model)
# 创建 DataLoader 以便批量训练
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = ConcreteNN(input_dim)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('损失函数曲线')
plt.legend()
plt.grid(True)
plt.show()
# 测试集预测
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)  
    y_actual = scaler_y.inverse_transform(y_test_tensor.numpy())

# 计算均方误差
mse = mean_squared_error(y_actual, y_pred)
print(f"测试集均方误差 (MSE): {mse:.4f}")
# 绘制真实值与预测值散点图
plt.figure(figsize=(6,6))
plt.scatter(y_actual, y_pred, alpha=0.6)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
plt.xlabel('真实抗压强度 (MPa)')
plt.ylabel('预测抗压强度 (MPa)')
plt.title('真实值 vs 预测值')
plt.grid(True)
plt.show()
