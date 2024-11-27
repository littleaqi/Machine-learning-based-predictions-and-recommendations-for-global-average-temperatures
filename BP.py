import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models

# 读取数据
data = pd.read_csv('GLOBALTEMPERATURES.CSV')

# 选择需要的列并处理缺失值
data = data[['dt', 'LandAverageTemperature']].dropna()

# 将日期列转换为 datetime 类型
data['dt'] = pd.to_datetime(data['dt'])

# 设置日期列为索引
data.set_index('dt', inplace=True)

# 归一化数据
scaler = MinMaxScaler()
data['LandAverageTemperature'] = scaler.fit_transform(data[['LandAverageTemperature']])

# 准备数据集
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# 设置时间步长
time_step = 10
X, Y = create_dataset(data.values, time_step)

# 定义 BP 神经网络模型
def create_model():
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=time_step, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 交叉验证
kf = KFold(n_splits=5)
mse_scores = []
mae_scores = []

for experiment in range(3):  # 运行 3 次实验
    print(f"Experiment {experiment + 1}")
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # 创建并训练 BP 神经网络模型
        model = create_model()
        history = model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=0)

        # 在测试集上进行评估
        Y_pred = model.predict(X_test)

        # 反归一化操作
        Y_pred = scaler.inverse_transform(Y_pred)
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # 计算均方误差（MSE）和平均绝对误差（MAE）指标并添加到列表中
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)

        # 打印本次折叠的损失历史（训练和验证损失）
        print("Train Loss History:", history.history['loss'])
        print("Validation Loss History:", history.history['val_loss'])

# 将指标数据整理成 DataFrame 格式
result_df = pd.DataFrame({
    'MSE': mse_scores,
    'MAE': mae_scores
})

# 计算平均值和标准差
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
mean_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)

# 打印平均值和标准差
print(f"Mean MSE: {mean_mse}, Std MSE: {std_mse}")
print(f"Mean MAE: {mean_mae}, Std MAE: {std_mae}")

# 绘制汇总图表
plt.figure(figsize=(14, 7))

# 绘制 MSE 分布图
plt.subplot(1, 2, 1)
plt.hist(mse_scores, bins=10, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(mean_mse, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_mse + std_mse, color='green', linestyle='dashed', linewidth=1)
plt.axvline(mean_mse - std_mse, color='green', linestyle='dashed', linewidth=1)
plt.title('MSE Distribution')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.legend(['Mean', 'Mean ± Std'])

# 绘制 MAE 分布图
plt.subplot(1, 2, 2)
plt.hist(mae_scores, bins=10, color='green', alpha=0.7, edgecolor='black')
plt.axvline(mean_mae, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_mae + std_mae, color='green', linestyle='dashed', linewidth=1)
plt.axvline(mean_mae - std_mae, color='green', linestyle='dashed', linewidth=1)
plt.title('MAE Distribution')
plt.xlabel('MAE')
plt.ylabel('Frequency')
plt.legend(['Mean', 'Mean ± Std'])

plt.tight_layout()
plt.show()
