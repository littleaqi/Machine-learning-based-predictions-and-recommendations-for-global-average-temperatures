import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取数据
data = pd.read_csv('GLOBALTEMPERATURES.CSV')

# 选择需要的列并处理缺失值
data = data[['dt', 'LandAverageTemperature']].dropna()

# 将日期列转换为 datetime 类型
data['dt'] = pd.to_datetime(data['dt'])

# 设置日期列为索引
data.set_index('dt', inplace=True)

# 准备数据集
X = np.arange(len(data))
Y = data['LandAverageTemperature'].values

# 定义 GM(1,1) 模型
def gm11(x0):
    if len(x0) < 2:
        raise ValueError("Input data is too short to fit GM(1,1) model.")
    x1 = np.cumsum(x0)  # 累加生成序列
    z1 = (x1[:-1] + x1[1:]) / 2.0  # 生成均值序列
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = x0[1:]
    params = np.linalg.inv(B.T @ B) @ B.T @ Y
    a, b = params[0], params[1]
    f = lambda k: (x0[0] - b / a) * np.exp(-a * k) + b / a
    return np.array([f(i) for i in range(len(x0))]), a, b

# 交叉验证
kf = KFold(n_splits=5)
mse_scores = []
mae_scores = []

for experiment in range(3):  # 运行 3 次实验
    print(f"Experiment {experiment + 1}")
    for train_index, test_index in kf.split(X):
        train, test = Y[train_index], Y[test_index]

        # 训练 GM(1,1) 模型
        try:
            train_pred, a, b = gm11(train)
        except ValueError as e:
            print(f"Skipping fold due to error: {e}")
            continue

        # 进行预测
        test_pred = []
        for i in range(len(test)):
            test_pred.append((train[-1] - b / a) * np.exp(-a * (i + 1)) + b / a)
        test_pred = np.array(test_pred)

        # 计算均方误差（MSE）和平均绝对误差（MAE）指标并添加到列表中
        mse = mean_squared_error(test, test_pred)
        mae = mean_absolute_error(test, test_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)

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
