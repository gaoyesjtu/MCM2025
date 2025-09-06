import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression

# 设置中文字体为微软雅黑
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv("D:\pycharm_codes\MCM2025_codes\clean_boys_data.csv")

# 计算BMI
df["BMI"] = df["体重"] / ((df["身高"] / 100) ** 2)

# --- 分区处理 ---
# 25-35 区间，步长0.5
bins1 = np.arange(28, 35.5, 0.5)
labels1 = (bins1[:-1] + bins1[1:]) / 2
df_sub1 = df[(df["BMI"] >= 25) & (df["BMI"] < 35)].copy()
df_sub1["BMI_bin"] = pd.cut(df_sub1["BMI"], bins=bins1, labels=labels1, include_lowest=True)
grouped1 = df_sub1.groupby("BMI_bin")["Y染色体浓度"].mean().reset_index()

# 35-37 区间，步长0.25
bins2 = np.arange(35, 37.25, 0.25)
labels2 = (bins2[:-1] + bins2[1:]) / 2
df_sub2 = df[(df["BMI"] >= 35) & (df["BMI"] <= 37)].copy()
df_sub2["BMI_bin"] = pd.cut(df_sub2["BMI"], bins=bins2, labels=labels2, include_lowest=True)
grouped2 = df_sub2.groupby("BMI_bin")["Y染色体浓度"].mean().reset_index()

# 合并结果
grouped_final = pd.concat([grouped1, grouped2], ignore_index=True).dropna()

# --- 绘制图像 ---
plt.figure(figsize=(10,6))
plt.plot(grouped_final["BMI_bin"], grouped_final["Y染色体浓度"], marker="o", linestyle="-", label="平均值")

# 线性拟合 (28–34)
fit_data = grouped_final[(grouped_final["BMI_bin"] >= 28) & (grouped_final["BMI_bin"] <= 33)]
X = fit_data["BMI_bin"].values.reshape(-1, 1)
y = fit_data["Y染色体浓度"].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

plt.plot(X, y_pred, color="red", linestyle="--", label="BMI区间28–33拟合直线")

plt.xlabel("孕妇BMI (区间中心点)")
plt.ylabel("平均Y染色体浓度")
plt.title("孕妇BMI区间与平均Y染色体浓度关系图")
plt.legend()
plt.grid(True)
plt.savefig("BMI与平均Y染色体浓度.pdf", format="pdf", bbox_inches="tight")
plt.show()

