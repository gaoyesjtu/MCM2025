# -*- coding: utf-8 -*-
"""
孕妇体重 69–110 kg 区间，步长 5kg 分箱
绘制平均 Y 浓度折线图，图例包含总样本数和回归直线
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression

# 设置中文字体
rcParams["font.sans-serif"] = ["Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

# === 读取数据 ===
csv_path = "D:\pycharm_codes\MCM2025_codes\clean_boys_data.csv"   # ← 修改为你本地路径
df = pd.read_csv(csv_path)

# 只考虑 69–110 kg 区间
df_weight = df[(df["体重"] >= 69) & (df["体重"] <= 105)][["体重", "Y染色体浓度"]].dropna()

# 分箱 (5 kg 步长)
bins = np.arange(69, 105 + 5, 5)
labels = (bins[:-1] + bins[1:]) / 2

df_weight["体重_bin"] = pd.cut(df_weight["体重"], bins=bins, labels=labels, include_lowest=True)
grouped = (
    df_weight.groupby("体重_bin", observed=True)["Y染色体浓度"]
    .mean()
    .reset_index()
    .dropna()
)
grouped["体重_bin"] = grouped["体重_bin"].astype(float)

# 总样本数
total_n = len(df_weight)

# ========== 可视化 ==========
plt.figure(figsize=(12, 6))
plt.plot(grouped["体重_bin"], grouped["Y染色体浓度"], marker="o", linestyle="-", linewidth=1.5, label="平均值")

# --- 整体回归直线 ---
X = grouped["体重_bin"].values.reshape(-1, 1)
y = grouped["Y染色体浓度"].values
if len(X) > 1:
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, color="red", linestyle="--", label=f"回归直线 (总样本数 n={total_n})")

plt.xlabel("孕妇体重 (kg, 分箱中心点)")
plt.ylabel("平均 Y 染色体浓度")
plt.title("孕妇体重 (69–105 kg) 与平均 Y 染色体浓度关系 (步长 5 kg)")
plt.legend(fontsize=12)  # 调大图例字体
plt.grid(True, alpha=0.3)

# 可选保存
plt.savefig("体重_Y浓度_69_105_含总样本数_回归.pdf", format="pdf", bbox_inches="tight", dpi=300)

plt.show()
