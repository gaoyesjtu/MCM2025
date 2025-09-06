# -*- coding: utf-8 -*-
"""
孕妇年龄 ≤37 岁，步长 1 岁分箱
绘制平均 Y 浓度折线图，显示总样本数，并加回归直线（不显示方程与 R²）
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

# 只考虑 ≤37 岁
df_age = df[df["年龄"] <= 37][["年龄", "Y染色体浓度"]].dropna()

# 分箱 (步长 1 岁)
min_age = int(np.floor(df_age["年龄"].min()))
max_age = int(np.ceil(df_age["年龄"].max()))
bins = np.arange(min_age, max_age + 1, 1)
labels = (bins[:-1] + bins[1:]) / 2

df_age["年龄_bin"] = pd.cut(df_age["年龄"], bins=bins, labels=labels, include_lowest=True)
grouped = (
    df_age.groupby("年龄_bin", observed=True)["Y染色体浓度"]
    .mean()
    .reset_index()
    .dropna()
)
grouped["年龄_bin"] = grouped["年龄_bin"].astype(float)

# 总样本数
total_n = len(df_age)

# ========== 可视化 ==========
plt.figure(figsize=(12, 6))
plt.plot(grouped["年龄_bin"], grouped["Y染色体浓度"], marker="o", linestyle="-", linewidth=1.5, label="平均值")

# --- 回归直线 ---
X = grouped["年龄_bin"].values.reshape(-1, 1)
y = grouped["Y染色体浓度"].values
if len(X) > 1:
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, color="red", linestyle="--", label=f"回归直线 (总样本数 n={total_n})")

plt.xlabel("孕妇年龄 (岁, 分箱中心点)")
plt.ylabel("平均 Y 染色体浓度")
plt.title("孕妇年龄 ≤37 岁与平均 Y 染色体浓度关系 (步长 1 岁)")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# 可选保存
plt.savefig("年龄_Y浓度_37岁以内.pdf", format="pdf", bbox_inches="tight", dpi=300)

plt.show()
