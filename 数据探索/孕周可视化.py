# -*- coding: utf-8 -*-
"""
检测孕周与平均 Y 浓度 (剔除 18–18.9、22–22.9，步长0.25)
绘制折线图，显示总样本数，并加回归直线（不显示方程与 R²）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression

# ========== 基本设置 ==========
rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 中文字体
rcParams["axes.unicode_minus"] = False

# === 读取数据文件 ===
csv_path = "D:\pycharm_codes\MCM2025_codes\clean_boys_data.csv"  # ← 修改为你本地路径
df = pd.read_csv(csv_path)

# ========== 数据处理 ==========
df = df[["检测孕周", "Y染色体浓度"]].dropna()

# 剔除 18–18.9 与 22–22.9
mask_bad_18 = (df["检测孕周"] >= 18) & (df["检测孕周"] < 19)
mask_bad_22 = (df["检测孕周"] >= 22) & (df["检测孕周"] < 23)
df_keep = df[~(mask_bad_18 | mask_bad_22)].copy()

# 分箱（步长0.25）
min_gw = float(np.floor(df_keep["检测孕周"].min()))
max_gw = float(np.ceil(df_keep["检测孕周"].max()))
bins = np.arange(min_gw, max_gw + 0.25, 0.25)
labels = (bins[:-1] + bins[1:]) / 2

df_keep["孕周_bin"] = pd.cut(df_keep["检测孕周"], bins=bins, labels=labels, include_lowest=True)
grouped = (
    df_keep.groupby("孕周_bin", observed=True)["Y染色体浓度"]
    .mean()
    .reset_index()
    .dropna()
)
grouped["孕周_bin"] = grouped["孕周_bin"].astype(float)

# 总样本数
total_n = len(df_keep)

# ========== 可视化 ==========
plt.figure(figsize=(12, 6))
plt.plot(
    grouped["孕周_bin"],
    grouped["Y染色体浓度"],
    marker="o",
    linestyle="-",
    linewidth=1.5,
    markersize=4,
    label="平均值"
)

# --- 拟合直线：孕周 >= 19 ---
fit_data = grouped[grouped["孕周_bin"] >= 19]
X = fit_data["孕周_bin"].values.reshape(-1, 1)
y = fit_data["Y染色体浓度"].values

if len(X) > 1:
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, color="red", linestyle="--", label=f"孕周>=19 拟合直线 (总样本数 n={total_n})")

plt.xlabel("检测孕周（分箱中心点）")
plt.ylabel("平均 Y 染色体浓度")
plt.title("检测孕周与平均 Y 染色体浓度 (剔除 18–18.9、22–22.9，步长0.25)")
plt.legend(fontsize=12)  # 放大图例字体
plt.grid(True, alpha=0.3)

# 保存 PDF（可选）
plt.savefig("检测孕周与平均Y浓度.pdf", format="pdf", bbox_inches="tight", dpi=300)

plt.show()
