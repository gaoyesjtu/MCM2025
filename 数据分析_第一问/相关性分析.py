# -*- coding: utf-8 -*-
# 问题一：Y染色体浓度 ~ 孕周数值 + 孕妇BMI
# 只用三列：孕妇BMI、Y染色体浓度、孕周数值

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1) 读取
df = pd.read_csv("男胎数据问题一.csv", encoding="utf-8")

# 2) 基础检查（可注释）
print("形状：", df.shape)
print("列：", list(df.columns))
print(df[["孕妇BMI","Y染色体浓度","孕周数值"]].describe())

# 3) 相关性（Pearson & Spearman）
y = df["Y染色体浓度"].values
ga = df["孕周数值"].values
bmi = df["孕妇BMI"].values

def corr_report(x, xname):
    pear = stats.pearsonr(x, y)
    spear = stats.spearmanr(x, y)
    print(f"\n{xname} 与 Y染色体浓度 相关性：")
    print(f"  Pearson r={pear.statistic:.4f}, p={pear.pvalue:.3e}")
    print(f"  Spearman ρ={spear.statistic:.4f}, p={spear.pvalue:.3e}")

corr_report(ga, "孕周数值")
corr_report(bmi, "孕妇BMI")

# 4) 线性回归：Y = β0 + β1*孕周 + β2*BMI
X = df[["孕周数值", "孕妇BMI"]].copy()
X = sm.add_constant(X)  # 加截距
model = sm.OLS(y, X).fit()

print("\n=== 线性回归结果（Y ~ 孕周数值 + 孕妇BMI）===")
print(model.summary())

# 5) 可选：两张最基本散点图（二维）便于直观看趋势 —— 可注释
plt.figure(figsize=(6,4))
plt.scatter(df["孕周数值"], df["Y染色体浓度"], s=10, alpha=0.6)
plt.xlabel("孕周数值")
plt.ylabel("Y染色体浓度")
plt.title("孕周数值 vs Y染色体浓度")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["孕妇BMI"], df["Y染色体浓度"], s=10, alpha=0.6)
plt.xlabel("孕妇BMI")
plt.ylabel("Y染色体浓度")
plt.title("孕妇BMI vs Y染色体浓度")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
