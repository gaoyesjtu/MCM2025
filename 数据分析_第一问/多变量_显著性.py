# -*- coding: utf-8 -*-
"""
最能体现模型显著性的两张图：
1) 系数森林图（含95%CI与p值）
2) 显著性柱状图（-log10 p，含0.05与Bonferroni阈值）
模型：方案A（BMI 主效应 + 体重残差微调），无随机项/无稳健SE
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
import matplotlib.pyplot as plt

# —— 统一中文字体为“微软雅黑” ——
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ========= 配置 =========
DATA_PATH = "D:\pycharm_codes\MCM2025_codes\clean_boys_data.csv"   # 改成你的路径
GA_KNOT = 19.0
BMI_KNOT = 33.0
EPS = 1e-6
OUT_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)
# =======================

# 1) 读取 & 预处理
df = pd.read_csv(DATA_PATH).dropna(subset=["Y染色体浓度","检测孕周","孕妇BMI","体重","年龄"])
# 响应：logit 变换
df["y_logit"] = np.log((df["Y染色体浓度"] + EPS) / (1 - df["Y染色体浓度"] + EPS))
# 分段特征
df["GA1"]  = np.minimum(df["检测孕周"], GA_KNOT)
df["GA2"]  = np.maximum(df["检测孕周"] - GA_KNOT, 0.0)
df["BMI1"] = np.minimum(df["孕妇BMI"], BMI_KNOT)
df["BMI2"] = np.maximum(df["孕妇BMI"] - BMI_KNOT, 0.0)
# 残差化体重（与 BMI 分段正交）
Xw = sm.add_constant(df[["BMI1","BMI2"]])
w_m = sm.OLS(df["体重"], Xw).fit()
df["体重残差"] = df["体重"] - w_m.predict(Xw)

# 2) 拟合 OLS（无随机项/无稳健SE）
X = sm.add_constant(df[["GA1","GA2","BMI1","BMI2","年龄","体重残差"]])
y = df["y_logit"]
ols = sm.OLS(y, X).fit()

# 3) 组织显著性表
ci = ols.conf_int()
coef_tbl = pd.DataFrame({
    "term": X.columns,
    "coef": ols.params.values,
    "se":   ols.bse.values,
    "p":    ols.pvalues.values,
    "low":  ci[0].values,
    "high": ci[1].values
})
coef_tbl = coef_tbl[coef_tbl["term"]!="const"].copy()  # 去掉截距
coef_tbl["minus_log10_p"] = -np.log10(coef_tbl["p"].clip(lower=1e-300))
# 按显著性排序（p 值从小到大）
coef_tbl = coef_tbl.sort_values("p", ascending=True).reset_index(drop=True)

# 优化版森林图：显著性分级配色，不显示系数数值
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# —— 设置字体为微软雅黑 ——
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 根据显著性分配颜色
def p_to_color(p):
    if p < 0.001:
        return "#019092"  # 红
    elif p < 0.05:
        return "#0095FF"  # 橙
    else:
        return "gray"    # 灰"#019092"

plt.figure(figsize=(7, 4.5))
ypos = np.arange(len(coef_tbl))

# 绘制误差线和点，点颜色依显著性变化
for i, row in coef_tbl.iterrows():
    plt.errorbar(
        x=row["coef"], y=i,
        xerr=[[row["coef"] - row["low"]], [row["high"] - row["coef"]]],
        fmt="o", capsize=4, color=p_to_color(row["p"]), markersize=7
    )

plt.yticks(ypos, coef_tbl["term"])
plt.axvline(0, linestyle="--", color="black", linewidth=1)
plt.title("回归系数及95%置信区间", fontsize=12)
plt.xlabel("系数估计值 (logit 尺度)", fontsize=11)
plt.tight_layout()

# 保存图片
out_path = os.path.join(OUT_DIR, "01_forest_coef_sigcolor.pdf")
plt.savefig(out_path, dpi=300)
plt.close()

print("优化后的森林图已保存：", out_path)



# —— 显著性柱状图（-log10 p）· 优化图例 + 红线说明 ——
plt.figure(figsize=(7, 4.5))

# 颜色同你的版本
colors = []
for p in coef_tbl["p"]:
    if p < 0.001:
        colors.append("#019092")   # 高度显著
    elif p < 0.05:
        colors.append("#6FDCB5")   # 显著
    else:
        colors.append("gray")      # 不显著

bars = plt.bar(coef_tbl["term"], coef_tbl["minus_log10_p"], color=colors)

plt.xticks(rotation=25, ha="right")
plt.ylabel("−log10(p)", fontsize=11)
plt.title("特征显著性", fontsize=12)

# 阈值线（抓住句柄，加入 legend）
alpha = 0.05
m = len(coef_tbl)
alpha_bonf = alpha / m
y_p05 = -np.log10(alpha)
y_bonf = -np.log10(alpha_bonf)

line_p05  = plt.axhline(y_p05,  color="gray", linestyle="--", linewidth=1,
                        label="显著性阈值 p = 0.05", zorder=3)
line_bonf = plt.axhline(y_bonf, color="red",  linestyle="--", linewidth=1,
                        label=f"Bonferroni 校正阈值 (p = {alpha_bonf:.3g})", zorder=3)


# 颜色图例（保留你的三段显著性分级）
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor="#019092", label="高度显著 (p<0.001)"),
    Patch(facecolor="#6FDCB5", label="显著 (p<0.05)"),
    Patch(facecolor="gray",    label="不显著 (p≥0.05)")
]

# 合并：颜色分级 + 两条阈值线 ——> 一个清晰的图例
plt.legend(
    handles=legend_patches + [line_p05, line_bonf],
    fontsize=9,
    loc="upper right",
    frameon=True,         # 打开图例边框
    edgecolor="gray"     # 边框颜色（黑色）
)


plt.tight_layout()
out_path2 = os.path.join(OUT_DIR, "02_pvalue_bar_sigcolor.pdf")
plt.savefig(out_path2, dpi=300)
plt.close()

print("优化后的显著性柱状图已保存：", out_path2)



print("已生成并保存两张图片：")
print("1) 系数森林图：", forest_path)
print("2) 显著性柱状图：", pbar_path)
