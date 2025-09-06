# -*- coding: utf-8 -*-
"""
方案A 可视化（精简版）
只保留 4 张关键图表，字体统一微软雅黑
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.special import expit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 设置字体为微软雅黑
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========= 配置 =========
DATA_PATH = "D:\pycharm_codes\MCM2025_codes\clean_boys_data.csv"  # 改成你的路径
GA_KNOT, BMI_KNOT, EPS, N_SPLITS = 19.0, 33.0, 1e-6, 5
# =======================

# 1) 数据读取与预处理
df = pd.read_csv(DATA_PATH).dropna(subset=["Y染色体浓度","检测孕周","孕妇BMI","体重","年龄","孕妇代码"])
df["y_logit"] = np.log((df["Y染色体浓度"] + EPS) / (1 - df["Y染色体浓度"] + EPS))
df["GA1"]  = np.minimum(df["检测孕周"], GA_KNOT)
df["GA2"]  = np.maximum(df["检测孕周"] - GA_KNOT, 0.0)
df["BMI1"] = np.minimum(df["孕妇BMI"], BMI_KNOT)
df["BMI2"] = np.maximum(df["孕妇BMI"] - BMI_KNOT, 0.0)
# 残差化体重
X_w = sm.add_constant(df[["BMI1","BMI2"]])
df["体重残差"] = df["体重"] - sm.OLS(df["体重"], X_w).fit().predict(X_w)

# 2) 拟合模型
X = sm.add_constant(df[["GA1","GA2","BMI1","BMI2","年龄","体重残差"]])
y = df["y_logit"]
ols = sm.OLS(y, X).fit()

# -------- 图表1：系数 ±95% CI --------
ci = ols.conf_int()
coef_tbl = pd.DataFrame({"term":X.columns,"coef":ols.params,"low":ci[0],"high":ci[1]})
coef_tbl = coef_tbl[coef_tbl["term"]!="const"].sort_values("coef")
plt.figure()
ypos = np.arange(len(coef_tbl))
plt.errorbar(coef_tbl["coef"], ypos,
             xerr=[coef_tbl["coef"]-coef_tbl["low"], coef_tbl["high"]-coef_tbl["coef"]],
             fmt="o", capsize=4)
plt.yticks(ypos, coef_tbl["term"])
plt.axvline(0, linestyle="--", color="grey")
plt.title("回归系数及95%置信区间")
plt.xlabel("系数 (logit尺度)")
plt.tight_layout()
plt.show()

# -------- 图表2：VIF --------
vif_df = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
})
vif_df = vif_df[vif_df["feature"]!="const"]
plt.figure()
plt.bar(vif_df["feature"], vif_df["VIF"])
plt.axhline(5, linestyle="--", color="red") # 警戒线
plt.title("多重共线性检查 (VIF)")
plt.ylabel("VIF 值")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# -------- 图表3：预测 vs 真实 --------
yhat = expit(ols.predict(X))
plt.figure()
plt.scatter(df["Y染色体浓度"], yhat, s=12)
lims = [min(df["Y染色体浓度"].min(), yhat.min()), max(df["Y染色体浓度"].max(), yhat.max())]
plt.plot(lims, lims, linestyle="--", color="grey")
plt.xlabel("真实 Y 浓度")
plt.ylabel("预测 Y 浓度")
plt.title("预测 vs 真实")
plt.tight_layout()
plt.show()

# -------- 图表4：交叉验证 RMSE --------
groups = df["孕妇代码"]
gkf = GroupKFold(n_splits=N_SPLITS)
fold_rmse = []
for tr, te in gkf.split(X, y, groups=groups):
    m = sm.OLS(y.iloc[tr], X.iloc[tr]).fit()
    pred = expit(m.predict(X.iloc[te]))
    truth = df["Y染色体浓度"].iloc[te]
    rmse = np.sqrt(mean_squared_error(truth, pred))
    fold_rmse.append(rmse)
plt.figure()
plt.bar([f"Fold{i+1}" for i in range(N_SPLITS)], fold_rmse)
plt.axhline(np.mean(fold_rmse), linestyle="--", color="grey")
plt.title(f"交叉验证RMSE (平均={np.mean(fold_rmse):.4f})")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()
