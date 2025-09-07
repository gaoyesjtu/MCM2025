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
