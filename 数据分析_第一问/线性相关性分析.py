# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# === 字体配置（你已安装 Microsoft YaHei） ===
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# === 数据路径 ===
CSV_PATH = r"D:\pycharm_codes\MCM2025_codes\数据分析_第一问\男胎数据_问题一.csv"

# === 读取数据 ===
df = pd.read_csv(CSV_PATH, encoding="utf-8")
y   = df["Y染色体浓度"].to_numpy()
ga  = df["检测孕周"].to_numpy()
bmi = df["孕妇BMI"].to_numpy()

# === 1. 控制 BMI，看检测孕周 ===
X_bmi = sm.add_constant(bmi)
resid_y = y - sm.OLS(y, X_bmi).fit().fittedvalues
resid_ga = ga - sm.OLS(ga, X_bmi).fit().fittedvalues
model_ga = sm.OLS(resid_y, sm.add_constant(resid_ga)).fit()

plt.figure(figsize=(6,4))
plt.scatter(resid_ga, resid_y, s=12, alpha=0.6, label="残差点")
x_line = np.linspace(resid_ga.min(), resid_ga.max(), 200)
y_line = model_ga.params[0] + model_ga.params[1]*x_line
plt.plot(x_line, y_line, color="red", label="拟合线")
plt.xlabel("检测孕周（控制BMI后的残差）")
plt.ylabel("Y染色体浓度（控制BMI后的残差）")
plt.title("控制BMI后：检测孕周与Y浓度的关系")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("控制BMI后：检测孕周系数 = %.4f, p=%.3e" % (model_ga.params[1], model_ga.pvalues[1]))

# === 2. 控制检测孕周，看BMI ===
X_ga = sm.add_constant(ga)
resid_y2 = y - sm.OLS(y, X_ga).fit().fittedvalues
resid_bmi = bmi - sm.OLS(bmi, X_ga).fit().fittedvalues
model_bmi = sm.OLS(resid_y2, sm.add_constant(resid_bmi)).fit()

plt.figure(figsize=(6,4))
plt.scatter(resid_bmi, resid_y2, s=12, alpha=0.6, label="残差点")
x_line = np.linspace(resid_bmi.min(), resid_bmi.max(), 200)
y_line = model_bmi.params[0] + model_bmi.params[1]*x_line
plt.plot(x_line, y_line, color="red", label="拟合线")
plt.xlabel("孕妇BMI（控制检测孕周后的残差）")
plt.ylabel("Y染色体浓度（控制检测孕周后的残差）")
plt.title("控制检测孕周后：BMI与Y浓度的关系")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("控制检测孕周后：BMI系数 = %.4f, p=%.3e" % (model_bmi.params[1], model_bmi.pvalues[1]))
