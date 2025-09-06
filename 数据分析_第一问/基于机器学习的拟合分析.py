# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CSV_PATH = r"D:\pycharm_codes\MCM2025_codes\clean_boys_data.csv"

# === 读取数据 ===
df = pd.read_csv(CSV_PATH, encoding="utf-8")
X = df[["检测孕周", "孕妇BMI","体重", "年龄"]].to_numpy(dtype=np.float32)
y = df["Y染色体浓度"].to_numpy(dtype=np.float32)

# === 划分训练/验证 ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 训练 XGBoost ===
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

# === 训练/验证效果 ===
y_pred_tr = xgb.predict(X_train)
print("Train  R² = %.3f" % r2_score(y_train, y_pred_tr))
print("Train  RMSE = %.5f" % root_mean_squared_error(y_train, y_pred_tr))

y_pred_val = xgb.predict(X_val)
print("Valid  R² = %.3f" % r2_score(y_val, y_pred_val))
print("Valid  RMSE = %.5f" % root_mean_squared_error(y_val, y_pred_val))

'''
# === 构建网格（用来画曲面） ===
ga_range  = np.linspace(df["检测孕周"].min(), df["检测孕周"].max(), 50)
bmi_range = np.linspace(df["孕妇BMI"].min(), df["孕妇BMI"].max(), 50)
GA, BMI = np.meshgrid(ga_range, bmi_range)
grid_X = np.c_[GA.ravel(), BMI.ravel()]
grid_y = xgb.predict(grid_X).reshape(GA.shape)


# =========================================================
# 图 1（含曲面）：透明曲面 + 底面等高线
# =========================================================
fig1 = plt.figure(figsize=(9, 7))
ax1 = fig1.add_subplot(111, projection='3d')
fig1.subplots_adjust(left=0.18, right=0.98, bottom=0.10, top=0.98)

# 散点
ax1.scatter(df["检测孕周"], df["孕妇BMI"], y,
            c="blue", alpha=0.75, s=15, label="原始数据")

# 半透明拟合曲面（避免遮挡）
ax1.plot_surface(GA, BMI, grid_y,
                 cmap="viridis", alpha=0.30, linewidth=0, antialiased=True)

# 在“地面（z = y.min()）”投影等高线，帮助理解曲面形状
z_min = float(y.min())
ax1.contourf(GA, BMI, grid_y, zdir='z', offset=z_min,
             levels=15, cmap="viridis", alpha=0.70)

# 轴/视角/范围
ax1.set_xlabel("检测孕周")
ax1.set_ylabel("孕妇BMI")
ax1.set_zlabel("Y染色体浓度")
ax1.set_title("XGBoost拟合曲面")
ax1.view_init(elev=30, azim=-68)
ax1.set_xlim(ga_range.min(), ga_range.max())
ax1.set_ylim(bmi_range.min(), bmi_range.max())
ax1.set_zlim(z_min, float(y.max()))
plt.tight_layout()
fig1.savefig("图1_含曲面_清晰版.pdf", bbox_inches="tight")
plt.close(fig1)

# =========================================
# 图 2（无曲面）：仅散点（按同一风格导出）
# =========================================
fig2 = plt.figure(figsize=(9, 7))
ax2 = fig2.add_subplot(111, projection='3d')
fig2.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.98)

ax2.scatter(df["检测孕周"], df["孕妇BMI"], y,
            c="blue", alpha=0.85, s=15, label="原始数据")

ax2.set_xlabel("检测孕周")
ax2.set_ylabel("孕妇BMI")
ax2.set_zlabel("Y染色体浓度")
ax2.set_title("原始数据")
ax2.view_init(elev=30, azim=-68)
ax2.set_xlim(ga_range.min(), ga_range.max())
ax2.set_ylim(bmi_range.min(), bmi_range.max())
ax2.set_zlim(z_min, float(y.max()))
plt.tight_layout()
fig2.savefig("图2_仅散点.pdf", bbox_inches="tight")
plt.close(fig2)

print("已生成：图1_含曲面_清晰版.pdf, 图2_仅散点.pdf")
'''