# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

CSV_PATH = r"D:\pycharm_codes\MCM2025_codes\数据分析_第一问\男胎数据_问题一.csv"

# 读取数据 -> numpy（y 一维）
df = pd.read_csv(CSV_PATH, encoding="utf-8")
X = df[["检测孕周","孕妇BMI"]].to_numpy(dtype=np.float32)
y = df["Y染色体浓度"].to_numpy(dtype=np.float32)   # shape: (n,)

# 划分训练/验证
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练 XGBoost
xgb = XGBRegressor(
    n_estimators=700,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

# —— 训练集效果 —— #
y_pred_tr = xgb.predict(X_train)
print("Train  R² = %.3f" % r2_score(y_train, y_pred_tr))
print("Train  RMSE = %.5f" % root_mean_squared_error(y_train, y_pred_tr))

# —— 验证集效果（保留你已有的对照）—— #
y_pred_val = xgb.predict(X_val)
print("Valid  RMSE = %.5f" % root_mean_squared_error(y_val, y_pred_val))


#=== （可选）交互式 3D 曲面，可保留你的原注释块逻辑 ===
ga_range  = np.linspace(df["检测孕周"].min(), df["检测孕周"].max(), 60)
bmi_range = np.linspace(df["孕妇BMI"].min(), df["孕妇BMI"].max(), 60)
GA, BMI = np.meshgrid(ga_range, bmi_range)
grid_X = np.c_[GA.ravel(), BMI.ravel()]
grid_y = xgb.predict(grid_X).reshape(GA.shape)

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=df["检测孕周"].to_numpy(),
    y=df["孕妇BMI"].to_numpy(),
    z=df["Y染色体浓度"].to_numpy(),
    mode="markers",
    marker=dict(size=3, color="blue", opacity=0.6),
    name="原始数据"
))
fig.add_trace(go.Surface(
    x=ga_range, y=bmi_range, z=grid_y,
    colorscale="Viridis", opacity=0.7,
    name="XGBoost拟合曲面"
))
fig.update_layout(
    scene=dict(xaxis_title="检测孕周", yaxis_title="孕妇BMI", zaxis_title="Y染色体浓度"),
    title="XGBoost 拟合效果（交互式3D）", width=900, height=700
)
fig.show()
