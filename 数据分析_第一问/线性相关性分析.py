# -*- coding: utf-8 -*-
"""
偏相关可视化（论文风格）
- 先对 X、Y 分别控制 Z（线性回归取残差）
- 画 rX vs rY 的散点/六边形密度图
- 叠加线性拟合线与 95% 置信带
- 角标展示：偏相关 r、p 值、样本量 n
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# ====== 全局图形风格（简洁清晰） ======
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']   # 你也可换成 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120                         # 交互查看更清晰
plt.rcParams['savefig.dpi'] = 300                        # 导出图片 300dpi

# ====== 数据路径与字段名 ======
CSV_PATH = r"D:\pycharm_codes\MCM2025_codes\数据分析_第一问\男胎数据_问题一.csv"
COL_Y   = "Y染色体浓度"
COL_X1  = "检测孕周"
COL_X2  = "孕妇BMI"

# ====== 工具函数 ======
def residuals_after_control(v, z):
    """对 v ~ [1, z] 回归后返回残差"""
    X = sm.add_constant(z)
    return v - sm.OLS(v, X, missing='drop').fit().fittedvalues

def partial_scatter_plot(x, y, z,
                         x_label="X（控制后的残差）",
                         y_label="Y（控制后的残差）",
                         title="部分回归图（控制后）",
                         out_path=None,
                         use_hexbin=False,
                         gridsize=35):
    """
    画：控制 z 后 x↔y 的可视化 + 线性拟合线 + 95%CI + r/p/n 角标
    """
    # 丢缺失
    df_ = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    x = df_["x"].to_numpy()
    y = df_["y"].to_numpy()
    z = df_["z"].to_numpy()
    n = len(df_)
    if n < 4:
        raise ValueError("样本量过小，无法计算偏相关（n<4）。")

    # 1) 残差
    rx = residuals_after_control(x, z)
    ry = residuals_after_control(y, z)

    # 2) 偏相关（残差间 Pearson 相关）
    r, p = stats.pearsonr(rx, ry)

    # 3) 拟合线 + 95%CI（对 ry ~ [1, rx]）
    X = sm.add_constant(rx)
    fit = sm.OLS(ry, X).fit()
    x_grid = np.linspace(np.nanmin(rx), np.nanmax(rx), 200)
    Xg = sm.add_constant(x_grid)
    pred = fit.get_prediction(Xg).summary_frame(alpha=0.05)  # mean_ci_lower/upper
    y_hat = pred["mean"].to_numpy()
    y_lo  = pred["mean_ci_lower"].to_numpy()
    y_hi  = pred["mean_ci_upper"].to_numpy()

    # 4) 作图
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    if use_hexbin:
        hb = ax.hexbin(rx, ry, gridsize=gridsize, mincnt=1)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("点密度")
    else:
        ax.scatter(rx, ry, s=14, alpha=0.65)

    ax.plot(x_grid, y_hat, lw=2)              # 拟合线
    ax.fill_between(x_grid, y_lo, y_hi, alpha=0.18, lw=0)  # 95%CI

    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.grid(alpha=0.3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # 角标：r / p / n
    note = f"r = {r:.3f}   p = {p:.2e}   样本数 = {n}"
    ax.text(0.02, 0.98, note, transform=ax.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.9))

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    plt.show()

    return {"r": r, "p": p, "n": n}

# ====== 读取数据并出两张图 ======
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    y   = df[COL_Y].to_numpy()
    x1  = df[COL_X1].to_numpy()  # 例：检测孕周
    x2  = df[COL_X2].to_numpy()  # 例：BMI

    # A) 控制 BMI，考察“检测孕周 ↔ Y”
    partial_scatter_plot(
        x=x1, y=y, z=x2,
        x_label=f"{COL_X1}（控制{COL_X2}后的残差）",
        y_label=f"{COL_Y}（控制{COL_X2}后的残差）",
        title=f"控制{COL_X2}后：{COL_X1} 与 {COL_Y}",
        out_path="图_A_控制BMI_检测孕周_vs_Y浓度.pdf",
        use_hexbin=False  # 点多时可改 True
    )

    # B) 控制 检测孕周，考察“BMI ↔ Y”
    partial_scatter_plot(
        x=x2, y=y, z=x1,
        x_label=f"{COL_X2}（控制{COL_X1}后的残差）",
        y_label=f"{COL_Y}（控制{COL_X1}后的残差）",
        title=f"控制{COL_X1}后：{COL_X2} 与 {COL_Y}",
        out_path="图_B_控制孕周_BMI_vs_Y浓度.pdf",
        use_hexbin=False,   # 用六边形密度，直观显示密集区
        gridsize=40
    )
# ====== 多元线性回归模块（精简版：仅 r 与 R²） ======
def multiple_linear_regression(df, y_col, x_cols,
                               title="多元线性回归：实际值 vs 预测值",
                               out_path_fig="MLR_实际_vs_预测.png"):
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy import stats

    data = df[[y_col] + x_cols].dropna()
    y = data[y_col].to_numpy()
    X = sm.add_constant(data[x_cols])

    model = sm.OLS(y, X).fit()
    y_hat = model.fittedvalues.to_numpy()

    r, _ = stats.pearsonr(y, y_hat)
    r2 = float(model.rsquared)

    print(f"r = {r:.6f}, R² = {r2:.6f}")

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(y_hat, y, s=16, alpha=0.7)
    lo, hi = min(y_hat.min(), y.min()), max(y_hat.max(), y.max())
    ax.plot([lo, hi], [lo, hi], lw=1)
    ax.set_xlabel("预测值 $\\hat{y}$")
    ax.set_ylabel("实际值 $y$")
    ax.set_title(title)
    ax.text(0.02, 0.98, f"r = {r:.3f}   R² = {r2:.3f}   n = {len(data)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.9))

    fig.tight_layout()
    if out_path_fig:
        fig.savefig(out_path_fig, bbox_inches="tight")
    plt.show()

    return {"r": float(r), "R2": float(r2), "样本数": int(len(data))}


if __name__ == "__main__":
    try:
        X_cols_demo = [COL_X1, COL_X2]
        multiple_linear_regression(
            df=df,
            y_col=COL_Y,
            x_cols=X_cols_demo,
            title=f"多元线性回归：{COL_Y} ~ {X_cols_demo}（实际值 vs 预测值）",
            out_path_fig="图_C_多元线性回归.pdf"
        )
    except Exception as e:
        print("多元线性回归示例运行失败：", e)