# -*- coding: utf-8 -*-
"""
单变量“相关强度”可视化（仅一张 PDF）：
- 指标：互信息 MI、|Spearman ρ|、|Kendall τ|
- 排序：按 MI 从强到弱
- 配色：#0095FF（MI）、#019092（|ρ|）、#6FDCB5（|τ|）
- 字体：微软雅黑（自动回退），白底 PDF，图例不被裁剪
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path

# ============== 配置 ==============
CSV_PATH = r"D:\pycharm_codes\MCM2025_codes\数据清洗\clean_boys_data__onehot.csv"  # ← 用你的路径
TARGET   = "Y染色体浓度"  # ← 目标列名
INCLUDE_FEATURES = [
    "检测孕周","孕妇BMI","年龄","身高","体重",
    "IVF妊娠_IUI（人工授精）","IVF妊娠_IVF（试管婴儿）","IVF妊娠_自然受孕",
    "怀孕次数","生产次数"
]
MIN_SAMPLES = 10
OUTFILE = "相关特性_强度条形图.pdf"

# —— 视觉参数（柱更宽、组更松）——
BAR_H     = 0.34   # 条形高度
SHIFT     = 0.42   # 组内三条的位移
GROUP_GAP = 1.45   # 组间距缩放（>1 更松）
FIG_W     = 9.0
TOP_K = 6  # 只显示前 6 个
# =================================

# ---- 字体/导出设置（更稳健） ----
# 首选微软雅黑，若系统没有则回退
fallback_fonts = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
matplotlib.rcParams["font.sans-serif"] = fallback_fonts
matplotlib.rcParams["axes.unicode_minus"] = False
# 让 PDF 嵌入可编辑文本并强制白底
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["savefig.facecolor"] = "white"
matplotlib.rcParams["figure.facecolor"] = "white"

matplotlib.rcParams.update({
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "legend.frameon": False,
    "axes.grid": False
})

# 颜色（按你的要求）
COLOR_MI       = "#0095FF"  # MI
COLOR_SPEARMAN = "#019092"  # |ρ|
COLOR_KENDALL  = "#6FDCB5"  # |τ|

# ---- 读取数据 ----
df = pd.read_csv(CSV_PATH)
assert TARGET in df.columns, f"找不到目标列：{TARGET}"
y = df[TARGET].values

miss = [c for c in INCLUDE_FEATURES if c not in df.columns]
if miss:
    raise KeyError(f"以下列在数据中不存在：{miss}")

# ---- 计算 MI / |ρ| / |τ| ----
rows = []
for col in INCLUDE_FEATURES:
    x = df[col].values
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < MIN_SAMPLES:
        continue

    # |Spearman ρ|
    try:
        rho, _ = spearmanr(x[mask], y[mask])
        abs_rho = abs(rho)
    except Exception:
        abs_rho = np.nan

    # |Kendall τ|
    try:
        tau, _ = kendalltau(x[mask], y[mask], nan_policy="omit")
        abs_tau = abs(tau)
    except Exception:
        abs_tau = np.nan

    # MI
    try:
        mi = mutual_info_regression(x[mask].reshape(-1,1), y[mask], random_state=0)[0]
    except Exception:
        mi = np.nan

    rows.append([col, n, abs_rho, abs_tau, mi])

res = pd.DataFrame(rows, columns=["feature","n_effective","abs_spearman_rho","abs_kendall_tau","mutual_info"])
if res.empty:
    raise ValueError("有效特征集合为空（样本量不足、列名不匹配或全为 NaN）。")

# 排序
res = res.sort_values(["mutual_info","abs_spearman_rho","abs_kendall_tau"],
                      ascending=[False, False, False]).reset_index(drop=True)

res = res.head(TOP_K).reset_index(drop=True)
# —— 若出现 NaN，绘图用 0 显示并提示（避免整条不显示）——
nan_cols = res.columns[2:]
vis = res.copy()
for c in nan_cols:
    vis[c] = np.where(np.isfinite(vis[c]), vis[c], 0.0)

# ---- 绘图 ----
H = max(4.8, 0.56 * len(res))
ypos = np.arange(len(res)) * GROUP_GAP
ypos = ypos[::-1]  # 从上到下

fig, ax = plt.subplots(figsize=(FIG_W, H))

b1 = ax.barh(ypos + SHIFT, vis["mutual_info"].values,
             height=BAR_H, color=COLOR_MI, edgecolor="#222222", linewidth=0.6, label="互信息 MI")
b2 = ax.barh(ypos,         vis["abs_spearman_rho"].values,
             height=BAR_H, color=COLOR_SPEARMAN, edgecolor="#222222", linewidth=0.6, label="|Spearman ρ|")
b3 = ax.barh(ypos - SHIFT, vis["abs_kendall_tau"].values,
             height=BAR_H, color=COLOR_KENDALL, edgecolor="#222222", linewidth=0.6, label="|Kendall τ|")

ax.set_yticks(ypos)
ax.set_yticklabels(res["feature"].values, fontsize=10)
ax.set_xlabel("相关强度（非线性 / 非参数）", fontsize=11)
ax.set_title(f"单变量相关强度Top6（目标：{TARGET}）", fontsize=13, pad=10)

# 图例放画布内，防止被裁剪；加大句柄
leg = ax.legend(loc="lower right", fontsize=10, handlelength=2.2, borderaxespad=0.8)
for lh in leg.legend_handles:
    lh.set_linewidth(2.0)

# 去除上/右脊
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# 在 MI 上标注数值
for y_i, v in zip(ypos + SHIFT, res["mutual_info"].values):
    if np.isfinite(v):
        ax.text(v, y_i, f" {v:.3f}", va="center", ha="left", fontsize=9, color=COLOR_MI)

fig.tight_layout()
plt.savefig(OUTFILE, format="pdf", bbox_inches="tight", facecolor="white")
plt.close()

print("Saved:", Path(OUTFILE).resolve())
if (nan_cols.size > 0) and vis[nan_cols].isna().any().any():
    print("注意：存在 NaN 指标，绘图时按 0 处理（仅用于显示）。请检查数据。")
