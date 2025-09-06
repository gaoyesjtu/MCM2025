# -*- coding: utf-8 -*-
"""
单变量"相关强度"数值输出：
- 指标：互信息 MI、|Spearman ρ|、|Kendall τ|
- 排序：按 MI 从强到弱
"""
import numpy as np
import pandas as pd
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
OUTFILE = "相关特性_数值结果.csv"
# =================================

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
        rho, p_rho = spearmanr(x[mask], y[mask])
        abs_rho = abs(rho)
    except Exception:
        abs_rho = np.nan
        p_rho = np.nan

    # |Kendall τ|
    try:
        tau, p_tau = kendalltau(x[mask], y[mask], nan_policy="omit")
        abs_tau = abs(tau)
    except Exception:
        abs_tau = np.nan
        p_tau = np.nan

    # MI
    try:
        mi = mutual_info_regression(x[mask].reshape(-1,1), y[mask], random_state=0)[0]
    except Exception:
        mi = np.nan

    rows.append([col, n, abs_rho, p_rho, abs_tau, p_tau, mi])

# 创建结果数据框
res = pd.DataFrame(rows, columns=[
    "feature", "n_effective", "abs_spearman_rho", "p_spearman",
    "abs_kendall_tau", "p_kendall", "mutual_info"
])

if res.empty:
    raise ValueError("有效特征集合为空（样本量不足、列名不匹配或全为 NaN）。")

# 排序
res = res.sort_values(["mutual_info","abs_spearman_rho","abs_kendall_tau"],
                      ascending=[False, False, False]).reset_index(drop=True)

# 输出结果到控制台
print("=" * 80)
print(f"目标变量: {TARGET}")
print("=" * 80)
print("特征与目标变量的相关性分析结果 (按互信息MI降序排列):")
print("=" * 80)

for idx, row in res.iterrows():
    print(f"{idx+1}. {row['feature']}:")
    print(f"   有效样本数: {row['n_effective']}")
    print(f"   互信息(MI): {row['mutual_info']:.6f}")
    print(f"   |Spearman ρ|: {row['abs_spearman_rho']:.6f} (p={row['p_spearman']:.6f})")
    print(f"   |Kendall τ|: {row['abs_kendall_tau']:.6f} (p={row['p_kendall']:.6f})")
    print("-" * 40)

# 保存结果到CSV文件
res.to_csv(OUTFILE, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到: {Path(OUTFILE).resolve()}")

# 检查是否有NaN值
nan_cols = res.columns[2:]
if (nan_cols.size > 0) and res[nan_cols].isna().any().any():
    print("注意：存在 NaN 指标，请检查数据。")