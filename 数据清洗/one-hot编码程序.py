import pandas as pd
from pathlib import Path

# === 配置 ===
CSV_PATH = r"D:\pycharm_codes\MCM2025_codes\clean_boys_data.csv"  # ← 改成你的文件路径
ONEHOT_COLS = ["IVF妊娠"]                  # 需要 one-hot 的列名

# === 读取 ===
df = pd.read_csv(CSV_PATH)

# === 检查列是否存在 ===
missing = [c for c in ONEHOT_COLS if c not in df.columns]
if missing:
    raise KeyError(f"以下列在数据中找不到：{missing}")

# === 转类别并 one-hot ===
for c in ONEHOT_COLS:
    df[c] = df[c].astype("category")

df_encoded = pd.get_dummies(df, columns=ONEHOT_COLS, prefix=ONEHOT_COLS, dummy_na=False)

# === 导出 ===
out_path = Path(CSV_PATH).with_name(Path(CSV_PATH).stem + "__onehot.csv")
df_encoded.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Done ->", out_path)
