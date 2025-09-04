"""
文件: process_data_w.py
用途: 提取第一问所需三列，并将孕周从 '11w+3' 或 '11w' 转为连续周数
输出: 男胎数据_处理后.csv
"""

import re
import sys
import pandas as pd

RE_W_PLUS_D = re.compile(r'^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$')   # 11w+3
RE_W_ONLY   = re.compile(r'^\s*(\d+)\s*[wW]\s*$')                # 11w
RE_NUM_ONLY = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*$')             # 11 或 11.5

def week_to_float(val):
    """将孕周文本转为连续周数(float)。支持: 11w+3 / 11w / 11"""
    if pd.isna(val):
        return None
    s = str(val).strip()

    m = RE_W_PLUS_D.fullmatch(s)
    if m:
        w = int(m.group(1))
        d = int(m.group(2))
        return w + d / 7.0

    m = RE_W_ONLY.fullmatch(s)
    if m:
        return float(m.group(1))

    m = RE_NUM_ONLY.fullmatch(s)
    if m:
        return float(m.group(1))

    return None  # 不符合格式时返回None

def main():
    in_file  = "D:\pycharm_codes\MCM2025_codes\男胎数据.csv"
    out_file = "男胎数据问题一、二.csv"

    # 读取并清理列名（去除BOM与空白）
    df = pd.read_csv(in_file, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    required = ["检测孕周", "孕妇BMI", "Y染色体浓度"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("❌ 下列必需列名未找到：", missing)
        print("现有列名：", list(df.columns))
        sys.exit(1)

    # 只保留三列并生成数值孕周
    df_sub = df[required].copy()
    df_sub["孕周数值"] = df_sub["检测孕周"].apply(week_to_float)

    # 丢弃孕周无法解析的行（通常极少）
    before = len(df_sub)
    df_sub = df_sub.dropna(subset=["孕周数值"])
    after = len(df_sub)
    df_sub=df_sub.drop("检测孕周", axis=1)

    # 保存
    df_sub.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("✅ 已生成:", out_file)
    print(f"记录数: {before} -> {after}")
    print(df_sub.head())

if __name__ == "__main__":
    main()
