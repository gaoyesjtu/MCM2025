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
    in_file  = r"D:\pycharm_codes\MCM2025_codes\原始数据与题面\女胎数据.csv"
    out_file = "女胎数据1.csv"
    df = pd.read_csv(in_file, encoding="utf-8-sig")
    df["检测孕周"] = df["检测孕周"].apply(week_to_float)
    # 保存
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
if __name__ == "__main__":
    main()
