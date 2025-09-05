
import re
import sys
import pandas as pd

def main():
    in_file  = r"D:\pycharm_codes\MCM2025_codes\原始数据与题面\boys_final_cleaned_data.csv"
    out_file = "男胎数据问题一、二.csv"

    # 读取并清理列名（去除BOM与空白）
    df = pd.read_csv(in_file, encoding="utf-8-sig")
    required = ["检测孕周", "孕妇BMI", "Y染色体浓度"]
    # 只保留三列并生成数值孕周
    df_sub = df[required].copy()
    # 保存
    df_sub.to_csv(out_file, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
