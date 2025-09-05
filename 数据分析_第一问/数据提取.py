import pandas as pd
def main():
    in_file  = "D:\pycharm_codes\MCM2025_codes\男胎数据_final.csv"
    out_file = "男胎数据_问题一.csv"

    # 读取数据
    df = pd.read_csv(in_file, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]  # 清理列名
    required_columns = ["检测孕周","孕妇BMI","Y染色体浓度"]
    df_new = df[required_columns].copy()
    # 保存
    df_new.to_csv(out_file, index=False, encoding="utf-8-sig")
if __name__ == "__main__":
    main()
