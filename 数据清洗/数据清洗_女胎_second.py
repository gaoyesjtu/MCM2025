import pandas as pd

def find_column(cols, keywords):
    """在列名列表中查找包含关键词的列"""
    for c in cols:
        if all(k in c for k in keywords):
            return c
    raise KeyError(f"未找到列，关键词: {keywords}")

def main():
    in_file  = "D:\pycharm_codes\MCM2025_codes\数据清洗\女胎数据1.csv"
    out_file = "女胎数据_去重后.csv"

    # 读取数据
    df = pd.read_csv(in_file, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]  # 清理列名
    print("列名预览:", df.columns.tolist())  # 便于确认

    # 自动匹配列名
    col_code  = find_column(df.columns, ["孕妇代码"])
    col_week  = find_column(df.columns, ["孕周"])    # 检测孕周/孕周数值
    col_reads = find_column(df.columns, ["唯一", "比对", "读段"])  # 唯一比对的读段数

    # 分组筛选
    df_new = df.loc[df.groupby([col_code, col_week])[col_reads].idxmax()]

    # 保存
    df_new.to_csv(out_file, index=False, encoding="utf-8-sig")
if __name__ == "__main__":
    main()
