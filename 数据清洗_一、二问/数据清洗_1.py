import pandas as pd
#转化为数值
def convert_week(week_str):
    try:
        if isinstance(week_str, str) and '+' in week_str:
            week, day = week_str.split('+')
            return int(week) + int(day) / 7
        else:
            return float(week_str)
    except Exception:
        return None

def main():
    df = pd.read_csv(r"D:\pycharm_codes\MCM2025_codes\男胎数据.csv")
    df_sub = df[['检测孕周', '孕妇BMI', 'Y染色体浓度']].copy()
    df_sub['孕周数值'] = df_sub['检测孕周'].apply(convert_week)
    df_sub = df_sub.dropna(subset=['孕周数值'])
    output_file = "男胎数据_处理后.csv"
    df_sub.to_csv(output_file, index=False, encoding='utf-8-sig')

    # === 5. 输出提示信息 ===
    print("✅ 数据处理完成！")
    print("已生成文件:", output_file)
    print("预览前 5 行：")
    print(df_sub.head())

if __name__ == "__main__":
    main()
