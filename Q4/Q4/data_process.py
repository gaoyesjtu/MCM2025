# %%
import pandas as pd

# %%
df = pd.read_csv(r'1题\girls_final_cleaned_data.csv', encoding='utf-8-sig')

# %%
# 查看数据基本信息
print("数据形状:", df.shape)
print("\n列名:")
print(df.columns.tolist())
print("\n前5行数据:")
print(df.head())
print("\n数据类型:")
print(df.dtypes)
print("\n缺失值统计:")
print(df.isnull().sum())

# %%
# 重新加载原始数据
df_clean = df.copy()
print(f"原始数据形状: {df_clean.shape}")

# 1. 删除完全空的列
empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
print(f"完全空的列: {empty_cols}")
if empty_cols:
    df_clean = df_clean.drop(columns=empty_cols)
    print(f"删除空列后数据形状: {df_clean.shape}")

# 2. 正确处理目标变量 - 创建is_abnormal指示变量
target_col = '染色体的非整倍体'
print(f"\n=== 正确处理目标变量 ===")
print(f"原始'{target_col}'列:")
print(f"  总样本数: {len(df_clean)}")
print(f"  有异常值的样本: {df_clean[target_col].notna().sum()}")
print(f"  正常样本(缺失值): {df_clean[target_col].isna().sum()}")

# 创建二分类目标变量
df_clean['is_abnormal'] = df_clean[target_col].notna().astype(int)
print(f"\n创建is_abnormal指示变量:")
print(f"  正常(0): {(df_clean['is_abnormal'] == 0).sum()}")
print(f"  异常(1): {(df_clean['is_abnormal'] == 1).sum()}")

# 同时保留原始的异常类型信息用于多分类
df_clean['abnormal_type'] = df_clean[target_col].fillna('正常')
print(f"\n异常类型分布:")
print(df_clean['abnormal_type'].value_counts())

print(f"\n所有样本都保留，数据形状: {df_clean.shape}")

# 8. 处理缺失值
print("\n=== 缺失值处理策略 ===")

# 对于重要的特征，根据缺失比例决定处理策略
high_missing_threshold = 20  # 20%以上缺失率
key_features = [
    'X染色体的Z值', 
    '21号染色体的Z值', 
    '18号染色体的Z值', 
    '13号染色体的Z值',
    '21号染色体的GC含量',
    '18号染色体的GC含量', 
    '13号染色体的GC含量',
    'GC含量',
    '原始读段数',
    '唯一比对的读段数',
    '在参考基因组上比对的比例',
    '重复读段的比例',
    '被过滤掉读段数的比例',
    '孕妇BMI'
]

for feature in key_features:
    if feature in df_clean.columns:
        missing_pct = df_clean[feature].isnull().sum() / len(df_clean) * 100
        if missing_pct > high_missing_threshold:
            print(f"{feature}: {missing_pct:.1f}% 缺失，考虑删除该特征")
        elif missing_pct > 0:
            print(f"{feature}: {missing_pct:.1f}% 缺失，使用中位数填充")
            # 用中位数填充数值型特征的缺失值
            if df_clean[feature].dtype in ['float64', 'int64']:
                median_val = df_clean[feature].median()
                df_clean[feature].fillna(median_val, inplace=True)
                print(f"  使用中位数 {median_val:.4f} 填充")

# 删除缺失率过高的特征
features_to_drop = []
for feature in key_features:
    if feature in df_clean.columns:
        missing_pct = df_clean[feature].isnull().sum() / len(df_clean) * 100
        if missing_pct > high_missing_threshold:
            features_to_drop.append(feature)

if features_to_drop:
    print(f"\n删除高缺失率特征: {features_to_drop}")
    df_clean = df_clean.drop(columns=features_to_drop)

print(f"\n处理缺失值后数据形状: {df_clean.shape}")

# 9. 最终检查
print("\n=== 最终数据检查 ===")
print(f"最终数据形状: {df_clean.shape}")
print(f"总缺失值数量: {df_clean.isnull().sum().sum()}")

# 检查关键列是否还存在
final_key_features = [f for f in key_features if f in df_clean.columns and f not in features_to_drop]
print(f"\n保留的关键特征 ({len(final_key_features)}个):")
for i, feature in enumerate(final_key_features):
    print(f"{i+1:2d}. {feature}")

print(f"\n目标变量 '{target_col}' 的值分布:")
print(df_clean[target_col].value_counts())

# %%
# 3. 处理关键特征的缺失值
print("\n=== 处理关键特征缺失值 ===")


missing_info = {}
for feature in key_features:
    if feature in df_clean.columns:
        missing_count = df_clean[feature].isnull().sum()
        missing_pct = missing_count / len(df_clean) * 100
        missing_info[feature] = {'count': missing_count, 'percentage': missing_pct}
        print(f"{feature}: {missing_count} ({missing_pct:.1f}%)")

# 4. 处理缺失值
print(f"\n=== 缺失值处理策略 ===")
high_missing_threshold = 30  # 30%以上缺失率考虑删除

for feature in key_features:
    if feature in df_clean.columns:
        missing_pct = df_clean[feature].isnull().sum() / len(df_clean) * 100
        if missing_pct > high_missing_threshold:
            print(f"{feature}: {missing_pct:.1f}% 缺失，缺失率过高")
        elif missing_pct > 0:
            # 用中位数填充数值型特征的缺失值
            if df_clean[feature].dtype in ['float64', 'int64']:
                median_val = df_clean[feature].median()
                df_clean[feature].fillna(median_val, inplace=True)
                print(f"{feature}: 用中位数 {median_val:.4f} 填充 {missing_pct:.1f}% 的缺失值")

# 5. 检查处理后的数据质量
print(f"\n=== 数据质量检查 ===")
print(f"数据形状: {df_clean.shape}")
print(f"总缺失值数量: {df_clean.isnull().sum().sum()}")

remaining_missing = df_clean.isnull().sum()
if remaining_missing.sum() > 0:
    print(f"仍有缺失值的列:")
    for col, missing in remaining_missing.items():
        if missing > 0:
            pct = missing / len(df_clean) * 100
            print(f"  {col}: {missing} ({pct:.1f}%)")

# 检查目标变量分布
print(f"\n目标变量分布:")
print(f"is_abnormal分布:")
print(df_clean['is_abnormal'].value_counts())
print(f"\nabnormal_type分布:")
print(df_clean['abnormal_type'].value_counts())

# %%
# 6. 数据类型优化和派生特征
print("\n=== 数据类型优化和特征工程 ===")

# 确保数值型列是正确的数据类型
numeric_features = [
    'X染色体的Z值', '21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值',
    '21号染色体的GC含量', '18号染色体的GC含量', '13号染色体的GC含量', 'GC含量',
    '原始读段数', '唯一比对的读段数', '在参考基因组上比对的比例', 
    '重复读段的比例', '被过滤掉读段数的比例', '孕妇BMI', '年龄', '身高', '体重', '检测孕周'
]

for feature in numeric_features:
    if feature in df_clean.columns:
        df_clean[feature] = pd.to_numeric(df_clean[feature], errors='coerce')

# 添加派生特征
print("添加派生特征:")

# 1. 读段比例特征
if '唯一比对的读段数' in df_clean.columns and '原始读段数' in df_clean.columns:
    df_clean['唯一比对读段比例'] = df_clean['唯一比对的读段数'] / df_clean['原始读段数']
    print("✓ 唯一比对读段比例")

# 2. Z值绝对值（距离正常值的偏离程度）
z_features = ['X染色体的Z值', '21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值']
for z_feature in z_features:
    if z_feature in df_clean.columns:
        abs_feature = z_feature.replace('Z值', 'Z值绝对值')
        df_clean[abs_feature] = abs(df_clean[z_feature])
        print(f"✓ {abs_feature}")

# 3. 综合染色体风险评分（Z值绝对值的加权和）
if all(f in df_clean.columns for f in ['21号染色体的Z值绝对值', '18号染色体的Z值绝对值', '13号染色体的Z值绝对值']):
    df_clean['染色体风险评分'] = (
        df_clean['21号染色体的Z值绝对值'] * 1.0 +  # 21号染色体权重1.0
        df_clean['18号染色体的Z值绝对值'] * 1.0 +  # 18号染色体权重1.0
        df_clean['13号染色体的Z值绝对值'] * 1.0    # 13号染色体权重1.0
    ) / 3
    print("✓ 染色体风险评分")

# 4. BMI分类
if '孕妇BMI' in df_clean.columns:
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return '偏瘦'
        elif bmi < 24:
            return '正常'
        elif bmi < 28:
            return '超重'
        else:
            return '肥胖'
    
    df_clean['BMI分类'] = df_clean['孕妇BMI'].apply(categorize_bmi)
    print("✓ BMI分类")
    print(f"  BMI分类分布: {df_clean['BMI分类'].value_counts().to_dict()}")

print(f"\n添加派生特征后数据形状: {df_clean.shape}")

# %%
# 7. 最终数据检查和保存
print("\n=== 最终数据检查 ===")
print(f"最终数据形状: {df_clean.shape}")
print(f"样本数量: {df_clean.shape[0]}")
print(f"特征数量: {df_clean.shape[1]}")

# 检查目标变量
print(f"\n目标变量分布:")
print("二分类目标 (is_abnormal):")
abnormal_dist = df_clean['is_abnormal'].value_counts()
for value, count in abnormal_dist.items():
    label = "正常" if value == 0 else "异常"
    percentage = count / len(df_clean) * 100
    print(f"  {label}({value}): {count} ({percentage:.1f}%)")

print(f"\n多分类目标 (abnormal_type):")
type_dist = df_clean['abnormal_type'].value_counts()
for value, count in type_dist.items():
    percentage = count / len(df_clean) * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# 检查关键特征
print(f"\n关键特征列表:")
key_feature_categories = {
    "Z值特征": ['X染色体的Z值', '21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值'],
    "Z值绝对值": ['X染色体的Z值绝对值', '21号染色体的Z值绝对值', '18号染色体的Z值绝对值', '13号染色体的Z值绝对值'],
    "GC含量": ['GC含量', '21号染色体的GC含量', '18号染色体的GC含量', '13号染色体的GC含量'],
    "读段特征": ['原始读段数', '唯一比对的读段数', '唯一比对读段比例'],
    "质量指标": ['在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例'],
    "生理指标": ['孕妇BMI', 'BMI分类'],
    "风险评分": ['染色体风险评分'],
    "目标变量": ['is_abnormal', 'abnormal_type']
}

for category, features in key_feature_categories.items():
    available_features = [f for f in features if f in df_clean.columns]
    print(f"  {category} ({len(available_features)}个): {available_features}")

# 数据质量最终检查
print(f"\n数据质量:")
print(f"  缺失值总数: {df_clean.isnull().sum().sum()}")
print(f"  重复行数: {df_clean.duplicated().sum()}")

# 异常值检查
print(f"\n异常值检查:")
for feature in ['孕妇BMI', '染色体风险评分']:
    if feature in df_clean.columns:
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_clean[(df_clean[feature] < lower_bound) | (df_clean[feature] > upper_bound)]
        print(f"  {feature}: {len(outliers)} 个异常值 ({len(outliers)/len(df_clean)*100:.1f}%)")

# 保存清洗后的数据
output_file = r'cleaned_data\clean_girls_data_Q4.csv'
df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 清洗后的数据已保存到: {output_file}")

# 保存关键特征列表（用于后续建模）
key_modeling_features = []
for features in key_feature_categories.values():
    key_modeling_features.extend([f for f in features if f in df_clean.columns and f not in ['is_abnormal', 'abnormal_type']])

print(f"\n建模特征数量: {len(key_modeling_features)}")
print("建模特征列表已准备就绪")


