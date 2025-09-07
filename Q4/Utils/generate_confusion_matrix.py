"""
生成集成模型在平衡阈值下的混淆矩阵图片
专注于生成未归一化的混淆矩阵可视化
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_preprocessing():
    """加载集成模型和预处理组件"""
    print("=== 加载模型和预处理组件 ===")
    
    try:
        # 加载集成模型
        with open('Model/ensemble_model.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)
        print("✅ 集成模型加载成功")
        
        # 加载预处理组件
        with open('Model/ensemble_preprocessing.pkl', 'rb') as f:
            preprocessing_components = pickle.load(f)
        print("✅ 预处理组件加载成功")
        
        # 默认平衡阈值
        balanced_threshold = 0.5
        print(f"使用平衡阈值: {balanced_threshold}")
        
        return ensemble_model, preprocessing_components, balanced_threshold
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        return None, None, None

def advanced_feature_engineering(df):
    """高级特征工程（与训练时完全一致）"""
    print("进行特征工程...")
    
    df_eng = df.copy()
    
    # 1. Z值相关特征
    z_cols = ['X染色体的Z值绝对值', '21号染色体的Z值绝对值', '18号染色体的Z值绝对值', '13号染色体的Z值绝对值']
    
    # 基础统计特征
    df_eng['最大Z值'] = df_eng[z_cols].max(axis=1)
    df_eng['平均Z值'] = df_eng[z_cols].mean(axis=1)
    df_eng['Z值标准差'] = df_eng[z_cols].std(axis=1)
    df_eng['Z值变异系数'] = df_eng[z_cols].std(axis=1) / (df_eng[z_cols].mean(axis=1) + 1e-8)
    df_eng['Z值范围'] = df_eng[z_cols].max(axis=1) - df_eng[z_cols].min(axis=1)
    
    # 异常指标（多阈值）
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for threshold in thresholds:
        df_eng[f'Z值超{threshold}_count'] = (df_eng[z_cols] > threshold).sum(axis=1)
        df_eng[f'Z值超{threshold}_比例'] = df_eng[f'Z值超{threshold}_count'] / len(z_cols)
    
    # 特定染色体组合异常
    df_eng['常染色体异常'] = (
        (df_eng['21号染色体的Z值绝对值'] > 2.0) |
        (df_eng['18号染色体的Z值绝对值'] > 2.0) |
        (df_eng['13号染色体的Z值绝对值'] > 2.0)
    ).astype(int)
    
    df_eng['X染色体异常'] = (df_eng['X染色体的Z值绝对值'] > 2.0).astype(int)
    
    # 2. GC含量特征
    gc_cols = ['13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
    
    df_eng['平均GC含量'] = df_eng[gc_cols].mean(axis=1)
    df_eng['GC标准差'] = df_eng[gc_cols].std(axis=1)
    df_eng['GC偏差'] = abs(df_eng['GC含量'] - df_eng['平均GC含量'])
    
    # GC含量异常指标
    df_eng['GC含量异常'] = (
        (df_eng['GC偏差'] > df_eng['GC偏差'].quantile(0.9)) |
        (df_eng['GC标准差'] > df_eng['GC标准差'].quantile(0.9))
    ).astype(int)
    
    # 3. X染色体特异性特征
    df_eng['X染色体浓度异常'] = (
        (df_eng['X染色体浓度'] < 0.45) | (df_eng['X染色体浓度'] > 0.55)
    ).astype(int)
    df_eng['X染色体浓度偏差'] = abs(df_eng['X染色体浓度'] - 0.5)
    
    # 4. 测序质量特征
    df_eng['测序深度充足'] = (df_eng['唯一比对的读段数'] > 3000000).astype(int)
    df_eng['比对质量良好'] = (df_eng['在参考基因组上比对的比例'] > 0.75).astype(int)
    df_eng['重复率低'] = (df_eng['重复读段的比例'] < 0.2).astype(int)
    
    # 5. 临床风险因子
    df_eng['高龄产妇'] = (df_eng['年龄'] >= 35).astype(int)
    df_eng['极高龄产妇'] = (df_eng['年龄'] >= 40).astype(int)
    df_eng['BMI过高'] = (df_eng['孕妇BMI'] > 30).astype(int)
    df_eng['BMI肥胖'] = (df_eng['孕妇BMI'] > 35).astype(int)
    
    # 6. 交互特征
    df_eng['年龄_Z值交互'] = df_eng['年龄'] * df_eng['最大Z值']
    df_eng['BMI_Z值交互'] = df_eng['孕妇BMI'] * df_eng['最大Z值']
    df_eng['GC_Z值交互'] = df_eng['GC偏差'] * df_eng['最大Z值']
    
    # 7. 多重风险评分
    df_eng['保守风险评分'] = (
        df_eng['最大Z值'] * 0.4 +
        df_eng['Z值超2.5_count'] * 0.3 +
        df_eng['X染色体浓度偏差'] * 0.2 +
        df_eng['高龄产妇'] * 0.1
    )
    
    df_eng['平衡风险评分'] = (
        df_eng['最大Z值'] * 0.3 +
        df_eng['Z值超2.0_count'] * 0.25 +
        df_eng['Z值超1.5_count'] * 0.2 +
        df_eng['GC偏差'] * 0.15 +
        df_eng['高龄产妇'] * 0.1
    )
    
    df_eng['敏感风险评分'] = (
        df_eng['最大Z值'] * 0.25 +
        df_eng['Z值超1.5_count'] * 0.3 +
        df_eng['Z值超1.0_count'] * 0.25 +
        df_eng['GC偏差'] * 0.1 +
        df_eng['高龄产妇'] * 0.05 +
        df_eng['BMI过高'] * 0.05
    )
    
    # 8. 复合异常指标
    df_eng['多重异常指标'] = (
        df_eng['Z值超2.0_count'] +
        df_eng['GC含量异常'] +
        df_eng['X染色体浓度异常'] +
        df_eng['高龄产妇']
    )
    
    print(f"特征工程完成，特征数量: {df_eng.shape[1]}")
    return df_eng

def preprocess_features(df, preprocessing_components):
    """特征预处理"""
    print("=== 特征预处理 ===")
    
    # 获取数值特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['序号', 'is_abnormal']
    feature_cols = [col for col in numeric_features if col not in exclude_cols]
    
    # 确保使用训练时相同的特征
    if 'feature_cols' in preprocessing_components:
        train_feature_cols = preprocessing_components['feature_cols']
        available_features = [col for col in train_feature_cols if col in df.columns]
        missing_features = [col for col in train_feature_cols if col not in df.columns]
        
        if missing_features:
            print(f"警告: 缺少训练时的特征: {missing_features}")
        
        feature_cols = available_features
    
    print(f"使用特征数量: {len(feature_cols)}")
    
    X = df[feature_cols]
    
    # 1. 缺失值处理
    imputer = preprocessing_components['imputer']
    X_imputed = imputer.transform(X)
    
    # 2. 特征选择
    selector = preprocessing_components['selector']
    X_selected = selector.transform(X_imputed)
    
    # 3. 标准化
    scaler = preprocessing_components['scaler']
    X_scaled = scaler.transform(X_selected)
    
    print(f"预处理后特征维度: {X_scaled.shape}")
    return X_scaled

def generate_confusion_matrix_plot(y_true, y_pred, balanced_threshold):
    """生成混淆矩阵图片"""
    print("=== 生成混淆矩阵图片 ===")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算性能指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 创建图片
    plt.figure(figsize=(10, 8))
    
    # 绘制混淆矩阵热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['正常', '异常'], 
               yticklabels=['正常', '异常'],
               cbar_kws={'label': '样本数量'})
    
    plt.title(f'混淆矩阵 (平衡阈值: {balanced_threshold:.3f})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=14)
    plt.ylabel('真实标签', fontsize=14)
    
    # 在图的右侧添加性能指标
    metrics_text = f"""性能指标:
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}

真阴性(TN): {cm[0,0]}
假阳性(FP): {cm[0,1]}
假阴性(FN): {cm[1,0]}
真阳性(TP): {cm[1,1]}"""
    
    plt.figtext(0.75, 0.6, metrics_text, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细结果
    print("\n📊 混淆矩阵结果:")
    print(f"真阴性 (TN): {cm[0,0]} - 正确预测为正常的样本")
    print(f"假阳性 (FP): {cm[0,1]} - 错误预测为异常的正常样本") 
    print(f"假阴性 (FN): {cm[1,0]} - 错误预测为正常的异常样本")
    print(f"真阳性 (TP): {cm[1,1]} - 正确预测为异常的样本")
    
    print("\n📈 性能指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return cm

def main():
    """主函数"""
    print("🎯 开始生成混淆矩阵")
    print("="*50)
    
    # 1. 加载模型
    ensemble_model, preprocessing_components, balanced_threshold = load_model_and_preprocessing()
    if ensemble_model is None:
        return
    
    # 2. 加载数据
    print("\n=== 加载数据 ===")
    df = pd.read_csv('clean_girls_data_Q4.csv')
    print(f"数据集大小: {df.shape}")
    
    # 检查是否有真实标签
    if 'is_abnormal' not in df.columns:
        print("❌ 数据中没有 'is_abnormal' 列，无法生成混淆矩阵")
        return
    
    y_true = np.array(df['is_abnormal'].astype(int))
    total_abnormal = int(y_true.sum())
    abnormal_rate = float(y_true.mean())
    print(f"真实异常样本: {total_abnormal} / {len(y_true)} ({abnormal_rate:.1%})")
    
    # 3. 特征工程
    processed_df = advanced_feature_engineering(df)
    
    # 4. 特征预处理
    X = preprocess_features(processed_df, preprocessing_components)
    
    # 5. 生成预测
    print(f"\n=== 使用平衡阈值 {balanced_threshold:.3f} 进行预测 ===")
    probabilities = ensemble_model.predict_proba(X)
    abnormal_probabilities = probabilities[:, 1]
    
    # 使用平衡阈值生成预测
    y_pred = (abnormal_probabilities > balanced_threshold).astype(int)
    
    positive_count = y_pred.sum()
    positive_rate = positive_count / len(y_pred)
    print(f"预测为阳性的样本数: {positive_count} / {len(y_pred)} ({positive_rate:.1%})")
    
    # 6. 生成混淆矩阵图片
    generate_confusion_matrix_plot(y_true, y_pred, balanced_threshold)
    
    print("\n🎉 混淆矩阵生成完成！")
    print("生成的文件: confusion_matrix.png")

if __name__ == "__main__":
    main()
