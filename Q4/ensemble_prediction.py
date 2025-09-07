"""
使用集成模型对新数据进行预测
完整的预测流水线，包括数据预处理、模型加载、预测和结果分析
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnsemblePredictionSystem:
    """集成模型预测系统"""
    
    def __init__(self):
        self.ensemble_model = None
        self.preprocessing_components = None
        self.model_info = None
        self.predictions = None
        self.data = None
        
    def load_ensemble_model(self):
        """加载集成模型和预处理组件"""
        print("=== 加载集成模型和预处理组件 ===")
        
        try:
            # 加载集成模型
            with open('ensemble_model.pkl', 'rb') as f:
                self.ensemble_model = pickle.load(f)
            print("✅ 集成模型加载成功")
            
            # 加载预处理组件
            with open('ensemble_preprocessing.pkl', 'rb') as f:
                self.preprocessing_components = pickle.load(f)
            print("✅ 预处理组件加载成功")
            
            # 加载模型信息
            with open('ensemble_model_info.json', 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            print("✅ 模型信息加载成功")
            
            print("\n📋 模型信息:")
            print(f"   集成组成: {self.model_info['ensemble_composition']}")
            print(f"   投票方式: {self.model_info['voting']}")
            print(f"   创建时间: {self.model_info['creation_date']}")
            print(f"   阈值设置: {self.model_info['optimal_thresholds']}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ 文件未找到: {e}")
            print("请确保已运行 create_ensemble_model.py 生成模型文件")
            return False
        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")
            return False
    
    def load_and_preprocess_data(self, filepath='clean_girls_data_Q4.csv'):
        """加载并预处理新数据"""
        print(f"\n=== 加载和预处理数据: {filepath} ===")
        
        # 加载数据
        self.data = pd.read_csv(filepath)
        print(f"数据集大小: {self.data.shape}")
        print(f"列名: {list(self.data.columns)}")
        
        # 检查是否有真实标签
        has_labels = 'is_abnormal' in self.data.columns
        if has_labels:
            print(f"包含真实标签: 异常样本 {self.data['is_abnormal'].sum()} / {len(self.data)} = {self.data['is_abnormal'].mean():.1%}")
        else:
            print("未包含真实标签，将进行纯预测")
        
        # 进行特征工程（与训练时保持一致）
        processed_data = self.advanced_feature_engineering(self.data)
        
        return processed_data, has_labels
    
    def advanced_feature_engineering(self, df):
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
    
    def preprocess_features(self, df):
        """特征预处理（缺失值、特征选择、标准化）"""
        print("\n=== 特征预处理 ===")
        
        if self.preprocessing_components is None:
            print("❌ 预处理组件未加载")
            return None, None
        
        # 获取数值特征
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['序号', 'is_abnormal']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        # 确保使用训练时相同的特征
        if 'feature_cols' in self.preprocessing_components:
            train_feature_cols = self.preprocessing_components['feature_cols']
            # 只保留训练时使用的特征
            available_features = [col for col in train_feature_cols if col in df.columns]
            missing_features = [col for col in train_feature_cols if col not in df.columns]
            
            if missing_features:
                print(f"警告: 缺少训练时的特征: {missing_features}")
            
            feature_cols = available_features
        
        print(f"使用特征数量: {len(feature_cols)}")
        
        X = df[feature_cols]
        
        # 1. 缺失值处理
        imputer = self.preprocessing_components['imputer']
        X_imputed = imputer.transform(X)
        
        # 2. 特征选择
        selector = self.preprocessing_components['selector']
        X_selected = selector.transform(X_imputed)
        
        # 3. 标准化
        scaler = self.preprocessing_components['scaler']
        X_scaled = scaler.transform(X_selected)
        
        print(f"预处理后特征维度: {X_scaled.shape}")
        
        return X_scaled, feature_cols
    
    def predict_with_ensemble(self, X):
        """使用集成模型进行预测"""
        print("\n=== 集成模型预测 ===")
        
        if self.ensemble_model is None:
            print("❌ 集成模型未加载")
            return None, None, None
        
        if self.model_info is None:
            print("❌ 模型信息未加载")
            return None, None, None
        
        # 获取概率预测
        probabilities = self.ensemble_model.predict_proba(X)
        abnormal_probabilities = probabilities[:, 1]
        
        # 使用不同阈值进行预测
        thresholds = self.model_info['optimal_thresholds']
        predictions = {}
        
        for scenario, threshold in thresholds.items():
            pred = (abnormal_probabilities > threshold).astype(int)
            predictions[scenario] = pred
            
            positive_count = pred.sum()
            positive_rate = positive_count / len(pred)
            print(f"{scenario}阈值 ({threshold:.4f}): {positive_count} 个阳性样本 ({positive_rate:.1%})")
        
        # 默认使用平衡阈值
        default_predictions = predictions['balanced']
        
        return abnormal_probabilities, predictions, default_predictions
    
    def generate_risk_categories(self, probabilities):
        """生成风险分级"""
        print("\n=== 风险分级 ===")
        
        risk_categories = []
        risk_levels = {'低风险': 0, '中风险': 0, '中-高风险': 0, '高风险': 0}
        
        for prob in probabilities:
            if prob >= 0.8:
                category = '高风险'
            elif prob >= 0.5:
                category = '中-高风险'
            elif prob >= 0.2:
                category = '中风险'
            else:
                category = '低风险'
            
            risk_categories.append(category)
            risk_levels[category] += 1
        
        print("风险分级统计:")
        for level, count in risk_levels.items():
            print(f"  {level}: {count} 例 ({count/len(probabilities):.1%})")
        
        return risk_categories, risk_levels
    
    def create_prediction_report(self, df, probabilities, predictions, risk_categories, has_labels=False):
        """创建预测报告"""
        print("\n=== 生成预测报告 ===")
        
        # 创建结果DataFrame
        results_df = df.copy()
        results_df['异常概率'] = probabilities
        results_df['风险等级'] = risk_categories
        
        # 添加不同阈值下的预测结果
        for scenario, pred in predictions.items():
            results_df[f'{scenario}_预测'] = pred
            results_df[f'{scenario}_预测结果'] = ['异常' if p == 1 else '正常' for p in pred]
        
        # 如果有真实标签，计算性能指标
        if has_labels:
            y_true = df['is_abnormal']
            
            print("\n📊 性能评估:")
            for scenario, pred in predictions.items():
                f1 = f1_score(y_true, pred)
                precision = precision_score(y_true, pred, zero_division=0)
                recall = recall_score(y_true, pred, zero_division=0)
                
                print(f"\n{scenario}阈值:")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
        
        # 保存详细结果
        results_df.to_csv('ensemble_prediction_results.csv', index=False, encoding='utf-8-sig')
        print(f"\n详细预测结果已保存至: ensemble_prediction_results.csv")
        
        # 创建汇总报告
        summary_cols = ['序号', '孕妇代码', '年龄', '孕妇BMI', '最大Z值', '异常概率', '风险等级', 'balanced_预测结果']
        if has_labels:
            summary_cols.append('is_abnormal')
        
        summary_df = results_df[summary_cols]
        summary_df.to_csv('prediction_summary.csv', index=False, encoding='utf-8-sig')
        print(f"预测汇总已保存至: prediction_summary.csv")
        
        return results_df
    
    def create_visualizations(self, probabilities, risk_categories, predictions, has_labels=False, y_true=None):
        """创建可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        if self.model_info is None:
            print("警告: 模型信息未加载，使用默认阈值")
            default_thresholds = {'balanced': 0.5, 'high_recall': 0.3, 'high_precision': 0.7}
        else:
            default_thresholds = self.model_info['optimal_thresholds']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 异常概率分布
        axes[0, 0].hist(probabilities, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=default_thresholds['balanced'], 
                          color='red', linestyle='--', label='平衡阈值')
        axes[0, 0].set_xlabel('异常概率')
        axes[0, 0].set_ylabel('样本数量')
        axes[0, 0].set_title('异常概率分布')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 风险等级分布
        risk_counts = pd.Series(risk_categories).value_counts()
        colors = ['green', 'yellow', 'orange', 'red']
        axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, 
                            colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('风险等级分布')
        
        # 3. 不同阈值下的预测统计
        threshold_names = list(predictions.keys())
        positive_counts = [pred.sum() for pred in predictions.values()]
        
        bars = axes[0, 2].bar(threshold_names, positive_counts, 
                             color=['blue', 'green', 'red'], alpha=0.7)
        axes[0, 2].set_title('不同阈值下的阳性预测数')
        axes[0, 2].set_ylabel('阳性样本数')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, positive_counts):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{count}', ha='center', va='bottom')
        
        # 4. 高风险样本的特征分析
        high_risk_mask = np.array(risk_categories) == '高风险'
        if high_risk_mask.sum() > 0:
            high_risk_probs = probabilities[high_risk_mask]
            other_probs = probabilities[~high_risk_mask]
            
            axes[1, 0].boxplot([other_probs, high_risk_probs], 
                              labels=['其他', '高风险'])
            axes[1, 0].set_title('高风险 vs 其他样本的概率分布')
            axes[1, 0].set_ylabel('异常概率')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 如果有真实标签，绘制混淆矩阵
        if has_labels and y_true is not None:
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, predictions['balanced'])
            axes[1, 1].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[1, 1].set_title('混淆矩阵 (平衡阈值)')
            
            # 添加数值标签
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > thresh else "black")
            
            axes[1, 1].set_ylabel('真实标签')
            axes[1, 1].set_xlabel('预测标签')
            axes[1, 1].set_xticks([0, 1])
            axes[1, 1].set_yticks([0, 1])
            axes[1, 1].set_xticklabels(['正常', '异常'])
            axes[1, 1].set_yticklabels(['正常', '异常'])
        
        # 6. 概率校准图
        sorted_indices = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_indices]
        
        axes[1, 2].plot(range(len(sorted_probs)), sorted_probs, 'b-', alpha=0.7)
        axes[1, 2].axhline(y=default_thresholds['balanced'], 
                          color='red', linestyle='--', label='平衡阈值')
        axes[1, 2].axhline(y=default_thresholds['high_recall'], 
                          color='green', linestyle='--', label='高召回阈值')
        axes[1, 2].axhline(y=default_thresholds['high_precision'], 
                          color='blue', linestyle='--', label='高精确阈值')
        axes[1, 2].set_xlabel('样本索引 (按概率排序)')
        axes[1, 2].set_ylabel('异常概率')
        axes[1, 2].set_title('样本概率分布 (排序)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_prediction_pipeline(self, filepath='Q4/clean_girls_data_Q4.csv'):
        """运行完整的预测流水线"""
        print("🚀 开始集成模型预测流水线")
        print("="*60)
        
        # 1. 加载模型
        if not self.load_ensemble_model():
            return None
        
        # 2. 加载和预处理数据
        processed_data, has_labels = self.load_and_preprocess_data(filepath)
        
        # 3. 特征预处理
        X, feature_cols = self.preprocess_features(processed_data)
        
        # 4. 模型预测
        probabilities, predictions, default_pred = self.predict_with_ensemble(X)
        
        # 5. 风险分级
        risk_categories, risk_levels = self.generate_risk_categories(probabilities)
        
        # 6. 生成预测报告
        results_df = self.create_prediction_report(
            processed_data, probabilities, predictions, risk_categories, has_labels
        )
        
        # 7. 创建可视化
        y_true = processed_data['is_abnormal'].values if has_labels else None
        self.create_visualizations(probabilities, risk_categories, predictions, has_labels, y_true)
        
        print("\n🎉 预测流水线完成！")
        print("生成的文件:")
        print("- ensemble_prediction_results.csv: 详细预测结果")
        print("- prediction_summary.csv: 预测汇总")
        print("- ensemble_prediction_analysis.png: 可视化分析图")
        
        return results_df, probabilities, predictions, risk_categories

if __name__ == "__main__":
    # 创建预测系统并运行
    predictor = EnsemblePredictionSystem()
    
    # 运行完整预测流水线
    results = predictor.run_prediction_pipeline('Q4/clean_girls_data_Q4.csv')
    
    if results is not None:
        results_df, probabilities, predictions, risk_categories = results
        print(f"\n✅ 预测完成！共处理 {len(results_df)} 个样本")
        print(f"高风险样本: {risk_categories.count('高风险')} 例")
        print(f"中-高风险样本: {risk_categories.count('中-高风险')} 例")
    else:
        print("❌ 预测失败")
