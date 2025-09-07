"""
测试集成模型在Q4数据上的性能指标
包括Accuracy、Precision、Recall、AUC、F1 score和NPV
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Q4PerformanceTester:
    """Q4数据性能测试器"""
    
    def __init__(self):
        self.ensemble_model = None
        self.preprocessing_components = None
        self.model_info = None
        
    def load_models(self):
        """加载模型和预处理组件"""
        print("=== 加载模型组件 ===")
        
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
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ 文件未找到: {e}")
            return False
        except Exception as e:
            print(f"❌ 加载模型时出错: {e}")
            return False
    
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
        """特征预处理"""
        print("进行特征预处理...")
        
        # 获取数值特征
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['序号', 'is_abnormal']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        # 确保使用训练时相同的特征
        if 'feature_cols' in self.preprocessing_components:
            train_feature_cols = self.preprocessing_components['feature_cols']
            available_features = [col for col in train_feature_cols if col in df.columns]
            feature_cols = available_features
        
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
        
        return X_scaled
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_prob, threshold_name="balanced"):
        """计算全面的性能指标"""
        print(f"\n=== {threshold_name}阈值性能指标 ===")
        
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算特异性和NPV
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
        
        # 计算AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception as e:
            auc = 0.0
            print(f"警告: 无法计算AUC: {e}")
        
        # 打印结果
        print(f"🎯 核心指标:")
        print(f"   Accuracy (准确率):     {accuracy:.4f} ({accuracy:.1%})")
        print(f"   Precision (精确率):    {precision:.4f} ({precision:.1%})")
        print(f"   Recall (召回率):       {recall:.4f} ({recall:.1%})")
        print(f"   F1 Score:             {f1:.4f}")
        print(f"   AUC:                  {auc:.4f}")
        print(f"   NPV (阴性预测值):      {npv:.4f} ({npv:.1%})")
        print(f"   Specificity (特异性): {specificity:.4f} ({specificity:.1%})")
        
        print(f"\n📊 混淆矩阵:")
        print(f"                实际")
        print(f"预测    正常    异常")
        print(f"正常     {tn:3d}     {fn:3d}")
        print(f"异常     {fp:3d}     {tp:3d}")
        
        # 返回指标字典
        metrics = {
            'threshold_name': threshold_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'npv': npv,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        return metrics
    
    def test_q4_performance(self, filepath='Q4/clean_girls_data_Q4.csv'):
        """测试Q4数据性能"""
        print("🚀 开始Q4数据性能测试")
        print("="*60)
        
        # 1. 加载模型
        if not self.load_models():
            return None
        
        # 2. 加载Q4数据
        print(f"\n=== 加载Q4数据: {filepath} ===")
        data = pd.read_csv(filepath)
        print(f"数据集大小: {data.shape}")
        
        # 检查标签分布
        if 'is_abnormal' not in data.columns:
            print("❌ Q4数据中未找到真实标签")
            return None
        
        y_true = data['is_abnormal']
        print(f"标签分布: 正常 {(y_true==0).sum()} | 异常 {(y_true==1).sum()}")
        print(f"异常率: {y_true.mean():.1%}")
        
        # 3. 特征工程
        processed_data = self.advanced_feature_engineering(data)
        
        # 4. 特征预处理
        X = self.preprocess_features(processed_data)
        print(f"预处理后特征维度: {X.shape}")
        
        # 5. 模型预测
        print(f"\n=== 模型预测 ===")
        probabilities = self.ensemble_model.predict_proba(X)[:, 1]
        
        # 6. 使用不同阈值进行预测和评估
        thresholds = self.model_info['optimal_thresholds']
        all_metrics = {}
        predictions_dict = {}
        
        for scenario, threshold in thresholds.items():
            y_pred = (probabilities > threshold).astype(int)
            predictions_dict[scenario] = y_pred
            metrics = self.calculate_comprehensive_metrics(y_true, y_pred, probabilities, scenario)
            all_metrics[scenario] = metrics
        
        # 6.5. 生成风险分级
        risk_categories = self.generate_risk_categories(probabilities)
        
        # 6.6. 创建详细预测结果CSV
        self.create_prediction_results_csv(data, processed_data, probabilities, predictions_dict, risk_categories, y_true)
        
        # 7. 创建性能对比表
        print(f"\n" + "="*60)
        print("📈 性能对比汇总")
        print("="*60)
        
        metrics_df = pd.DataFrame(all_metrics).T
        display_cols = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc', 'npv']
        
        print("\n指标对比表:")
        print(metrics_df[display_cols].round(4))
        
        # 8. 保存结果
        metrics_df.to_csv('Q4_performance_metrics.csv', encoding='utf-8-sig')
        print(f"\n📁 性能指标已保存至: Q4_performance_metrics.csv")
        
        # 9. 创建可视化
        self.create_performance_visualization(all_metrics, y_true, probabilities)
        
        print("\n📁 生成的文件:")
        print("- Q4_performance_metrics.csv: 性能指标对比")
        print("- Q4_detailed_prediction_results.csv: 详细预测结果")
        print("- Q4_prediction_summary.csv: 预测汇总")
        print("- Q4_false_positive_cases.csv: 假阳性案例分析")
        print("- Q4_false_negative_cases.csv: 假阴性案例分析")
        print("- Q4_performance_analysis.png: 性能可视化图表")
        
        return all_metrics, metrics_df
    
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
        
        return risk_categories
    
    def create_prediction_results_csv(self, original_data, processed_data, probabilities, predictions_dict, risk_categories, y_true):
        """创建详细的预测结果CSV文件"""
        print("\n=== 生成预测结果CSV ===")
        
        # 创建结果DataFrame，从原始数据开始
        results_df = original_data.copy()
        
        # 添加预测相关列
        results_df['异常概率'] = probabilities
        results_df['风险等级'] = risk_categories
        
        # 添加不同阈值下的预测结果
        for scenario, predictions in predictions_dict.items():
            results_df[f'{scenario}_预测值'] = predictions
            results_df[f'{scenario}_预测结果'] = ['异常' if p == 1 else '正常' for p in predictions]
        
        # 添加一些关键的工程特征
        if '最大Z值' in processed_data.columns:
            results_df['最大Z值'] = processed_data['最大Z值']
        if 'Z值超2.0_count' in processed_data.columns:
            results_df['Z值超2.0_count'] = processed_data['Z值超2.0_count']
        if '高龄产妇' in processed_data.columns:
            results_df['高龄产妇'] = processed_data['高龄产妇']
        
        # 添加预测准确性分析（针对每个阈值）
        for scenario in predictions_dict.keys():
            y_pred = predictions_dict[scenario]
            results_df[f'{scenario}_预测正确'] = (y_true == y_pred).astype(int)
            results_df[f'{scenario}_预测类型'] = ''
            
            # 标记预测类型
            for i in range(len(y_true)):
                if y_true.iloc[i] == 1 and y_pred[i] == 1:
                    results_df.loc[i, f'{scenario}_预测类型'] = 'TP'
                elif y_true.iloc[i] == 0 and y_pred[i] == 0:
                    results_df.loc[i, f'{scenario}_预测类型'] = 'TN'
                elif y_true.iloc[i] == 0 and y_pred[i] == 1:
                    results_df.loc[i, f'{scenario}_预测类型'] = 'FP'
                else:  # y_true == 1 and y_pred == 0
                    results_df.loc[i, f'{scenario}_预测类型'] = 'FN'
        
        # 保存详细预测结果
        results_df.to_csv('Q4_detailed_prediction_results.csv', index=False, encoding='utf-8-sig')
        print("详细预测结果已保存至: Q4_detailed_prediction_results.csv")
        
        # 创建简化的汇总表
        summary_cols = ['序号', '孕妇代码', '年龄', '孕妇BMI', '异常概率', '风险等级', 
                       'balanced_预测结果', 'is_abnormal', 'balanced_预测正确', 'balanced_预测类型']
        
        # 添加最大Z值（如果存在）
        if '最大Z值' in results_df.columns:
            summary_cols.insert(-4, '最大Z值')
        
        summary_df = results_df[summary_cols]
        summary_df.to_csv('Q4_prediction_summary.csv', index=False, encoding='utf-8-sig')
        print("预测汇总已保存至: Q4_prediction_summary.csv")
        
        # 创建错误分析报告
        self.create_error_analysis_report(results_df, y_true, predictions_dict)
        
        return results_df
    
    def create_error_analysis_report(self, results_df, y_true, predictions_dict):
        """创建错误分析报告"""
        print("\n=== 错误分析 ===")
        
        error_analysis = {}
        
        for scenario in predictions_dict.keys():
            y_pred = predictions_dict[scenario]
            
            # 假阳性分析
            fp_mask = (y_true == 0) & (y_pred == 1)
            fp_cases = results_df[fp_mask]
            
            # 假阴性分析  
            fn_mask = (y_true == 1) & (y_pred == 0)
            fn_cases = results_df[fn_mask]
            
            error_analysis[scenario] = {
                'false_positive_count': fp_mask.sum(),
                'false_negative_count': fn_mask.sum(),
                'fp_avg_probability': results_df.loc[fp_mask, '异常概率'].mean() if fp_mask.sum() > 0 else 0,
                'fn_avg_probability': results_df.loc[fn_mask, '异常概率'].mean() if fn_mask.sum() > 0 else 0,
            }
            
            print(f"\n{scenario}阈值错误分析:")
            print(f"  假阳性: {fp_mask.sum()} 例, 平均概率: {error_analysis[scenario]['fp_avg_probability']:.4f}")
            print(f"  假阴性: {fn_mask.sum()} 例, 平均概率: {error_analysis[scenario]['fn_avg_probability']:.4f}")
        
        # 保存错误案例详情
        if 'balanced' in predictions_dict:
            balanced_pred = predictions_dict['balanced']
            fp_mask = (y_true == 0) & (balanced_pred == 1)
            fn_mask = (y_true == 1) & (balanced_pred == 0)
            
            if fp_mask.sum() > 0:
                fp_cases = results_df[fp_mask][['序号', '孕妇代码', '年龄', '孕妇BMI', '异常概率', '风险等级', 'is_abnormal']]
                fp_cases.to_csv('Q4_false_positive_cases.csv', index=False, encoding='utf-8-sig')
                print(f"假阳性案例已保存至: Q4_false_positive_cases.csv ({fp_mask.sum()} 例)")
                
            if fn_mask.sum() > 0:
                fn_cases = results_df[fn_mask][['序号', '孕妇代码', '年龄', '孕妇BMI', '异常概率', '风险等级', 'is_abnormal']]
                fn_cases.to_csv('Q4_false_negative_cases.csv', index=False, encoding='utf-8-sig')
                print(f"假阴性案例已保存至: Q4_false_negative_cases.csv ({fn_mask.sum()} 例)")
        
    def create_performance_visualization(self, all_metrics, y_true, probabilities):
        """创建性能可视化图表"""
        print(f"\n=== 生成性能可视化 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 指标对比雷达图
        scenarios = list(all_metrics.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc', 'npv']
        
        # 准备雷达图数据
        angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax1 = plt.subplot(2, 2, 1, projection='polar')
        colors = ['blue', 'green', 'red']
        
        for i, scenario in enumerate(scenarios):
            values = [all_metrics[scenario][metric] for metric in metrics_names]
            values += values[:1]  # 闭合图形
            
            ax1.plot(angles, values, 'o-', linewidth=2, label=scenario, color=colors[i])
            ax1.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(['准确率', '精确率', '召回率', '特异性', 'F1', 'AUC', 'NPV'])
        ax1.set_ylim(0, 1)
        ax1.set_title('性能指标雷达图', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 2. ROC曲线（如果可能）
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, probabilities)
            auc_score = all_metrics['balanced']['auc']
            
            axes[0, 1].plot(fpr, tpr, 'b-', label=f'ROC曲线 (AUC = {auc_score:.4f})')
            axes[0, 1].plot([0, 1], [0, 1], 'r--', label='随机分类器')
            axes[0, 1].set_xlabel('假阳性率 (1-特异性)')
            axes[0, 1].set_ylabel('真阳性率 (敏感性)')
            axes[0, 1].set_title('ROC曲线')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'无法绘制ROC曲线\n{str(e)}', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('ROC曲线 (绘制失败)')
        
        # 3. 概率分布直方图
        normal_probs = probabilities[y_true == 0]
        abnormal_probs = probabilities[y_true == 1]
        
        axes[1, 0].hist(normal_probs, bins=30, alpha=0.7, color='green', label='正常样本', density=True)
        axes[1, 0].hist(abnormal_probs, bins=30, alpha=0.7, color='red', label='异常样本', density=True)
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='阈值=0.5')
        axes[1, 0].set_xlabel('异常概率')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].set_title('异常概率分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 混淆矩阵热力图
        balanced_cm = np.array([[all_metrics['balanced']['confusion_matrix']['tn'], 
                                all_metrics['balanced']['confusion_matrix']['fn']],
                               [all_metrics['balanced']['confusion_matrix']['fp'], 
                                all_metrics['balanced']['confusion_matrix']['tp']]])
        
        sns.heatmap(balanced_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['正常', '异常'], yticklabels=['正常', '异常'],
                   ax=axes[1, 1])
        axes[1, 1].set_title('混淆矩阵 (平衡阈值)')
        axes[1, 1].set_xlabel('预测标签')
        axes[1, 1].set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.savefig('Q4_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 性能可视化图表已保存至: Q4_performance_analysis.png")

if __name__ == "__main__":
    # 创建测试器并运行
    tester = Q4PerformanceTester()
    
    # 测试Q4数据性能
    results = tester.test_q4_performance('Q4/clean_girls_data_Q4.csv')
    
    if results is not None:
        all_metrics, metrics_df = results
        print(f"\n✅ Q4性能测试完成！")
        print(f"\n🎯 推荐使用平衡阈值，主要指标:")
        balanced = all_metrics['balanced']
        print(f"   准确率 (Accuracy): {balanced['accuracy']:.4f}")
        print(f"   精确率 (Precision): {balanced['precision']:.4f}")
        print(f"   召回率 (Recall): {balanced['recall']:.4f}")
        print(f"   F1分数: {balanced['f1_score']:.4f}")
        print(f"   AUC: {balanced['auc']:.4f}")
        print(f"   阴性预测值 (NPV): {balanced['npv']:.4f}")
    else:
        print("❌ Q4性能测试失败")
