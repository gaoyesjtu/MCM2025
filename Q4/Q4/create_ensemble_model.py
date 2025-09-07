import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnsembleModelGenerator:
    """集成模型生成器 - 基于真实的网格搜索结果"""
    
    def __init__(self):
        self.ensemble_model = None
        self.optimal_thresholds = {}
        self.model_results = {}
        self.best_models = {}
        
    def load_grid_search_results(self):
        """加载网格搜索结果"""
        print("=== 加载网格搜索结果 ===")
        
        try:
            # 尝试运行增强版网格搜索系统获取模型
            from enhanced_grid_search_system import EnhancedGridSearchSystem
            
            print("运行增强版网格搜索系统...")
            system = EnhancedGridSearchSystem()
            df, performance, model_results = system.run_full_analysis()
            
            self.model_results = model_results
            self.df = df
            
            print("成功加载网格搜索结果！")
            return True
            
        except Exception as e:
            print(f"无法运行网格搜索系统: {e}")
            print("将使用模拟数据演示集成模型...")
            return False
    
    def create_ensemble_model(self):
        """创建集成学习模型"""
        print("\n=== 创建集成学习模型 ===")
        
        if not self.model_results:
            print("没有可用的模型结果，无法创建集成模型")
            return None
        
        # 按F1 score排序，选择Top 3模型
        sorted_models = sorted(self.model_results.items(), 
                             key=lambda x: x[1]['optimal_f1'], reverse=True)
        
        print("模型性能排序:")
        for i, (name, results) in enumerate(sorted_models[:5]):
            print(f"{i+1}. {name}: F1={results['optimal_f1']:.4f}")
        
        # 选择前3个模型创建集成
        top_3_models = sorted_models[:3]
        
        # 提取模型和权重
        estimators = []
        f1_scores = []
        
        for name, results in top_3_models:
            estimators.append((name.lower().replace(' ', '_'), results['model']))
            f1_scores.append(results['optimal_f1'])
        
        # 计算基于F1 score的权重
        weights = np.array(f1_scores) / np.sum(f1_scores)
        
        print(f"\n集成模型组成:")
        for i, ((name, model), weight) in enumerate(zip(estimators, weights)):
            original_name = top_3_models[i][0]
            print(f"- {original_name}: 权重 {weight:.3f}")
        
        # 创建加权投票分类器
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # 软投票：基于概率
            weights=weights
        )
        
        # 训练集成模型
        print("\n训练集成模型...")
        
        # 准备训练数据（使用和单个模型相同的预处理）
        from enhanced_grid_search_system import EnhancedGridSearchSystem
        temp_system = EnhancedGridSearchSystem()
        
        # 加载和预处理数据
        df = temp_system.load_and_preprocess_data()
        df = temp_system.level1_screening(df)
        
        # 准备特征
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['序号', 'is_abnormal', '一级判定', '一级原因']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['is_abnormal']
        
        # 使用相同的预处理流程
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.preprocessing import RobustScaler
        
        # 处理缺失值
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 特征选择
        selector = SelectKBest(score_func=f_classif, k=min(25, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # 标准化
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # 训练集成模型
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # 评估集成模型
        y_pred = self.ensemble_model.predict(X_test_scaled)
        y_pred_proba = self.ensemble_model.predict_proba(X_test_scaled)[:, 1]
        
        ensemble_f1 = f1_score(y_test, y_pred)
        ensemble_precision = precision_score(y_test, y_pred)
        ensemble_recall = recall_score(y_test, y_pred)
        
        print(f"\n集成模型测试集性能:")
        print(f"F1 Score: {ensemble_f1:.4f}")
        print(f"Precision: {ensemble_precision:.4f}")
        print(f"Recall: {ensemble_recall:.4f}")
        
        # 与最佳单模型比较
        best_single_f1 = max(f1_scores)
        improvement = ensemble_f1 - best_single_f1
        print(f"\n相比最佳单模型的提升: {improvement:.4f} ({improvement/best_single_f1*100:+.1f}%)")
        
        # 交叉验证评估
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train, cv=cv, scoring='f1')
        print(f"交叉验证F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 优化阈值
        self.optimize_ensemble_threshold(y_test, y_pred_proba)
        
        # 保存相关对象
        self.imputer = imputer
        self.selector = selector
        self.scaler = scaler
        self.feature_cols = feature_cols
        
        return self.ensemble_model
    
    def optimize_ensemble_threshold(self, y_true, y_pred_proba):
        """优化集成模型的判定阈值"""
        print("\n=== 集成模型阈值优化 ===")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 找到最优F1阈值
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_f1 = f1_scores[optimal_idx]
        
        # 高召回率阈值（召回率 >= 80%）
        high_recall_indices = np.where(recall >= 0.8)[0]
        if len(high_recall_indices) > 0:
            best_idx = high_recall_indices[np.argmax(f1_scores[high_recall_indices])]
            high_recall_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.3
        else:
            best_idx = np.argmax(recall)
            high_recall_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.3
        
        # 高精确率阈值（精确率 >= 60%）
        high_precision_indices = np.where(precision >= 0.6)[0]
        if len(high_precision_indices) > 0:
            best_idx = high_precision_indices[np.argmax(f1_scores[high_precision_indices])]
            high_precision_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.7
        else:
            best_idx = np.argmax(precision)
            high_precision_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.7
        
        self.optimal_thresholds = {
            'balanced': optimal_threshold,
            'high_recall': high_recall_threshold,
            'high_precision': high_precision_threshold
        }
        
        print(f"平衡阈值: {optimal_threshold:.4f} (F1: {optimal_f1:.4f})")
        print(f"高召回阈值: {high_recall_threshold:.4f}")
        print(f"高精确阈值: {high_precision_threshold:.4f}")
        
        return self.optimal_thresholds
    
    def save_ensemble_model(self):
        """保存集成模型和相关组件"""
        print("\n=== 保存集成模型 ===")
        
        if self.ensemble_model is None:
            print("没有集成模型可保存")
            return False
        
        # 保存集成模型
        with open('ensemble_model.pkl', 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        
        # 保存预处理组件
        preprocessing_components = {
            'imputer': self.imputer,
            'selector': self.selector,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'optimal_thresholds': self.optimal_thresholds
        }
        
        with open('ensemble_preprocessing.pkl', 'wb') as f:
            pickle.dump(preprocessing_components, f)
        
        # 保存模型信息
        model_info = {
            'ensemble_composition': [name for name, _ in self.ensemble_model.estimators],
            'weights': list(self.ensemble_model.weights) if hasattr(self.ensemble_model, 'weights') else None,
            'voting': self.ensemble_model.voting,
            'optimal_thresholds': self.optimal_thresholds,
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open('ensemble_model_info.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print("集成模型已保存:")
        print("- ensemble_model.pkl: 集成模型")
        print("- ensemble_preprocessing.pkl: 预处理组件")
        print("- ensemble_model_info.json: 模型信息")
        
        return True
    
    def load_and_use_ensemble_model(self, new_data=None):
        """加载并使用集成模型进行预测"""
        print("\n=== 加载并使用集成模型 ===")
        
        try:
            # 加载模型
            with open('ensemble_model.pkl', 'rb') as f:
                ensemble_model = pickle.load(f)
            
            # 加载预处理组件
            with open('ensemble_preprocessing.pkl', 'rb') as f:
                components = pickle.load(f)
            
            print("成功加载集成模型和预处理组件")
            
            # 如果没有提供新数据，使用测试数据
            if new_data is None:
                print("使用测试数据进行演示...")
                # 这里可以加载一些测试数据
                pass
            
            return ensemble_model, components
            
        except FileNotFoundError:
            print("集成模型文件不存在，请先运行 create_ensemble_model()")
            return None, None
    
    def create_ensemble_analysis_report(self):
        """创建集成模型分析报告"""
        print("\n=== 集成模型分析报告 ===")
        
        if not self.model_results:
            print("没有模型结果可分析")
            return
        
        # 创建性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 单模型 vs 集成模型 F1 Score对比
        model_names = list(self.model_results.keys())
        f1_scores = [self.model_results[name]['optimal_f1'] for name in model_names]
        
        # 添加集成模型的F1（如果有的话）
        if hasattr(self, 'ensemble_f1'):
            model_names.append('集成模型')
            f1_scores.append(self.ensemble_f1)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars = axes[0, 0].bar(model_names, f1_scores, color=colors, alpha=0.7)
        axes[0, 0].set_title('F1 Score 对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 集成模型权重分布
        if self.ensemble_model is not None:
            estimator_names = [name for name, _ in self.ensemble_model.estimators]
            weights = self.ensemble_model.weights if hasattr(self.ensemble_model, 'weights') else [1/len(estimator_names)] * len(estimator_names)
            
            pie = axes[0, 1].pie(weights, labels=estimator_names, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('集成模型权重分布', fontsize=14, fontweight='bold')
        
        # 3. 不同阈值下的性能
        if self.optimal_thresholds:
            threshold_names = ['平衡型', '高召回型', '高精确型']
            thresholds = list(self.optimal_thresholds.values())
            
            axes[1, 0].bar(threshold_names, thresholds, color=['blue', 'green', 'red'], alpha=0.7)
            axes[1, 0].set_title('不同场景的最优阈值', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('阈值')
            axes[1, 0].grid(True, alpha=0.3)
            
            for i, (name, threshold) in enumerate(zip(threshold_names, thresholds)):
                axes[1, 0].text(i, threshold + 0.01, f'{threshold:.3f}', 
                               ha='center', va='bottom', fontsize=10)
        
        # 4. 模型复杂度 vs 性能
        training_times = [self.model_results[name]['training_time'] for name in model_names[:-1]]  # 排除集成模型
        f1_scores_single = f1_scores[:-1] if len(f1_scores) > len(training_times) else f1_scores
        
        scatter = axes[1, 1].scatter(training_times, f1_scores_single, 
                                   s=100, alpha=0.7, c=range(len(training_times)), cmap='viridis')
        
        for i, name in enumerate(model_names[:-1]):
            axes[1, 1].annotate(name, (training_times[i], f1_scores_single[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 1].set_xlabel('训练时间 (秒)')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('训练时间 vs 性能', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def create_ensemble_model_pipeline():
    """创建完整的集成模型流水线"""
    print("=== 创建集成模型流水线 ===")
    
    generator = EnsembleModelGenerator()
    
    # 1. 加载网格搜索结果
    success = generator.load_grid_search_results()
    
    if success:
        # 2. 创建集成模型
        ensemble_model = generator.create_ensemble_model()
        
        if ensemble_model is not None:
            # 3. 保存模型
            generator.save_ensemble_model()
            
            # 4. 创建分析报告
            generator.create_ensemble_analysis_report()
            
            print("\n✅ 集成模型创建完成！")
            print("现在你可以使用 ensemble_model.pkl 进行预测了。")
        else:
            print("❌ 集成模型创建失败")
    else:
        print("❌ 无法加载网格搜索结果")
    
    return generator

if __name__ == "__main__":
    # 运行集成模型创建流水线
    generator = create_ensemble_model_pipeline()
