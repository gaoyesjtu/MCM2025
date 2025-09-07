import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix, 
                           f1_score, precision_score, recall_score,
                           precision_recall_curve, make_scorer)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedGridSearchSystem:
    """增强版网格搜索优化的女胎染色体非整倍体判定系统"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = None
        self.imputer = None
        self.selector = None
        self.best_model = None
        self.best_model_name = ""
        self.feature_importance = None
        self.results = {}
        self.optimal_threshold = 0.5
        self.grid_search_results = {}
        self.best_params = {}
        self.model_comparison = None
        
    def load_and_preprocess_data(self, filepath=r"D:\pycharm_codes\MCM2025_codes\Q4\Q4\clean_girls_data_Q4.csv"):
        """加载和预处理数据"""
        print("=== 数据加载和预处理 ===")
        
        # 加载数据
        self.data = pd.read_csv(filepath)
        print(f"数据集大小: {self.data.shape}")
        print(f"异常样本: {self.data['is_abnormal'].sum()} / {len(self.data)} = {self.data['is_abnormal'].mean():.1%}")
        
        # 预处理数据
        self.processed_data = self.advanced_feature_engineering(self.data)
        
        return self.processed_data
    
    def advanced_feature_engineering(self, df):
        """高级特征工程"""
        print("\n=== 高级特征工程 ===")
        
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
        
        print(f"特征工程后特征数量: {df_eng.shape[1]}")
        
        return df_eng
    
    def level1_screening(self, df):
        """一级筛查 - 基于Z值的医学规则"""
        print("\n=== 一级筛查 - 基于Z值的医学规则 ===")
        
        def z_value_diagnosis(row):
            """基于Z值的判定规则"""
            z_values = [
                row['X染色体的Z值绝对值'],
                row['21号染色体的Z值绝对值'],
                row['18号染色体的Z值绝对值'],
                row['13号染色体的Z值绝对值']
            ]
            
            max_z = max(z_values)
            z_over_3 = sum(1 for z in z_values if z > 3.0)
            z_over_2_5 = sum(1 for z in z_values if z > 2.5)
            z_over_2 = sum(1 for z in z_values if z > 2.0)
            z_over_1_5 = sum(1 for z in z_values if z > 1.5)
            
            # 高风险判定
            if z_over_3 >= 1 or z_over_2_5 >= 2:
                return "高风险", f"严重Z值异常 (最大Z值: {max_z:.2f})"
            
            # 中高风险判定
            elif z_over_2 >= 1 and row['年龄'] >= 35:
                return "高风险", f"高龄产妇+Z值异常 (Z值: {max_z:.2f})"
            elif z_over_2 >= 2:
                return "中-高风险", f"多染色体Z值异常 (数量: {z_over_2})"
            elif z_over_2 >= 1 and row['孕妇BMI'] > 30:
                return "中-高风险", f"BMI过高+Z值异常 (Z值: {max_z:.2f})"
            
            # 中风险判定
            elif z_over_1_5 >= 3:
                return "中风险", f"多染色体轻度异常 (数量: {z_over_1_5})"
            elif max_z > 1.5 and row['年龄'] >= 40:
                return "中风险", f"极高龄产妇+轻度异常 (Z值: {max_z:.2f})"
            elif max_z > 1.2:
                return "中风险", f"单个染色体borderline异常 (Z值: {max_z:.2f})"
            
            # 低风险
            else:
                return "低风险", "各项指标在正常范围"
        
        # 应用一级筛查
        screening_results = df.apply(z_value_diagnosis, axis=1)
        df['一级判定'] = [r[0] for r in screening_results]
        df['一级原因'] = [r[1] for r in screening_results]
        
        print("一级筛查结果分布:")
        print(df['一级判定'].value_counts())
        
        return df
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """寻找最优阈值以最大化F1 score"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 找到最大F1 score对应的阈值
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_f1 = f1_scores[optimal_idx]
        
        print(f"最优阈值: {optimal_threshold:.4f}")
        print(f"最优F1 Score: {optimal_f1:.4f}")
        
        return optimal_threshold
    
    def comprehensive_grid_search(self, df):
        """全面的网格搜索优化"""
        print("\n=== 全面网格搜索优化 ===")
        
        # 准备特征
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['序号', 'is_abnormal', '一级判定', '一级原因']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['is_abnormal']
        
        print(f"特征数量: {len(feature_cols)}")
        print(f"样本数量: {len(X)}")
        print(f"正样本比例: {y.mean():.1%}")
        
        # 处理缺失值
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 特征选择
        print("进行特征选择...")
        self.selector = SelectKBest(score_func=f_classif, k=min(25, len(feature_cols)))
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        X_test_selected = self.selector.transform(X_test)
        
        selected_features = [feature_cols[i] for i in self.selector.get_support(indices=True)]
        print(f"选择了 {len(selected_features)} 个特征")
        
        # 标准化
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # 处理类别不平衡
        print("处理类别不平衡...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        print(f"原始训练集: {X_train_scaled.shape}, 异常样本: {y_train.sum()}")
        print(f"平衡后训练集: {X_train_balanced.shape}, 异常样本: {y_train_balanced.sum()}")
        
        # 定义模型网格
        model_grids = self.define_comprehensive_model_grids()
        
        # 训练和评估模型
        model_results = {}
        best_f1 = 0
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model_info in model_grids.items():
            print(f"\n=== 网格搜索 {name} ===")
            start_time = time.time()
            
            try:
                # 创建网格搜索
                grid_search = GridSearchCV(
                    model_info['model'],
                    model_info['params'],
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                # 训练
                grid_search.fit(X_train_scaled, y_train)
                
                # 获取最佳模型
                best_model = grid_search.best_estimator_
                
                # 在测试集上评估
                y_pred = best_model.predict(X_test_scaled)
                y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None
                
                # 计算指标
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                
                # 交叉验证
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='f1')
                
                # 寻找最优阈值
                if y_pred_proba is not None:
                    optimal_threshold = self.find_optimal_threshold(y_test, y_pred_proba)
                    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
                    f1_optimal = f1_score(y_test, y_pred_optimal)
                    precision_optimal = precision_score(y_test, y_pred_optimal)
                    recall_optimal = recall_score(y_test, y_pred_optimal)
                else:
                    optimal_threshold = 0.5
                    f1_optimal = f1
                    precision_optimal = precision
                    recall_optimal = recall
                
                # 记录结果
                model_results[name] = {
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'test_f1': f1,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_auc': auc,
                    'cv_f1_mean': cv_scores.mean(),
                    'cv_f1_std': cv_scores.std(),
                    'optimal_threshold': optimal_threshold,
                    'optimal_f1': f1_optimal,
                    'optimal_precision': precision_optimal,
                    'optimal_recall': recall_optimal,
                    'model': best_model,
                    'training_time': time.time() - start_time
                }
                
                print(f"  最佳参数: {grid_search.best_params_}")
                print(f"  网格搜索最佳F1: {grid_search.best_score_:.4f}")
                print(f"  测试集F1: {f1:.4f}")
                print(f"  测试集Precision: {precision:.4f}")
                print(f"  测试集Recall: {recall:.4f}")
                print(f"  测试集AUC: {auc:.4f}")
                print(f"  交叉验证F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"  最优阈值: {optimal_threshold:.4f}")
                print(f"  最优F1: {f1_optimal:.4f}")
                print(f"  训练时间: {time.time() - start_time:.2f}秒")
                
                # 更新最佳模型
                if f1_optimal > best_f1:
                    best_f1 = f1_optimal
                    self.best_model = best_model
                    self.best_model_name = name
                    self.optimal_threshold = optimal_threshold
                    
            except Exception as e:
                print(f"  训练{name}时出错: {str(e)}")
                continue
        
        # 保存结果
        self.grid_search_results = model_results
        self.best_params = {name: results['best_params'] for name, results in model_results.items()}
        
        print(f"\n最佳模型: {self.best_model_name}, F1 Score: {best_f1:.4f}")
        print(f"最优阈值: {self.optimal_threshold:.4f}")
        
        # 对整个数据集进行预测
        X_all_imputed = self.imputer.transform(X)
        X_all_selected = self.selector.transform(X_all_imputed)
        X_all_scaled = self.scaler.transform(X_all_selected)
        
        if self.best_model is not None and hasattr(self.best_model, 'predict_proba'):
            model_probs = self.best_model.predict_proba(X_all_scaled)[:, 1]
        elif self.best_model is not None:
            model_probs = self.best_model.predict(X_all_scaled).astype(float)
        else:
            model_probs = np.zeros(X_all_scaled.shape[0])
        
        df['模型概率'] = model_probs
        
        # 特征重要性
        if self.best_model is not None and hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        return df, selected_features, model_results
    
    def define_comprehensive_model_grids(self):
        """定义全面的模型网格"""
        model_grids = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [8, 12, 16, None],
                    'min_samples_split': [2, 4, 6],
                    'min_samples_leaf': [1, 2, 3],
                    'max_features': ['sqrt', 'log2', 0.5]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
                'params': {
                    'n_estimators': [200, 300, 400],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'scale_pos_weight': [2, 3, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [200, 300, 400],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9],
                    'max_features': ['sqrt', 'log2', 0.5]
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [8, 12, 16, None],
                    'min_samples_split': [2, 4, 6],
                    'min_samples_leaf': [1, 2, 3],
                    'max_features': ['sqrt', 'log2', 0.5]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.1, 0.5, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42, class_weight='balanced'),
                'params': {
                    'C': [1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
        return model_grids
    
    def comprehensive_diagnosis(self, df):
        """综合判定"""
        print("\n=== 综合判定 ===")
        
        def final_diagnosis(row):
            """综合判定逻辑"""
            level1_result = row['一级判定']
            model_prob = row['模型概率']
            
            # 使用最优阈值进行模型风险评级
            if model_prob > self.optimal_threshold:
                model_risk = "高风险"
            elif model_prob > self.optimal_threshold * 0.6:
                model_risk = "中-高风险"
            elif model_prob > self.optimal_threshold * 0.3:
                model_risk = "中风险"
            else:
                model_risk = "低风险"
            
            # 风险等级映射
            risk_levels = {"低风险": 0, "中风险": 1, "中-高风险": 2, "高风险": 3}
            
            # 综合判定（取更严重的等级）
            final_risk = max(level1_result, model_risk, key=lambda x: risk_levels.get(x, 0))
            
            # 特殊规则：如果任何Z值>3.0，直接判定为高风险
            z_values = [
                row['X染色体的Z值绝对值'],
                row['21号染色体的Z值绝对值'],
                row['18号染色体的Z值绝对值'],
                row['13号染色体的Z值绝对值']
            ]
            if max(z_values) > 3.0:
                final_risk = "高风险"
            
            return final_risk, model_prob
        
        # 应用综合判定
        diagnosis_results = df.apply(final_diagnosis, axis=1)
        df['最终判定'] = [r[0] for r in diagnosis_results]
        df['最终概率'] = [r[1] for r in diagnosis_results]
        
        print("综合判定结果分布:")
        print(df['最终判定'].value_counts())
        
        return df
    
    def evaluate_performance(self, df):
        """评估系统性能"""
        print("\n=== 系统性能评估 ===")
        
        # 将风险等级转换为二分类
        high_risk_prediction = df['最终判定'].isin(['高风险', '中-高风险'])
        actual_abnormal = df['is_abnormal'] == 1
        
        # 计算指标
        tp = sum(high_risk_prediction & actual_abnormal)
        fp = sum(high_risk_prediction & ~actual_abnormal)
        tn = sum(~high_risk_prediction & ~actual_abnormal)
        fn = sum(~high_risk_prediction & actual_abnormal)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"敏感性 (Recall): {sensitivity:.4f}")
        print(f"特异性: {specificity:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"准确率: {accuracy:.4f}")
        print(f"阴性预测值: {npv:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(actual_abnormal, high_risk_prediction)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
        plt.title('增强网格搜索系统混淆矩阵')
        plt.ylabel('实际')
        plt.xlabel('预测')
        plt.savefig('enhanced_grid_search_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'accuracy': accuracy,
            'npv': npv
        }
    
    def create_model_comparison_visualization(self, model_results):
        """创建模型比较可视化"""
        print("\n=== 生成模型比较可视化 ===")
        
        # 准备数据
        models = list(model_results.keys())
        metrics_data = {
            'Model': models,
            'F1_Score': [model_results[m]['optimal_f1'] for m in models],
            'Precision': [model_results[m]['optimal_precision'] for m in models],
            'Recall': [model_results[m]['optimal_recall'] for m in models],
            'AUC': [model_results[m]['test_auc'] for m in models],
            'CV_F1': [model_results[m]['cv_f1_mean'] for m in models],
            'Training_Time': [model_results[m]['training_time'] for m in models]
        }
        
        comparison_df = pd.DataFrame(metrics_data)
        self.model_comparison = comparison_df
        
        # 创建多子图比较
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # F1 Score比较
        bars1 = axes[0, 0].bar(comparison_df['Model'], comparison_df['F1_Score'], 
                              color='#91CDC8', alpha=0.7)
        axes[0, 0].set_title('F1 Score比较', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Precision vs Recall
        scatter = axes[0, 1].scatter(comparison_df['Recall'], comparison_df['Precision'], 
                                   s=150, alpha=0.7, c=comparison_df['F1_Score'], 
                                   cmap='viridis')
        for i, model in enumerate(comparison_df['Model']):
            axes[0, 1].annotate(model, (comparison_df['Recall'][i], comparison_df['Precision'][i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall (颜色表示F1 Score)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # AUC比较
        bars3 = axes[0, 2].bar(comparison_df['Model'], comparison_df['AUC'], 
                              color='#6FB9D0', alpha=0.7)
        axes[0, 2].set_title('AUC比较', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('AUC')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 交叉验证F1
        bars4 = axes[1, 0].bar(comparison_df['Model'], comparison_df['CV_F1'], 
                              color='#5499BD', alpha=0.7)
        axes[1, 0].set_title('交叉验证F1 Score', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('CV F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars4:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 训练时间
        bars5 = axes[1, 1].bar(comparison_df['Model'], comparison_df['Training_Time'], 
                              color='#3981AF', alpha=0.7)
        axes[1, 1].set_title('训练时间 (秒)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('时间 (秒)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars5:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 综合性能雷达图
        metrics = ['F1_Score', 'Precision', 'Recall', 'AUC']
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['optimal_f1'])
        best_values = [
            model_results[best_model_name]['optimal_f1'],
            model_results[best_model_name]['optimal_precision'],
            model_results[best_model_name]['optimal_recall'],
            model_results[best_model_name]['test_auc']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        best_values = best_values + [best_values[0]]
        
        axes[1, 2].plot(angles, best_values, 'o-', linewidth=3, 
                       label=f'最佳模型: {best_model_name}', color='#386195')
        axes[1, 2].fill(angles, best_values, alpha=0.25, color='#386195')
        axes[1, 2].set_xticks(angles[:-1])
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].set_title('最佳模型性能雷达图', fontsize=14, fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def generate_comprehensive_report(self):
        """生成全面报告"""
        print("\n" + "="*80)
        print("增强版网格搜索优化女胎染色体非整倍体判定系统报告")
        print("="*80)
        
        print("\n【一级筛查标准 - 基于Z值】")
        print("1. 高风险指标:")
        print("   - 任何染色体Z值绝对值 > 3.0")
        print("   - 2个或以上染色体Z值绝对值 > 2.5")
        print("   - 高龄产妇(≥35岁) + 任何染色体Z值 > 2.0")
        
        print("\n2. 中-高风险指标:")
        print("   - 2个或以上染色体Z值绝对值 > 2.0")
        print("   - BMI>30 + 任何染色体Z值 > 2.0")
        
        print("\n3. 中风险指标:")
        print("   - 3个或以上染色体Z值绝对值 > 1.5")
        print("   - 极高龄产妇(≥40岁) + 任何染色体Z值 > 1.5")
        print("   - 单个染色体Z值绝对值 1.2-2.0")
        
        print("\n【二级筛查标准 - 网格搜索优化的机器学习模型】")
        print(f"最佳模型: {self.best_model_name}")
        if self.best_model_name in self.best_params:
            print(f"最佳参数: {self.best_params[self.best_model_name]}")
        print(f"最优判定阈值: {self.optimal_threshold:.4f}")
        
        print("\n【模型性能比较】")
        if self.model_comparison is not None:
            print(self.model_comparison.round(4))
        
        print("\n【综合判定原则】")
        print("1. 最终风险等级 = max(一级判定, 模型判定)")
        print("2. 特殊规则：任何染色体Z值>3.0直接判定为高风险")
        print("3. 使用网格搜索找到的最优参数和阈值，最大化F1 score")
        
        print("\n【临床建议】")
        print("- 高风险: 立即进行羊水穿刺或CVS确诊")
        print("- 中-高风险: 强烈建议侵入性产前诊断")
        print("- 中风险: 建议重复NIPT检测或考虑侵入性诊断")
        print("- 低风险: 按常规产检流程")
        
        return True
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("开始增强版网格搜索女胎染色体非整倍体优化分析...")
        
        # 1. 数据加载和预处理
        df = self.load_and_preprocess_data()
        
        # 2. 一级筛查
        df = self.level1_screening(df)
        
        # 3. 全面网格搜索训练优化模型
        df, selected_features, model_results = self.comprehensive_grid_search(df)
        
        # 4. 综合判定
        df = self.comprehensive_diagnosis(df)
        
        # 5. 性能评估
        performance = self.evaluate_performance(df)
        
        # 6. 模型比较可视化
        comparison_df = self.create_model_comparison_visualization(model_results)
        
        # 7. 特征重要性可视化
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(10)

            # 归一化重要性，用来映射颜色深浅
            norm = (top_features['Importance'] - top_features['Importance'].min()) / \
                   (top_features['Importance'].max() - top_features['Importance'].min())

            # 这里用 Blues 渐变色，重要性大 = 深蓝，重要性小 = 浅蓝
            colors = plt.cm.Blues(norm)

            bars = plt.barh(range(len(top_features)),
                            top_features['Importance'],
                            color=colors, alpha=0.9)

            plt.yticks(range(len(top_features)), top_features['Feature'].tolist())
            plt.xlabel('重要性')
            plt.title(f'{self.best_model_name} - 前10个最重要特征')
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 8. 生成综合报告
        self.generate_comprehensive_report()
        
        # 保存结果
        df.to_csv('enhanced_grid_search_results.csv', index=False, encoding='utf-8-sig')
        if self.feature_importance is not None:
            self.feature_importance.to_csv('enhanced_feature_importance.csv', index=False)
        comparison_df.to_csv('enhanced_model_comparison.csv', index=False)
        
        # 保存详细的网格搜索结果
        detailed_results = []
        for model_name, results in model_results.items():
            detailed_results.append({
                'Model': model_name,
                'Best_Params': str(results['best_params']),
                'CV_F1_Score': results['cv_f1_mean'],
                'CV_F1_Std': results['cv_f1_std'],
                'Test_F1': results['test_f1'],
                'Test_Precision': results['test_precision'],
                'Test_Recall': results['test_recall'],
                'Test_AUC': results['test_auc'],
                'Optimal_Threshold': results['optimal_threshold'],
                'Optimal_F1': results['optimal_f1'],
                'Optimal_Precision': results['optimal_precision'],
                'Optimal_Recall': results['optimal_recall'],
                'Training_Time': results['training_time']
            })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv('enhanced_detailed_results.csv', index=False)
        
        print("\n增强版网格搜索优化分析完成！结果已保存到相应文件中。")
        
        return df, performance, model_results

if __name__ == "__main__":
    # 创建和运行增强版网格搜索系统
    system = EnhancedGridSearchSystem()
    results, performance, model_results = system.run_full_analysis()
