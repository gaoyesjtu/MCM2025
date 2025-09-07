#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIPT Y染色体浓度预测与最佳检测时点分析 - 随机森林分年龄段版本
使用随机森林算法按年龄段进行预测（替换XGBoost）
解决问题3：综合考虑多种因素，为不同BMI群体提供最佳NIPT时点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# 设置微软雅黑字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class NIPTRandomForestPredictor:
    def __init__(self):
        self.models = {}  # 存储不同年龄段的模型
        self.scalers = {}  # 存储不同年龄段的标准化器
        self.feature_importance = {}  # 存储不同年龄段的特征重要性
        self.age_groups = {}  # 存储年龄分组信息
        self.results = []
        self.data = None

    def load_and_preprocess_data(self, csv_file='boys_data.csv'):
        """加载和预处理数据"""
        print("正在加载数据...")

        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file, encoding='gb2312')
            except:
                df = pd.read_csv(csv_file, encoding='gbk')

        print(f"原始数据形状: {df.shape}")
        print(f"前5列名: {df.columns.tolist()[:5]}")

        # 数据清洗：只保留男胎数据（有Y染色体浓度的数据）
        # 过滤无效数据
        valid_mask = (
                df['Y染色体浓度'].notna() &
                (df['Y染色体浓度'] > 0) &
                df['年龄'].notna() & (df['年龄'] > 0) &
                df['身高'].notna() & (df['身高'] > 0) &
                df['体重'].notna() & (df['体重'] > 0) &
                df['孕妇BMI'].notna() & (df['孕妇BMI'] > 0) &
                df['检测孕周'].notna() & (df['检测孕周'] > 0)
        )

        self.data = df[valid_mask].copy()
        print(f"有效男胎数据点: {len(self.data)}")

        # 转换数值列
        numeric_columns = ['Y染色体浓度', '检测孕周', '孕妇BMI', '身高', '体重', '年龄']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # 再次过滤缺失值
        self.data = self.data.dropna(subset=numeric_columns)
        print(f"数值转换后有效数据点: {len(self.data)}")

        # 显示数据基本统计
        print("\n=== 数据基本统计 ===")
        features = ['年龄', '身高', '体重', '孕妇BMI', '检测孕周', 'Y染色体浓度']
        print(self.data[features].describe())

        # Y染色体浓度达标分析
        above_4_percent = (self.data['Y染色体浓度'] >= 0.04).sum()
        total_samples = len(self.data)
        print(
            f"\nY染色体浓度≥4%的样本: {above_4_percent}/{total_samples} ({above_4_percent / total_samples * 100:.1f}%)")

        return self.data

    def create_age_groups(self):
        """创建年龄分组"""
        print("\n=== 创建年龄分组 ===")

        # 按年龄分组（4组）
        self.data['Age_4group'] = pd.qcut(self.data['年龄'], q=4, labels=['Very Young', 'Young', 'Middle', 'Old'])

        print("年龄分组分布:")
        age_distribution = self.data['Age_4group'].value_counts()
        print(age_distribution)

        # 存储年龄分组的边界信息
        age_quartiles = self.data['年龄'].quantile([0, 0.25, 0.5, 0.75, 1.0])
        self.age_groups = {
            'Very Young': (age_quartiles[0], age_quartiles[0.25]),
            'Young': (age_quartiles[0.25], age_quartiles[0.5]),
            'Middle': (age_quartiles[0.5], age_quartiles[0.75]),
            'Old': (age_quartiles[0.75], age_quartiles[1.0])
        }

        print("\n年龄分组范围:")
        for group, (min_age, max_age) in self.age_groups.items():
            print(f"{group}: {min_age:.1f} - {max_age:.1f} 岁")

        return self.data

    def prepare_features(self):
        """准备特征和目标变量"""
        # 基础特征（与Q4_demo保持一致）
        self.feature_names = ['检测孕周', '孕妇BMI', '身高', '体重', '年龄']

        print(f"\n特征名称: {self.feature_names}")
        print(f"目标变量: Y染色体浓度")

        return self.feature_names

    def train_models_by_age_group(self, test_size=0.2, random_state=42):
        """按年龄段训练随机森林模型"""
        print("\n=== 按年龄段训练随机森林模型 ===")

        # 准备特征
        feature_names = self.prepare_features()

        # 为每个年龄组训练模型
        for group in ['Very Young', 'Young', 'Middle', 'Old']:
            print(f"\n训练 {group} 年龄组模型...")

            group_data = self.data[self.data['Age_4group'] == group]
            print(f"{group} 组样本数: {len(group_data)}")

            if len(group_data) < 20:
                print(f"{group} 组数据不足，跳过训练")
                continue

            # 准备特征和目标变量
            X_group = group_data[feature_names]
            y_group = group_data['Y染色体浓度']

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_group, y_group, test_size=test_size, random_state=random_state
            )

            # 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 训练随机森林模型
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )

            rf_model.fit(X_train_scaled, y_train)

            # 预测和评估
            y_train_pred = rf_model.predict(X_train_scaled)
            y_test_pred = rf_model.predict(X_test_scaled)

            # 计算评估指标
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            print(f"{group} - 训练集 RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}, MAE: {train_mae:.6f}")
            print(f"{group} - 测试集 RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}, MAE: {test_mae:.6f}")

            # 交叉验证
            try:
                cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
                print(f"{group} - 5折交叉验证 R² 分数: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except:
                print(f"{group} - 交叉验证跳过")

            # 存储模型和标准化器
            self.models[group] = rf_model
            self.scalers[group] = scaler

            # 特征重要性
            try:
                feature_name_mapping = {
                    '检测孕周': '检测孕周',
                    '孕妇BMI': '孕妇BMI',
                    '身高': '身高',
                    '体重': '体重',
                    '年龄': '年龄'
                }

                self.feature_importance[group] = pd.DataFrame({
                    'feature': [feature_name_mapping.get(name, name) for name in feature_names],
                    'feature_en': feature_names,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)

                print(f"{group} 特征重要性排序:")
                for idx, row in self.feature_importance[group].iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
            except:
                print(f"{group} 特征重要性计算跳过")

        print(f"\n成功训练了 {len(self.models)} 个年龄组模型")

    def get_age_group_for_age(self, age):
        """根据年龄确定所属的年龄组"""
        for group, (min_age, max_age) in self.age_groups.items():
            if group == 'Old':  # 最后一组包含上边界
                if min_age <= age <= max_age:
                    return group
            else:
                if min_age <= age < max_age:
                    return group
        # 如果没有匹配到，返回最接近的组
        if age < list(self.age_groups.values())[0][0]:
            return 'Very Young'
        else:
            return 'Old'

    def predict_y_concentration(self, age, height, weight, bmi, gestational_week):
        """根据年龄组预测Y染色体浓度"""
        # 确定年龄组
        age_group = self.get_age_group_for_age(age)

        if age_group not in self.models:
            # 如果该年龄组没有模型，使用最相似的年龄组
            available_groups = list(self.models.keys())
            if not available_groups:
                raise ValueError("没有可用的训练模型")
            age_group = available_groups[0]  # 使用第一个可用模型

        model = self.models[age_group]
        scaler = self.scalers[age_group]

        # 构造特征向量
        features = np.array([[
            gestational_week, bmi, height, weight, age
        ]])

        # 标准化特征
        features_scaled = scaler.transform(features)

        # 预测
        prediction = model.predict(features_scaled)[0]
        return max(0, prediction)  # 确保预测值非负

    def find_optimal_timing(self, week_range=(10, 25), step=5.0):
        """为每个孕妇找到最佳检测时点（按年龄段预测）"""
        print("\n=== 寻找最佳检测时点（分年龄段） ===")

        # 获取每个孕妇的基本信息
        women_info = self.data.groupby('孕妇代码').agg({
            '年龄': 'first',
            '身高': 'first',
            '体重': 'first',
            '孕妇BMI': 'mean'  # BMI使用平均值
        }).reset_index()

        optimal_results = []
        target_concentration = 0.04  # 4%

        print(f"分析 {len(women_info)} 个孕妇的最佳检测时点...")

        for idx, woman in women_info.iterrows():
            if idx % 50 == 0:
                print(f"进度: {idx}/{len(women_info)}")

            best_week = None
            min_diff = float('inf')
            best_concentration = None
            age_group_used = None

            # 确定年龄组
            age_group = self.get_age_group_for_age(woman['年龄'])
            if age_group not in self.models:
                # 如果该年龄组没有模型，跳过或使用默认值
                available_groups = list(self.models.keys())
                if available_groups:
                    age_group = available_groups[0]
                else:
                    # 使用默认值
                    best_week = 15.0
                    best_concentration = target_concentration
                    min_diff = 0
                    age_group_used = "默认"

            if best_week is None:
                # 在指定孕周范围内搜索
                weeks = np.arange(week_range[0], week_range[1] + step, step)

                for week in weeks:
                    try:
                        predicted_conc = self.predict_y_concentration(
                            woman['年龄'], woman['身高'], woman['体重'],
                            woman['孕妇BMI'], week
                        )

                        diff = abs(predicted_conc - target_concentration)
                        if diff < min_diff:
                            min_diff = diff
                            best_week = week
                            best_concentration = predicted_conc
                            age_group_used = age_group
                    except:
                        continue

                # 如果没找到合适的，使用默认值
                if best_week is None:
                    best_week = 15.0
                    best_concentration = target_concentration
                    min_diff = 0
                    age_group_used = "默认"

            # 风险等级分类
            if best_week <= 12:
                risk_level = '低风险'
            elif best_week <= 27:
                risk_level = '中风险'
            else:
                risk_level = '高风险'

            optimal_results.append({
                '孕妇代码': woman['孕妇代码'],
                '年龄': woman['年龄'],
                '身高': woman['身高'],
                '体重': woman['体重'],
                'BMI': woman['孕妇BMI'],
                '年龄组': age_group_used,
                '预计到达时间': best_week,
                '预测Y染色体浓度': best_concentration,
                '与目标差值': min_diff,
                '风险等级': risk_level
            })

        self.results = pd.DataFrame(optimal_results)
        print(f"完成预测，共处理 {len(self.results)} 个孕妇")

        # 显示各年龄组的使用情况
        print("\n各年龄组模型使用情况:")
        age_group_usage = self.results['年龄组'].value_counts()
        print(age_group_usage)

        return self.results

    def analyze_bmi_groups(self):
        """BMI分组分析"""
        if len(self.results) == 0:
            print("请先运行最佳时点预测")
            return None

        print("\n=== BMI分组分析 ===")

        # 定义BMI分组
        bmi_groups = [
            ('[20,28)', 20, 28),
            ('[28,32)', 28, 32),
            ('[32,36)', 32, 36),
            ('[36,40)', 36, 40),
            ('40+', 40, float('inf'))
        ]

        group_analysis = []

        for group_name, min_bmi, max_bmi in bmi_groups:
            if max_bmi == float('inf'):
                group_data = self.results[self.results['BMI'] >= min_bmi]
            else:
                group_data = self.results[(self.results['BMI'] >= min_bmi) & (self.results['BMI'] < max_bmi)]

            if len(group_data) == 0:
                continue

            avg_week = group_data['预计到达时间'].mean()
            avg_bmi = group_data['BMI'].mean()
            low_risk_count = (group_data['风险等级'] == '低风险').sum()
            mid_risk_count = (group_data['风险等级'] == '中风险').sum()
            high_risk_count = (group_data['风险等级'] == '高风险').sum()
            avg_concentration = group_data['预测Y染色体浓度'].mean()

            group_analysis.append({
                'BMI组': group_name,
                '人数': len(group_data),
                '平均BMI': avg_bmi,
                '预计到达时间': avg_week,
                '低风险人数': low_risk_count,
                '中风险人数': mid_risk_count,
                '高风险人数': high_risk_count,
                '低风险比例(%)': low_risk_count / len(group_data) * 100,
                '平均预测浓度': avg_concentration
            })

        group_df = pd.DataFrame(group_analysis)

        print("\nBMI分组统计结果:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(group_df.round(2))

        return group_df

    def plot_results(self):
        """绘制单独的结果图表"""
        if len(self.results) == 0:
            print("请先运行最佳时点预测")
            return

        try:
            # 确保字体设置
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False

            # 图1: BMI vs 最佳检测孕周散点图（单独一页）
            fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))

            scatter = ax1.scatter(self.results['BMI'], self.results['预计到达时间'],
                                  c=self.results['预测Y染色体浓度'], cmap='viridis',
                                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

            ax1.set_xlabel('孕妇BMI (kg/m²)', fontsize=14, fontfamily='Microsoft YaHei')
            ax1.set_ylabel('最佳检测孕周 (周)', fontsize=14, fontfamily='Microsoft YaHei')
            ax1.set_title('孕妇BMI与最佳检测孕周关系（随机森林分年龄段）', fontsize=16, fontweight='bold',
                          fontfamily='Microsoft YaHei')
            ax1.grid(True, alpha=0.3)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('预测Y染色体浓度', fontsize=12, fontfamily='Microsoft YaHei')

            # 设置坐标轴刻度标签字体
            ax1.tick_params(axis='both', which='major', labelsize=12)
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontfamily('Microsoft YaHei')

            plt.tight_layout()
            plt.savefig('BMI_vs_optimal_week_RF.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 图2: 年龄组模型性能对比图
            if self.feature_importance:
                fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

                # 计算各年龄组的平均特征重要性
                all_features = set()
                for group_importance in self.feature_importance.values():
                    all_features.update(group_importance['feature'].tolist())

                avg_importance = {}
                for feature in all_features:
                    importances = []
                    for group, group_importance in self.feature_importance.items():
                        feature_row = group_importance[group_importance['feature'] == feature]
                        if not feature_row.empty:
                            importances.append(feature_row['importance'].iloc[0])
                    if importances:
                        avg_importance[feature] = np.mean(importances)

                # 绘制平均特征重要性 - 使用指定颜色 #6FDCB5
                features = list(avg_importance.keys())
                importances = list(avg_importance.values())

                bars = ax2.barh(features, importances,
                                color='#6FDCB5', alpha=0.8, edgecolor='black', linewidth=0.5)

                ax2.set_xlabel('平均特征重要性', fontsize=14, fontfamily='Microsoft YaHei')
                ax2.set_ylabel('特征名称', fontsize=14, fontfamily='Microsoft YaHei')
                ax2.set_title('随机森林模型平均特征重要性分析', fontsize=16, fontweight='bold',
                              fontfamily='Microsoft YaHei')
                ax2.grid(True, alpha=0.3, axis='x')

                # 在柱子上添加数值标签
                for bar, importance in zip(bars, importances):
                    width = bar.get_width()
                    ax2.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                             f'{importance:.3f}', ha='left', va='center',
                             fontsize=10, fontfamily='Microsoft YaHei')

                # 设置坐标轴刻度标签字体
                ax2.tick_params(axis='both', which='major', labelsize=12)
                for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                    label.set_fontfamily('Microsoft YaHei')

                plt.tight_layout()
                plt.savefig('feature_importance_RF.png', dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print("特征重要性数据不可用，跳过特征重要性图")

        except Exception as e:
            print(f"绘图出错: {e}")
            print("跳过图表绘制")

    def save_results(self, filename='reach_time_results_RF.csv'):
        """保存结果到文件"""
        if len(self.results) == 0:
            print("没有结果可保存")
            return

        try:
            # 保存为CSV格式（为了与原程序兼容）
            self.results.to_csv('reach_time_results.csv', index=False, encoding='utf-8-sig')
            print("结果已保存为: reach_time_results.csv")

            # 同时保存Excel格式
            with pd.ExcelWriter(filename.replace('.csv', '.xlsx'), engine='openpyxl') as writer:
                # 详细结果
                self.results.to_excel(writer, sheet_name='详细预测结果', index=False)

                # BMI分组分析
                group_analysis = self.analyze_bmi_groups()
                if group_analysis is not None:
                    group_analysis.to_excel(writer, sheet_name='BMI分组分析', index=False)

                # 各年龄组特征重要性
                for group, importance_df in self.feature_importance.items():
                    sheet_name = f'{group}_特征重要性'
                    importance_df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"Excel结果已保存到 {filename.replace('.csv', '.xlsx')}")

        except Exception as e:
            print(f"保存文件出错: {e}")
            # 备用保存方案
            self.results.to_csv('reach_time_results.csv', index=False, encoding='utf-8-sig')
            print("已保存为CSV格式: reach_time_results.csv")


def main():
    """主函数"""
    print("NIPT Y染色体浓度预测与最佳检测时点分析 - 随机森林分年龄段版本")
    print("=" * 70)

    # 创建预测器实例
    predictor = NIPTRandomForestPredictor()

    try:
        # 1. 加载和预处理数据
        data = predictor.load_and_preprocess_data('../boys_data.csv')

        # 2. 创建年龄分组
        data_with_groups = predictor.create_age_groups()

        # 3. 按年龄段训练模型
        predictor.train_models_by_age_group()

        # 4. 预测最佳检测时点
        optimal_results = predictor.find_optimal_timing()

        # 5. BMI分组分析
        # group_analysis = predictor.analyze_bmi_groups()

        # 6. 显示部分结果
        print("\n=== 预测结果示例（前10个） ===")
        pd.set_option('display.max_columns', None)
        print(optimal_results.head(10).round(3))

        # 7. 绘制结果图表
        predictor.plot_results()

        # 8. 保存核心结果
        predictor.save_results()

        # 9. 显示保存的核心结果统计
        if len(optimal_results) > 0:
            core_results = optimal_results[['孕妇代码', 'BMI', '预计到达时间', '年龄组']].copy()

            print(f"\n=== 核心结果统计 ===")
            print(f"总孕妇数: {len(core_results)}")
            print(f"BMI范围: {core_results['BMI'].min():.1f} - {core_results['BMI'].max():.1f}")
            print(
                f"最佳检测孕周范围: {core_results['预计到达时间'].min():.1f} - {core_results['预计到达时间'].max():.1f} 周")
            print(f"平均最佳检测孕周: {core_results['预计到达时间'].mean():.1f} 周")

            print(f"\n各年龄组预测结果统计:")
            age_group_stats = core_results.groupby('年龄组').agg({
                'BMI': ['count', 'mean'],
                '预计到达时间': ['mean', 'std']
            }).round(2)
            print(age_group_stats)

        print("\n分析完成！核心结果已保存，图表已生成。")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()