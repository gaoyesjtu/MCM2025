import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class SimplifiedSplineReachTimePredictor:
    """简化的基于三次样条的达标时间预测器"""

    def __init__(self, target_concentration=0.04, max_extrapolation_weeks=15, tolerance=0.002):
        self.target_concentration = target_concentration
        self.max_extrapolation_weeks = max_extrapolation_weeks
        self.tolerance = tolerance

    def predict_reach_time(self, woman_data, spline_result):
        """预测达标时间"""

        if not spline_result['success']:
            return None

        spline = spline_result['spline_object']

        # 检查是否已经达标
        if woman_data['reached_target']:
            return self._find_observed_reach_time(woman_data, spline)

        # 预测达标时间
        prediction_result = self._solve_for_target_time(woman_data, spline)

        return prediction_result

    def _find_observed_reach_time(self, woman_data, spline):
        """为已达标的孕妇精确计算达标时间"""
        time_points = woman_data['time_points']
        concentrations = woman_data['y_concentrations']

        reach_index = None
        for i, conc in enumerate(concentrations):
            if conc >= self.target_concentration:
                reach_index = i
                break

        if reach_index == 0:
            return time_points[0]
        else:
            t1, t2 = time_points[reach_index - 1], time_points[reach_index]

            try:
                reach_time = brentq(
                    lambda t: spline(t) - self.target_concentration,
                    t1, t2, xtol=0.01
                )
                return reach_time
            except:
                # 线性插值备选
                c1, c2 = concentrations[reach_index - 1], concentrations[reach_index]
                reach_time = t1 + (self.target_concentration - c1) * (t2 - t1) / (c2 - c1)
                return reach_time

    def _solve_for_target_time(self, woman_data, spline):
        """求解达到目标浓度的时间"""
        last_time = woman_data['last_test_time']
        max_search_time = last_time + self.max_extrapolation_weeks

        # 定义目标函数
        def target_function(t):
            try:
                return spline(t) - self.target_concentration
            except:
                return float('inf')

        # 检查终点是否能达到目标
        try:
            final_concentration = spline(max_search_time)
            if final_concentration < self.target_concentration - self.tolerance:
                # 尝试线性外推估算
                return self._linear_extrapolation_estimate(woman_data, spline)
        except:
            pass

        # 密集搜索
        search_step = 0.5
        search_points = np.arange(last_time, max_search_time + search_step, search_step)

        # 寻找符号变化的区间
        for i in range(len(search_points) - 1):
            t1, t2 = search_points[i], search_points[i + 1]

            try:
                f1 = target_function(t1)
                f2 = target_function(t2)

                if f1 * f2 < 0:  # 符号相反
                    try:
                        solution = brentq(target_function, t1, t2, xtol=0.01)
                        return solution
                    except:
                        continue
            except:
                continue

        # 如果Brent方法失败，尝试最小化
        try:
            result = minimize_scalar(
                lambda t: abs(target_function(t)),
                bounds=(last_time, max_search_time),
                method='bounded'
            )

            if result.success and result.fun < self.tolerance:
                return result.x
        except:
            pass

        # 最后尝试线性外推
        return self._linear_extrapolation_estimate(woman_data, spline)

    def _linear_extrapolation_estimate(self, woman_data, spline):
        """线性外推估算"""
        time_points = woman_data['time_points']
        concentrations = woman_data['y_concentrations']

        if len(time_points) >= 2:
            # 使用最后两个点计算增长率
            time_span = time_points[-1] - time_points[-2]
            conc_change = concentrations[-1] - concentrations[-2]

            if time_span > 0 and conc_change > 0:
                growth_rate = conc_change / time_span
                current_conc = concentrations[-1]
                remaining_conc = self.target_concentration - current_conc

                if remaining_conc > 0:
                    estimated_time = time_points[-1] + remaining_conc / growth_rate

                    # 限制在合理范围内
                    if estimated_time <= time_points[-1] + 25:
                        return estimated_time

        return None


class SimplifiedDataProcessor:
    """简化的数据处理器"""

    def __init__(self, target_concentration=0.04):
        self.target_concentration = target_concentration

    def preprocess_data(self, raw_data):
        """预处理数据，计算BMI平均值"""
        print("开始数据预处理...")

        # 清理列名
        raw_data.columns = raw_data.columns.str.strip()

        # 筛选有效数据
        valid_data = raw_data.dropna(subset=['Y染色体浓度', '检测孕周', '孕妇BMI']).copy()
        valid_data = valid_data[valid_data['Y染色体浓度'] > 0]
        valid_data = valid_data[valid_data['孕妇BMI'] > 0]

        print(f"有效数据记录数: {len(valid_data)}")

        grouped = valid_data.groupby('孕妇代码')
        processed_women = {}
        stats = {'total_women': len(grouped), 'single_test': 0, 'multiple_test': 0}

        for woman_code, group in grouped:
            test_count = len(group)

            if test_count == 1:
                stats['single_test'] += 1
                continue

            sorted_group = group.sort_values('检测孕周')

            # 去除重复时间点
            unique_data = []
            seen_times = set()

            for _, row in sorted_group.iterrows():
                time_point = round(row['检测孕周'], 2)
                if time_point not in seen_times:
                    unique_data.append({
                        'time': time_point,
                        'concentration': row['Y染色体浓度'],
                        'bmi': row['孕妇BMI']
                    })
                    seen_times.add(time_point)

            if len(unique_data) < 2:
                stats['single_test'] += 1
                continue

            stats['multiple_test'] += 1

            # 提取数据
            time_points = np.array([d['time'] for d in unique_data])
            concentrations = np.array([d['concentration'] for d in unique_data])
            bmi_values = np.array([d['bmi'] for d in unique_data])

            # 计算BMI平均值
            avg_bmi = np.mean(bmi_values)

            processed_women[woman_code] = {
                'time_points': time_points,
                'y_concentrations': concentrations,
                'avg_bmi': avg_bmi,
                'test_count': len(unique_data),
                'reached_target': np.any(concentrations >= self.target_concentration),
                'max_concentration': np.max(concentrations),
                'last_test_time': np.max(time_points),
                'first_test_time': np.min(time_points)
            }

        print(f"数据预处理完成:")
        print(f"  总孕妇数: {stats['total_women']}")
        print(f"  有效多次检测: {stats['multiple_test']}")
        print(f"  排除的单次检测: {stats['single_test']}")

        return processed_women, stats


class SimplifiedSplineFitter:
    """简化的三次样条拟合器"""

    def fit_cubic_spline(self, time_points, y_concentrations):
        """拟合三次样条"""
        try:
            cs = CubicSpline(time_points, y_concentrations, bc_type='natural', extrapolate=True)

            fitted_values = cs(time_points)
            residuals = y_concentrations - fitted_values

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_concentrations - np.mean(y_concentrations)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

            return {
                'success': True,
                'spline_object': cs,
                'r_squared': r_squared
            }

        except Exception as e:
            # 备选方案
            try:
                spline = InterpolatedUnivariateSpline(
                    time_points, y_concentrations, k=min(3, len(time_points) - 1), ext=0
                )

                return {
                    'success': True,
                    'spline_object': spline,
                    'r_squared': 0.9  # 估算值
                }
            except:
                return {'success': False, 'error': str(e)}


class SimplifiedCompletePredictor:
    """简化的完整预测系统"""

    def __init__(self, target_concentration=0.04):
        self.data_processor = SimplifiedDataProcessor(target_concentration=target_concentration)
        self.spline_fitter = SimplifiedSplineFitter()
        self.reach_predictor = SimplifiedSplineReachTimePredictor(target_concentration=target_concentration)

        self.processed_data = None
        self.results = []

    def run_complete_analysis(self, raw_data):
        """运行完整分析"""

        print("=" * 60)
        print("简化版Y浓度达标时间预测分析")
        print("=" * 60)

        # 数据预处理
        self.processed_data, stats = self.data_processor.preprocess_data(raw_data)

        if len(self.processed_data) == 0:
            print("错误：没有找到符合条件的多次检测数据！")
            return None

        # 样条拟合和预测
        print(f"\n开始为 {len(self.processed_data)} 个孕妇进行预测...")

        successful_predictions = 0

        for i, (woman_code, woman_data) in enumerate(self.processed_data.items()):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  处理进度: {i + 1}/{len(self.processed_data)}")

            # 拟合三次样条
            spline_result = self.spline_fitter.fit_cubic_spline(
                woman_data['time_points'],
                woman_data['y_concentrations']
            )

            # 预测达标时间
            predicted_time = self.reach_predictor.predict_reach_time(woman_data, spline_result)

            # 只保留成功预测的结果
            if predicted_time is not None:
                self.results.append({
                    '孕妇代码': woman_code,
                    'BMI': round(woman_data['avg_bmi'], 2),
                    '预计到达时间': round(predicted_time, 2)
                })
                successful_predictions += 1

        print(f"\n分析完成!")
        print(
            f"  成功预测: {successful_predictions}/{len(self.processed_data)} ({successful_predictions / len(self.processed_data):.1%})")

        return True

    def get_simplified_results(self):
        """获取简化的结果"""
        return pd.DataFrame(self.results)

    def plot_bmi_vs_reach_time(self, save_path=None):
        """绘制BMI与预计到达时间的关系曲线"""
        if len(self.results) == 0:
            print("没有可用的预测结果！")
            return

        # 设置微软雅黑字体为首选
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        df = pd.DataFrame(self.results)

        # 准备数据
        bmi_values = df['BMI'].values
        reach_times = df['预计到达时间'].values

        # 创建图表
        plt.figure(figsize=(12, 8))

        # 散点图
        plt.scatter(bmi_values, reach_times, alpha=0.6, s=50, color='blue', label='预测数据')

        # 拟合多项式曲线
        '''try:
            # 二次多项式拟合
            poly_features = PolynomialFeatures(degree=2)
            poly_reg = Pipeline([
                ('poly', poly_features),
                ('linear', LinearRegression())
            ])

            bmi_reshaped = bmi_values.reshape(-1, 1)
            poly_reg.fit(bmi_reshaped, reach_times)

            # 生成平滑曲线
            bmi_range = np.linspace(bmi_values.min(), bmi_values.max(), 100)
            reach_time_pred = poly_reg.predict(bmi_range.reshape(-1, 1))

            plt.plot(bmi_range, reach_time_pred, color='red', linewidth=2, label='二次拟合')

            # 计算R²
            r2_score = poly_reg.score(bmi_reshaped, reach_times)

        except Exception as e:
            print(f"多项式拟合失败: {e}")
            r2_score = 0

        # 添加线性趋势线
        try:
            linear_reg = LinearRegression()
            linear_reg.fit(bmi_reshaped, reach_times)
            reach_time_linear = linear_reg.predict(bmi_range.reshape(-1, 1))

            plt.plot(bmi_range, reach_time_linear, color='green', linewidth=2,
                     linestyle='--', alpha=0.7, label='线性趋势')

            linear_r2 = linear_reg.score(bmi_reshaped, reach_times)

        except Exception as e:
            print(f"线性拟合失败: {e}")
            linear_r2 = 0'''

        # 设置图表（使用中文）
        plt.xlabel('BMI', fontsize=12, fontfamily='Microsoft YaHei')
        plt.ylabel('预测到达4%浓度时间（周）', fontsize=12, fontfamily='Microsoft YaHei')
        plt.title('BMI与Y染色体浓度达到4%预测时间关系', fontsize=14, fontweight='bold', fontfamily='Microsoft YaHei')
        plt.legend(fontsize=10, prop={'family': 'Microsoft YaHei'})
        plt.grid(True, alpha=0.3)

        # 添加统计信息（使用中文）
        stats_text = f'样本数量: {len(self.results)}\n'
        stats_text += f'BMI范围: {bmi_values.min():.1f} - {bmi_values.max():.1f}\n'
        stats_text += f'时间范围: {reach_times.min():.1f} - {reach_times.max():.1f} 周\n'
        # stats_text += f'二次拟合 R²: {r2_score:.3f}\n'
        #  += f'线性拟合 R²: {linear_r2:.3f}'

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='Microsoft YaHei',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        # 打印统计分析
        print("\n" + "=" * 60)
        print("BMI与预计到达时间关系分析")
        print("=" * 60)

        # 计算相关系数
        correlation = np.corrcoef(bmi_values, reach_times)[0, 1]
        print(f"相关系数 (Correlation): {correlation:.4f}")

        if abs(correlation) < 0.1:
            correlation_strength = "几乎无相关"
        elif abs(correlation) < 0.3:
            correlation_strength = "弱相关"
        elif abs(correlation) < 0.7:
            correlation_strength = "中等相关"
        else:
            correlation_strength = "强相关"

        correlation_direction = "正相关" if correlation > 0 else "负相关"
        print(f"相关性评价: {correlation_strength}的{correlation_direction}")

        # BMI分组分析
        print(f"\nBMI分组分析:")
        bmi_groups = pd.cut(df['BMI'], bins=[0, 25, 30, 35, 50],
                            labels=['<25', '25-30', '30-35', '≥35'])

        for group_name in ['<25', '25-30', '30-35', '≥35']:
            group_data = df[bmi_groups == group_name]
            if len(group_data) > 0:
                group_times = group_data['预计到达时间']
                print(f"  BMI {group_name}: n={len(group_times)}, "
                      f"均值={group_times.mean():.2f}周, "
                      f"中位数={group_times.median():.2f}周, "
                      f"范围={group_times.min():.1f}-{group_times.max():.1f}周")

        return correlation  #, r2_score, linear_r2

    def plot_two_women_curves(self, woman_codes=['A001', 'A024'], save_path=None):
        """绘制两位孕妇的Y染色体浓度拟合曲线"""
        if self.processed_data is None or len(self.processed_data) == 0:
            print("错误：没有处理过的数据！")
            return

        # 设置微软雅黑字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 如果没有指定孕妇代码，自动选择两位有代表性的孕妇
        if woman_codes is None:
            # 选择一位已达标和一位未达标的孕妇，且测试次数较多的
            reached_women = []
            not_reached_women = []

            for code, data in self.processed_data.items():
                if data['test_count'] >= 3:  # 至少3次测试
                    if data['reached_target']:
                        reached_women.append((code, data))
                    else:
                        not_reached_women.append((code, data))

            # 选择测试次数最多的
            if reached_women:
                reached_women.sort(key=lambda x: x[1]['test_count'], reverse=True)
                woman1_code, woman1_data = reached_women[0]
            elif len(list(self.processed_data.items())) > 0:
                woman1_code, woman1_data = list(self.processed_data.items())[0]
            else:
                print("没有找到合适的孕妇数据！")
                return

            if not_reached_women:
                not_reached_women.sort(key=lambda x: x[1]['test_count'], reverse=True)
                woman2_code, woman2_data = not_reached_women[0]
            elif len(list(self.processed_data.items())) > 1:
                woman2_code, woman2_data = list(self.processed_data.items())[1]
            else:
                # 如果只有一位未达标的，选择第二位已达标的
                if len(reached_women) > 1:
                    woman2_code, woman2_data = reached_women[1]
                else:
                    print("只找到一位合适的孕妇数据！")
                    return
        else:
            # 使用指定的孕妇代码
            if len(woman_codes) != 2:
                print("请提供两个孕妇代码！")
                return

            woman1_code, woman2_code = woman_codes
            if woman1_code not in self.processed_data or woman2_code not in self.processed_data:
                print("指定的孕妇代码不存在！")
                return

            woman1_data = self.processed_data[woman1_code]
            woman2_data = self.processed_data[woman2_code]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 为两位孕妇分别绘制曲线
        women_info = [
            (woman1_code, woman1_data, ax1, "#019092"),
            (woman2_code, woman2_data, ax2, "#5DBFE9")
        ]

        for i, (code, data, ax, color) in enumerate(women_info):
            # 获取数据
            time_points = data['time_points']
            concentrations = data['y_concentrations']
            avg_bmi = data['avg_bmi']
            reached_target = data['reached_target']

            # 拟合三次样条
            spline_result = self.spline_fitter.fit_cubic_spline(time_points, concentrations)

            if spline_result['success']:
                spline = spline_result['spline_object']
                r_squared = spline_result['r_squared']

                # 生成平滑曲线的时间点
                time_min = time_points.min()
                time_max = time_points.max()

                # 如果未达标，延长时间轴用于预测
                if not reached_target:
                    time_max = min(time_max + 10, time_points.max() + 15)

                smooth_time = np.linspace(time_min, time_max, 200)

                try:
                    smooth_concentrations = spline(smooth_time)

                    # 绘制拟合曲线
                    ax.plot(smooth_time, smooth_concentrations, color=color, linewidth=2,
                            label=f'拟合曲线 (R²={r_squared:.3f})')

                    # 预测到达时间
                    predicted_time = self.reach_predictor.predict_reach_time(data, spline_result)

                except Exception as e:
                    print(f"样条插值失败: {e}")
                    smooth_concentrations = None
                    predicted_time = None
            else:
                print(f"孕妇 {code} 的样条拟合失败")
                predicted_time = None

            # 绘制原始数据点
            ax.scatter(time_points, concentrations, color=color, s=60, alpha=0.8,
                       zorder=5, label='观测数据', edgecolors='white', linewidth=1)

            # 添加目标线
            ax.axhline(y=self.reach_predictor.target_concentration, color='green',
                       linestyle='--', alpha=0.7, label='目标浓度 4%')

            # 如果有预测时间，添加预测点
            if predicted_time is not None:
                target_conc = self.reach_predictor.target_concentration
                ax.scatter([predicted_time], [target_conc], color='orange', s=100,
                           marker='*', zorder=6, label=f'预测到达时间: {predicted_time:.1f}周')

                # 添加垂直线显示预测时间
                ax.axvline(x=predicted_time, color='#C5272D', linestyle=':', alpha=0.7)

            # 设置图表属性
            status = "已达标" if reached_target else "未达标"
            ax.set_title(f'孕妇 {code}\nBMI: {avg_bmi:.1f}, 状态: {status}',
                         fontsize=12, fontweight='bold', fontfamily='Microsoft YaHei')
            ax.set_xlabel('孕周', fontsize=11, fontfamily='Microsoft YaHei')
            ax.set_ylabel('Y染色体浓度', fontsize=11, fontfamily='Microsoft YaHei')
            ax.legend(fontsize=9, prop={'family': 'Microsoft YaHei'})
            ax.grid(True, alpha=0.3)

            # 设置y轴范围，确保能看到目标线
            y_min = min(concentrations.min(), self.reach_predictor.target_concentration) * 0.8
            y_max = max(concentrations.max(), self.reach_predictor.target_concentration) * 1.2
            ax.set_ylim(y_min, y_max)

            # 添加数据信息
            info_text = f'测试次数: {data["test_count"]}\n'
            info_text += f'最大浓度: {data["max_concentration"]:.4f}\n'
            info_text += f'时间跨度: {time_points.min():.1f}-{time_points.max():.1f}周'

            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='Microsoft YaHei',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()

        # 打印详细信息
        print("\n" + "=" * 80)
        print("两位孕妇的拟合曲线分析")
        print("=" * 80)

        for i, (code, data) in enumerate([(woman1_code, woman1_data), (woman2_code, woman2_data)], 1):
            print(f"\n孕妇 {i} (代码: {code}):")
            print(f"  BMI: {data['avg_bmi']:.2f}")
            print(f"  测试次数: {data['test_count']}")
            print(f"  时间范围: {data['first_test_time']:.1f} - {data['last_test_time']:.1f} 周")
            print(f"  浓度范围: {data['y_concentrations'].min():.4f} - {data['max_concentration']:.4f}")
            print(f"  是否达标: {'是' if data['reached_target'] else '否'}")

            # 查找预测结果
            prediction_found = False
            for result in self.results:
                if result['孕妇代码'] == code:
                    print(f"  预测到达时间: {result['预计到达时间']:.2f} 周")
                    prediction_found = True
                    break

            if not prediction_found:
                print(f"  预测到达时间: 无法预测")

        return (woman1_code, woman1_data), (woman2_code, woman2_data)

    def show_available_women(self, top_n=10):
        """显示可用的孕妇数据概览"""
        if self.processed_data is None:
            print("错误：没有处理过的数据！")
            return

        print("\n" + "=" * 80)
        print("可用孕妇数据概览")
        print("=" * 80)

        women_list = []
        for code, data in self.processed_data.items():
            women_list.append({
                '孕妇代码': code,
                'BMI': round(data['avg_bmi'], 1),
                '测试次数': data['test_count'],
                '是否达标': '是' if data['reached_target'] else '否',
                '最大浓度': round(data['max_concentration'], 4),
                '时间跨度': f"{data['first_test_time']:.1f}-{data['last_test_time']:.1f}"
            })

        # 按测试次数排序
        women_list.sort(key=lambda x: x['测试次数'], reverse=True)

        print(f"显示前 {min(top_n, len(women_list))} 位孕妇（按测试次数排序）:")
        print()

        for i, woman in enumerate(women_list[:top_n]):
            print(f"{i + 1:2d}. 代码: {woman['孕妇代码']:>8} | BMI: {woman['BMI']:>5} | "
                  f"测试: {woman['测试次数']:>2}次 | 达标: {woman['是否达标']:>1} | "
                  f"最大浓度: {woman['最大浓度']:>7} | 跨度: {woman['时间跨度']:>12}周")

        print(f"\n总计: {len(women_list)} 位孕妇")
        print("\n使用方法:")
        print("predictor.plot_two_women_curves()  # 自动选择两位代表性孕妇")
        print("predictor.plot_two_women_curves(['代码1', '代码2'])  # 指定两位孕妇")


def analyze_results_summary(results_df):
    """分析结果总结"""
    print("\n" + "=" * 60)
    print("预测结果总结")
    print("=" * 60)

    print(f"成功预测的孕妇数量: {len(results_df)}")

    # BMI统计
    bmi_values = results_df['BMI']
    print(f"\nBMI统计:")
    print(f"  均值: {bmi_values.mean():.2f}")
    print(f"  中位数: {bmi_values.median():.2f}")
    print(f"  范围: {bmi_values.min():.1f} - {bmi_values.max():.1f}")
    print(f"  标准差: {bmi_values.std():.2f}")

    # 预计到达时间统计
    reach_times = results_df['预计到达时间']
    print(f"\n预计到达时间统计:")
    print(f"  均值: {reach_times.mean():.2f} 周")
    print(f"  中位数: {reach_times.median():.2f} 周")
    print(f"  范围: {reach_times.min():.1f} - {reach_times.max():.1f} 周")
    print(f"  标准差: {reach_times.std():.2f} 周")

    # 时间分布
    time_ranges = {
        '≤12周': len(reach_times[reach_times <= 12]),
        '13-15周': len(reach_times[(reach_times > 12) & (reach_times <= 15)]),
        '16-18周': len(reach_times[(reach_times > 15) & (reach_times <= 18)]),
        '19-21周': len(reach_times[(reach_times > 18) & (reach_times <= 21)]),
        '>21周': len(reach_times[reach_times > 21])
    }

    print(f"\n预计到达时间分布:")
    for time_range, count in time_ranges.items():
        percentage = count / len(reach_times) * 100
        print(f"  {time_range}: {count} ({percentage:.1f}%)")


def main():
    try:
        # 读取数据
        print("读取数据文件: ../boys_data.csv")
        raw_data = pd.read_csv(r'D:\pycharm_codes\MCM2025_codes\Q2\boys_data.csv')
        print(f"原始数据shape: {raw_data.shape}")

        # 创建简化预测器
        predictor = SimplifiedCompletePredictor(target_concentration=0.04)

        # 运行分析
        success = predictor.run_complete_analysis(raw_data)

        if not success:
            return

        # 获取简化结果
        results_df = predictor.get_simplified_results()

        if len(results_df) == 0:
            print("没有成功的预测结果！")
            return

        # 保存简化结果
        output_file = 'reach_time_results.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n结果已保存到: {output_file}")

        # 显示前几行结果
        print(f"\n预测结果示例 (前10行):")
        print(results_df.head(10).to_string(index=False))

        # 分析结果
        analyze_results_summary(results_df)

        # 绘制BMI与预计到达时间的关系曲线
        print(f"\n绘制BMI与预计到达时间关系图...")
        correlation = predictor.plot_bmi_vs_reach_time('bmi_vs_reach_time.png')

        predictor.plot_two_women_curves(save_path='two_women_curves_auto.pdf')

        print("\n" + "=" * 60)
        print("简化分析完成！")
        print(f"结果文件: {output_file}")
        print(f"关系图: bmi_vs_reach_time.png")
        print("=" * 60)

        return predictor, results_df

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    predictor, results_df = main()