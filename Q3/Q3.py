import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedMedicalRiskFunction:
    """增强的医学风险函数：过晚检测用更陡峭的分段指数函数，过早检测用二次函数"""

    def __init__(self,
                 early_quadratic_coeff=0.015,
                 late_exp_base_1=0.3, late_exp_rate_1=0.15,
                 late_exp_base_2=2.5, late_exp_rate_2=0.40,
                 late_exp_base_3=6.0, late_exp_rate_3=0.65,
                 base_risk=1.0):
        """
        风险函数参数（增强过晚检测斜率）：

        过早检测（test_time < reach_time）：
        - early_quadratic_coeff: 二次函数系数

        过晚检测（test_time > reach_time）分段指数：
        - 第1段 (0 ≤ delay ≤ 12): base_1 * exp(rate_1 * delay)
        - 第2段 (12 < delay ≤ 27): base_2 * exp(rate_2 * (delay-12))
        - 第3段 (delay > 27): base_3 * exp(rate_3 * (delay-27))

        - base_risk: 基础风险
        """
        self.early_quadratic_coeff = early_quadratic_coeff
        self.late_exp_base_1 = late_exp_base_1
        self.late_exp_rate_1 = late_exp_rate_1
        self.late_exp_base_2 = late_exp_base_2
        self.late_exp_rate_2 = late_exp_rate_2
        self.late_exp_base_3 = late_exp_base_3
        self.late_exp_rate_3 = late_exp_rate_3
        self.base_risk = base_risk

    def calculate_individual_risk(self, test_time, reach_time):
        """计算单个孕妇的风险"""

        if test_time <= reach_time:
            # 过早检测：二次函数风险
            time_diff = reach_time - test_time
            early_risk = self.early_quadratic_coeff * (time_diff ** 2) + 0.2
            return self.base_risk + early_risk

        else:
            # 过晚检测：分段指数函数风险
            delay = test_time - reach_time

            # if delay <= 12:
                # 第1段：相对温和的指数增长
            late_risk = self.late_exp_base_1 * np.exp(self.late_exp_rate_1 * delay)
            '''elif delay <= 27:
                # 第2段：中等的指数增长（斜率显著增大）
                # 确保连续性
                boundary_1 = self.late_exp_base_1 * np.exp(self.late_exp_rate_1 * 12)
                late_risk = boundary_1 + self.late_exp_base_2 * np.exp(self.late_exp_rate_2 * (delay - 12))
            else:
                # 第3段：急剧的指数增长（斜率极大）
                # 确保连续性
                boundary_1 = self.late_exp_base_1 * np.exp(self.late_exp_rate_1 * 12)
                boundary_2 = boundary_1 + self.late_exp_base_2 * np.exp(self.late_exp_rate_2 * 15)  # 27-12=15
                late_risk = boundary_2 + self.late_exp_base_3 * np.exp(self.late_exp_rate_3 * (delay - 27))'''

            return self.base_risk + late_risk

    def calculate_group_risk(self, test_time, reach_times, return_details=False):
        """计算一组孕妇的平均风险"""
        individual_risks = []
        early_count = 0
        late_count = 0
        on_time_count = 0

        for reach_time in reach_times:
            risk = self.calculate_individual_risk(test_time, reach_time)
            individual_risks.append(risk)

            if test_time < reach_time:
                early_count += 1
            elif test_time > reach_time:
                late_count += 1
            else:
                on_time_count += 1

        avg_risk = np.mean(individual_risks)
        detection_success_rate = np.mean([reach_time <= test_time for reach_time in reach_times])

        if return_details:
            return {
                'avg_risk': avg_risk,
                'detection_success_rate': detection_success_rate,
                'early_count': early_count,
                'late_count': late_count,
                'on_time_count': on_time_count,
                'individual_risks': individual_risks
            }
        else:
            return avg_risk, detection_success_rate

    def plot_risk_function(self, reach_time=11, test_time_range=(3, 35), save_path=None):
        """绘制风险函数形状（修正版本）"""
        test_times = np.linspace(test_time_range[0], test_time_range[1], 500)
        risks = [self.calculate_individual_risk(t, reach_time) for t in test_times]

        plt.figure(figsize=(16, 12))

        # 创建子图
        gs = plt.GridSpec(3, 2, hspace=0.4, wspace=0.3)

        # 主图：完整的风险函数
        ax1 = plt.subplot(gs[0, :])
        plt.plot(test_times, risks, 'b-', linewidth=3, label='Enhanced Risk Function')

        # 标记达标时间点
        plt.axvline(x=reach_time, color='red', linestyle='--', linewidth=2,
                    label=f'Reach Time = {reach_time} weeks')

        # 分别绘制早期和晚期区域
        early_times = test_times[test_times < reach_time]
        late_times = test_times[test_times > reach_time]

        if len(early_times) > 0:
            early_risks = [self.calculate_individual_risk(t, reach_time) for t in early_times]
            plt.fill_between(early_times, early_risks, alpha=0.3, color='green',
                             label='Early Detection (Quadratic)')

        if len(late_times) > 0:
            late_risks = [self.calculate_individual_risk(t, reach_time) for t in late_times]
            plt.fill_between(late_times, late_risks, alpha=0.3, color='orange',
                             label='Late Detection (Enhanced Exponential)')

        # 标记分段边界（基于延迟时间）
        '''plt.axvline(x=reach_time + 12, color='orange', linestyle=':', alpha=0.7,
                    label='Segment 1→2 (delay=12w)')
        plt.axvline(x=reach_time + 27, color='red', linestyle=':', alpha=0.7,
                    label='Segment 2→3 (delay=27w)')'''

        plt.xlabel('Test Time (weeks)')
        plt.ylabel('Risk')
        plt.title(f'Enhanced Medical Risk Function (Reach Time = {reach_time} weeks)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # 子图1：早期检测部分详细显示
        ax2 = plt.subplot(gs[1, 0])
        early_range = np.linspace(max(reach_time - 8, test_time_range[0]), reach_time, 100)
        early_risks = [self.calculate_individual_risk(t, reach_time) for t in early_range]
        plt.plot(early_range, early_risks, 'g-', linewidth=3)
        plt.axvline(x=reach_time, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Test Time (weeks)')
        plt.ylabel('Risk')
        plt.title('Early Detection: Quadratic Risk')
        plt.grid(True, alpha=0.3)

        # 子图2：晚期检测部分详细显示（线性坐标）
        ax3 = plt.subplot(gs[1, 1])
        late_range = np.linspace(reach_time, min(reach_time + 30, test_time_range[1]), 200)
        late_risks = [self.calculate_individual_risk(t, reach_time) for t in late_range]
        plt.plot(late_range, late_risks, 'orange', linewidth=3)
        plt.axvline(x=reach_time, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Test Time (weeks)')
        plt.ylabel('Risk')
        plt.title('Late Detection: Enhanced Exponential (Linear Scale)')
        plt.grid(True, alpha=0.3)

        ''''# 子图3：风险斜率分析
        ax4 = plt.subplot(gs[2, 0])
        # 计算延迟0-40周的风险值和斜率
        delays = np.linspace(0, 40, 200)
        delay_risks = [self.calculate_individual_risk(reach_time + d, reach_time) for d in delays]

        plt.plot(delays, delay_risks, 'r-', linewidth=3, label='Risk vs Delay')
        plt.axvline(x=12, color='orange', linestyle=':', alpha=0.7, label='Boundary 1 (12w)')
        plt.axvline(x=27, color='red', linestyle=':', alpha=0.7, label='Boundary 2 (27w)')
        plt.xlabel('Delay (weeks)')
        plt.ylabel('Risk')
        plt.title('Risk vs Delay Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')

        # 子图4：风险增长率分析
        ax5 = plt.subplot(gs[2, 1])
        # 显示关键时间点的风险值
        key_delays = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
        key_risks = [self.calculate_individual_risk(reach_time + d, reach_time) for d in key_delays]

        plt.bar(key_delays, key_risks, alpha=0.7, color=['lightblue'] * 4 + ['orange'] * 4 + ['red'] * 5)
        plt.axvline(x=12, color='orange', linestyle=':', alpha=0.7)
        plt.axvline(x=27, color='red', linestyle=':', alpha=0.7)
        plt.xlabel('Delay (weeks)')
        plt.ylabel('Risk')
        plt.title('Risk at Key Delay Points')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')'''

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def find_optimal_test_time(self, reach_times, test_time_range=(10, 30)):
        """为给定的达标时间分布找到最优检测时间"""
        test_times = np.linspace(test_time_range[0], test_time_range[1], 200)
        risks = [self.calculate_group_risk(t, reach_times)[0] for t in test_times]

        optimal_idx = np.argmin(risks)
        optimal_time = test_times[optimal_idx]
        optimal_risk = risks[optimal_idx]

        return optimal_time, optimal_risk


class RobustBMIGroupingOptimizer:
    """鲁棒的BMI分组优化器 - 解决大组数优化问题"""

    def __init__(self, data, risk_function, min_group_size=6, max_groups=6,
                 optimization_attempts=5):
        """
        参数：
        - data: 包含BMI和预计到达时间的DataFrame
        - risk_function: 增强的医学风险函数对象
        - min_group_size: 每组最小样本量（降低要求）
        - max_groups: 最大分组数
        - optimization_attempts: 每个分组的优化尝试次数
        """
        self.data = data.copy()
        self.risk_function = risk_function
        self.min_group_size = min_group_size
        self.max_groups = max_groups
        self.optimization_attempts = optimization_attempts

        # 数据预处理
        self.bmi_values = self.data['BMI'].values
        self.reach_times = self.data['预计到达时间'].values
        self.bmi_min = self.bmi_values.min()
        self.bmi_max = self.bmi_values.max()

        # 可行性检查
        max_feasible_groups = len(self.data) // self.min_group_size
        if self.max_groups > max_feasible_groups:
            print(f"⚠️ 警告: 样本量{len(self.data)}最多支持{max_feasible_groups}组(每组≥{self.min_group_size}样本)")
            self.max_groups = min(self.max_groups, max_feasible_groups)

        self.best_solution = None
        self.optimization_history = []

        print(f"鲁棒BMI分组优化器初始化完成:")
        print(f"  数据集大小: {len(self.data)}")
        print(f"  BMI范围: {self.bmi_min:.1f} - {self.bmi_max:.1f}")
        print(f"  达标时间范围: {self.reach_times.min():.1f} - {self.reach_times.max():.1f} 周")
        print(f"  最大可行分组数: {self.max_groups}")
        print(f"  每组最小样本量: {self.min_group_size}")

    def assign_groups(self, boundaries):
        """根据边界点分配组别"""
        group_assignments = np.zeros(len(self.bmi_values), dtype=int)

        for i, bmi in enumerate(self.bmi_values):
            group = 0
            for boundary in boundaries:
                if bmi >= boundary:
                    group += 1
                else:
                    break
            group_assignments[i] = group

        return group_assignments

    def evaluate_solution(self, solution_vector, num_groups):
        """评估解决方案的风险（改进版本）"""
        try:
            # 解码解决方案
            boundaries = solution_vector[:num_groups - 1]
            test_times = solution_vector[num_groups - 1:num_groups - 1 + num_groups]

            # 确保边界点单调递增且有合理间隔
            boundaries = np.sort(boundaries)
            boundaries = np.clip(boundaries, self.bmi_min + 0.1, self.bmi_max - 0.1)

            # 检查边界点间距
            if len(boundaries) > 1:
                min_gap = np.min(np.diff(boundaries))
                if min_gap < 1.0:  # BMI间距至少1.0
                    return 50000.0

            # 确保检测时间在合理范围内
            test_times = np.clip(test_times, 10.0, 28.0)

            group_assignments = self.assign_groups(boundaries)

            total_risk = 0
            total_weight = 0
            penalty = 0

            # 检查每组的样本量和风险
            group_sizes = []
            for group_id in range(num_groups):
                group_mask = group_assignments == group_id
                group_size = np.sum(group_mask)
                group_sizes.append(group_size)

                if group_size < self.min_group_size:
                    # 样本量不足的严重惩罚，随分组数增加
                    penalty_factor = 2000.0 * (1 + 0.2 * num_groups)
                    penalty += penalty_factor * (self.min_group_size - group_size) ** 2

                if group_size > 0:
                    group_reach_times = self.reach_times[group_mask]
                    test_time = test_times[group_id]

                    group_risk, _ = self.risk_function.calculate_group_risk(
                        test_time, group_reach_times
                    )

                    # 对过小的组给予额外惩罚
                    if group_size < self.min_group_size * 1.5:
                        group_risk *= (1 + 0.2 * (self.min_group_size * 1.5 - group_size))

                    total_risk += group_risk * group_size
                    total_weight += group_size

            # 检查组大小分布的均衡性
            if len(group_sizes) > 1:
                size_std = np.std(group_sizes)
                size_mean = np.mean(group_sizes)
                if size_mean > 0:
                    cv = size_std / size_mean  # 变异系数
                    if cv > 1.0:  # 分布过不均衡
                        penalty += 100.0 * cv

            if total_weight == 0:
                return 50000.0

            avg_risk = total_risk / total_weight + penalty

            # 添加轻微的复杂度惩罚
            complexity_penalty = 0.001 * (num_groups - 2) ** 1.5
            avg_risk += complexity_penalty

            return avg_risk

        except Exception as e:
            return 50000.0

    def get_smart_initial_boundaries(self, num_groups):
        """智能生成初始边界点"""
        # 基于BMI分布的分位数来设置初始边界
        if num_groups <= 2:
            return []

        # 使用分位数作为初始边界
        quantiles = np.linspace(1 / (num_groups), 1 - 1 / (num_groups), num_groups - 1)
        boundaries = np.quantile(self.bmi_values, quantiles)

        # 确保边界点有合理间隔
        min_gap = 1.5
        for i in range(1, len(boundaries)):
            if boundaries[i] - boundaries[i - 1] < min_gap:
                boundaries[i] = boundaries[i - 1] + min_gap

        # 调整超出范围的边界点
        boundaries = np.clip(boundaries, self.bmi_min + 1, self.bmi_max - 1)

        return boundaries

    def optimize_for_k_groups(self, num_groups):
        """为指定数量的组进行优化（增强版本）"""
        print(f"  优化 {num_groups} 个分组...")

        # 检查可行性
        min_required_samples = num_groups * self.min_group_size
        if len(self.data) < min_required_samples:
            print(f"    样本量不足: 需要{min_required_samples}, 实际{len(self.data)}")
            return {'success': False, 'num_groups': num_groups}

        # 智能设置边界点范围
        bmi_range = self.bmi_max - self.bmi_min
        min_gap = max(1.0, bmi_range / (num_groups * 3))  # 确保有足够间隔

        bounds = []

        # 边界点的边界 - 更智能的设置
        for i in range(num_groups - 1):
            lower = self.bmi_min + min_gap * (i + 1)
            upper = self.bmi_max - min_gap * (num_groups - i - 1)
            bounds.append((lower, upper))

        # 检测时间的边界
        for i in range(num_groups):
            bounds.append((10.0, 28.0))

        # 多次尝试优化
        best_result = None
        best_risk = float('inf')

        # 动态调整算法参数
        max_iter = min(600 + num_groups * 150, 1500)
        pop_size = min(60 + num_groups * 15, 150)

        for attempt in range(self.optimization_attempts):
            try:
                # 为每次尝试使用不同的参数
                mutation_range = (0.3 + attempt * 0.1, 1.8 + attempt * 0.1)
                recombination = 0.7 + attempt * 0.05

                result = differential_evolution(
                    lambda x: self.evaluate_solution(x, num_groups),
                    bounds,
                    seed=42 + attempt * 7,
                    maxiter=max_iter,
                    popsize=pop_size,
                    mutation=mutation_range,
                    recombination=min(recombination, 0.9),
                    polish=True,
                    atol=1e-8,
                    tol=1e-8,
                    workers=1,
                    updating='deferred' if num_groups > 4 else 'immediate'
                )

                if result.success and result.fun < best_risk and result.fun < 40000:
                    best_result = result
                    best_risk = result.fun
                    print(f"    尝试 {attempt + 1}: 风险 = {result.fun:.4f} ✓")
                else:
                    print(f"    尝试 {attempt + 1}: {'成功' if result.success else '失败'}, 风险 = {result.fun:.1f}")

            except Exception as e:
                print(f"    尝试 {attempt + 1} 异常: {str(e)}")
                continue

        if best_result is not None:
            boundaries = best_result.x[:num_groups - 1]
            test_times = best_result.x[num_groups - 1:num_groups - 1 + num_groups]

            # 确保边界点单调递增
            boundaries = np.sort(boundaries)
            boundaries = np.clip(boundaries, self.bmi_min + 0.5, self.bmi_max - 0.5)
            test_times = np.clip(test_times, 10.0, 28.0)

            # 验证解的可行性
            group_assignments = self.assign_groups(boundaries)
            group_sizes = [np.sum(group_assignments == i) for i in range(num_groups)]

            if all(size >= self.min_group_size for size in group_sizes):
                print(f"    ✓ 成功! 组大小: {group_sizes}")
                return {
                    'num_groups': num_groups,
                    'boundaries': boundaries,
                    'test_times': test_times,
                    'risk': best_result.fun,
                    'success': True,
                    'optimization_result': best_result,
                    'group_sizes': group_sizes
                }
            else:
                print(f"    ✗ 解不满足最小组大小要求: {group_sizes}")
                return {'success': False, 'num_groups': num_groups}
        else:
            print(f"    ✗ 所有尝试都失败")
            return {'success': False, 'num_groups': num_groups}

    def run_optimization(self):
        """运行完整的优化过程"""
        print("=" * 70)
        print("开始鲁棒BMI分组优化")
        print("=" * 70)

        best_solution = None
        best_risk = float('inf')
        all_solutions = []

        # 尝试不同的分组数
        for k in range(2, self.max_groups + 1):
            print(f"\n【{k}组优化】")
            solution = self.optimize_for_k_groups(k)

            if solution['success']:
                all_solutions.append(solution)
                group_sizes = solution.get('group_sizes', [])
                test_times = solution['test_times']
                boundaries = solution['boundaries']

                print(f"  成功! 风险 = {solution['risk']:.4f}")
                print(f"  组大小: {group_sizes}")
                print(f"  最佳NIPT时间: {[f'{t:.1f}周' for t in test_times]}")

                # 显示BMI分组范围和对应的检测时间
                print(f"  分组方案:")
                if k == 1:
                    print(f"    组1: BMI [{self.bmi_min:.1f}, {self.bmi_max:.1f}] → 检测时间: {test_times[0]:.1f}周")
                else:
                    # 第一组
                    print(f"    组1: BMI [{self.bmi_min:.1f}, {boundaries[0]:.1f}) → 检测时间: {test_times[0]:.1f}周")
                    # 中间组
                    for i in range(1, len(boundaries)):
                        print(
                            f"    组{i + 1}: BMI [{boundaries[i - 1]:.1f}, {boundaries[i]:.1f}) → 检测时间: {test_times[i]:.1f}周")
                    # 最后一组
                    print(
                        f"    组{len(boundaries) + 1}: BMI [{boundaries[-1]:.1f}, {self.bmi_max:.1f}] → 检测时间: {test_times[-1]:.1f}周")

                if solution['risk'] < best_risk:
                    best_risk = solution['risk']
                    best_solution = solution
            else:
                print(f"  失败 - 可能原因: 样本量不足或约束冲突")

        if best_solution is not None:
            self.best_solution = best_solution
            self.optimization_history = all_solutions

            print(f"\n🎯 最佳方案: {best_solution['num_groups']}组, 风险 = {best_risk:.4f}")
            return best_solution
        else:
            print("\n❌ 所有优化方案都失败了！")
            print("建议: 减少最大分组数或降低最小组大小要求")
            return None

    def get_detailed_results(self):
        """获取详细的优化结果"""
        if self.best_solution is None:
            print("没有可用的优化结果！")
            return None

        solution = self.best_solution
        boundaries = solution['boundaries']
        test_times = solution['test_times']
        num_groups = solution['num_groups']

        # 分配组别
        group_assignments = self.assign_groups(boundaries)

        # 构建边界信息
        bmi_ranges = []
        if num_groups == 1:
            bmi_ranges.append(f"[{self.bmi_min:.1f}, {self.bmi_max:.1f}]")
        else:
            # 第一组
            bmi_ranges.append(f"[{self.bmi_min:.1f}, {boundaries[0]:.1f})")

            # 中间组
            for i in range(1, len(boundaries)):
                bmi_ranges.append(f"[{boundaries[i - 1]:.1f}, {boundaries[i]:.1f})")

            # 最后一组
            bmi_ranges.append(f"[{boundaries[-1]:.1f}, {self.bmi_max:.1f}]")

        # 计算每组的详细统计
        group_details = []
        total_risk = 0
        total_samples = 0

        for group_id in range(num_groups):
            group_mask = group_assignments == group_id
            group_bmis = self.bmi_values[group_mask]
            group_reach_times = self.reach_times[group_mask]
            test_time = test_times[group_id]

            group_size = len(group_reach_times)

            # 使用增强风险函数计算详细信息
            risk_details = self.risk_function.calculate_group_risk(
                test_time, group_reach_times, return_details=True
            )

            # 计算达标时间统计
            reach_time_stats = {
                'mean': np.mean(group_reach_times),
                'median': np.median(group_reach_times),
                'std': np.std(group_reach_times),
                'min': np.min(group_reach_times),
                'max': np.max(group_reach_times)
            }

            # 计算时间差统计
            time_diffs = group_reach_times - test_time

            group_details.append({
                'group_id': group_id + 1,
                'bmi_range': bmi_ranges[group_id],
                'size': group_size,
                'avg_bmi': np.mean(group_bmis),
                'optimal_test_time': test_time,
                'group_risk': risk_details['avg_risk'],
                'detection_success_rate': risk_details['detection_success_rate'],
                'early_count': risk_details['early_count'],
                'late_count': risk_details['late_count'],
                'on_time_count': risk_details['on_time_count'],
                'reach_time_stats': reach_time_stats,
                'avg_time_diff': np.mean(time_diffs),
                'std_time_diff': np.std(time_diffs)
            })

            total_risk += risk_details['avg_risk'] * group_size
            total_samples += group_size

        overall_risk = total_risk / total_samples

        return {
            'solution': solution,
            'group_details': group_details,
            'overall_risk': overall_risk,
            'group_assignments': group_assignments
        }

    def print_detailed_results(self):
        """打印详细结果"""
        detailed_results = self.get_detailed_results()

        if detailed_results is None:
            return

        print("\n" + "=" * 90)
        print("鲁棒BMI分组优化详细结果")
        print("=" * 90)

        print(f"最优分组数: {self.best_solution['num_groups']}")
        print(f"总体风险: {detailed_results['overall_risk']:.4f}")

        print(f"\n各组详细信息:")
        print("-" * 90)

        for group in detailed_results['group_details']:
            print(f"\n组 {group['group_id']}:")
            print(f"  BMI范围: {group['bmi_range']}")
            print(f"  样本数量: {group['size']}")
            print(f"  平均BMI: {group['avg_bmi']:.2f}")
            print(f"  最佳检测时间: {group['optimal_test_time']:.1f} 周")
            print(f"  组风险: {group['group_risk']:.4f}")
            print(f"  检测成功率: {group['detection_success_rate']:.1%}")

            print(f"  检测时机分布:")
            print(f"    早期检测: {group['early_count']} 人")
            print(f"    准时检测: {group['on_time_count']} 人")
            print(f"    晚期检测: {group['late_count']} 人")
            print(f"    平均时间差: {group['avg_time_diff']:.2f} 周 (±{group['std_time_diff']:.2f})")

            stats = group['reach_time_stats']
            print(f"  预计达标时间统计:")
            print(f"    均值: {stats['mean']:.1f}周, 中位数: {stats['median']:.1f}周")
            print(f"    范围: {stats['min']:.1f}-{stats['max']:.1f}周, 标准差: {stats['std']:.1f}周")

        print("\n" + "=" * 90)

        # 生成最终建议
        print("临床应用建议:")
        print("-" * 50)

        for group in detailed_results['group_details']:
            if group['group_risk'] < 2:
                risk_level = "低风险"
            elif group['group_risk'] < 5:
                risk_level = "中风险"
            else:
                risk_level = "高风险"

            early_rate = group['early_count'] / group['size'] * 100
            late_rate = group['late_count'] / group['size'] * 100

            print(f"BMI {group['bmi_range']}: 第 {group['optimal_test_time']:.1f} 周检测")
            print(f"  → {risk_level}, 成功率 {group['detection_success_rate']:.1%}, "
                  f"早检 {early_rate:.1f}%, 晚检 {late_rate:.1f}%")


def run_enhanced_medical_optimization(prediction_results_file='reach_time_results_Q3.csv'):
    """运行增强医学风险优化的主函数"""
    try:
        # 读取预测结果
        print("读取预测结果...")
        data = pd.read_csv(prediction_results_file)
        print(f"数据加载完成，共 {len(data)} 个样本")

        # 创建增强的医学风险函数（大幅增加过晚检测斜率）
        risk_function = EnhancedMedicalRiskFunction(
            early_quadratic_coeff=0.02,  # 早期检测二次函数系数
            late_exp_base_1=0.3,  # 第1段指数基数
            late_exp_rate_1=0.15,  # 第1段指数增长率
            late_exp_base_2=2.5,  # 第2段指数基数（大幅增加）
            late_exp_rate_2=0.40,  # 第2段指数增长率（大幅增加）
            late_exp_base_3=6.0,  # 第3段指数基数（大幅增加）
            late_exp_rate_3=0.65,  # 第3段指数增长率（极大增加）
            base_risk=1.0  # 基础风险
        )

        # 展示增强的风险函数特性
        print("\n展示增强医学风险函数特性...")
        risk_function.plot_risk_function(reach_time=11, save_path='exp_medical_risk_function.png')

        # 创建鲁棒优化器
        optimizer = RobustBMIGroupingOptimizer(
            data=data,
            risk_function=risk_function,
            min_group_size=6,  # 降低最小组大小要求
            max_groups=5,  # 最大分组数
            optimization_attempts=1  # 每个分组尝试5次
        )

        # 运行优化
        best_solution = optimizer.run_optimization()

        if best_solution is not None:
            # 打印详细结果
            optimizer.print_detailed_results()

            # 绘制最终分组结果可视化
            # print("\n生成最终分组结果可视化...")
            # optimizer.plot_final_grouping_results('final_bmi_grouping_visualization.png')

            # 保存结果
            detailed_results = optimizer.get_detailed_results()
            result_data = data.copy()
            result_data['Group'] = detailed_results['group_assignments'] + 1

            # 添加每组的最佳检测时间和风险信息
            for group in detailed_results['group_details']:
                group_mask = result_data['Group'] == group['group_id']
                result_data.loc[group_mask, 'Optimal_Test_Time'] = group['optimal_test_time']
                result_data.loc[group_mask, 'Group_Risk'] = group['group_risk']
                result_data.loc[group_mask, 'Detection_Success_Rate'] = group['detection_success_rate']

            result_data.to_csv('exp_medical_bmi_grouping_results.csv', index=False)
            print(f"\n结果已保存到: exp_medical_bmi_grouping_results.csv")

            return optimizer
        else:
            print("\n优化失败！")
            return None

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    optimizer = run_enhanced_medical_optimization()