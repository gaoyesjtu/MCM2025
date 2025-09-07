import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')

# 设置微软雅黑字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SimplePerturbationAnalyzer:
    """简化版扰动分析器"""

    def __init__(self, original_data):
        self.original_data = original_data.copy()

    def add_perturbation(self, bmi_std=0.5, reach_time_std=0.8):
        """添加高斯扰动"""
        perturbed_data = self.original_data.copy()
        n_samples = len(perturbed_data)

        # 高斯扰动
        bmi_noise = np.random.normal(0, bmi_std, n_samples)
        time_noise = np.random.normal(0, reach_time_std, n_samples)

        # 应用扰动
        perturbed_data['BMI'] = perturbed_data['BMI'] + bmi_noise
        perturbed_data['预计到达时间'] = perturbed_data['预计到达时间'] + time_noise

        # 确保数据在合理范围内
        perturbed_data['BMI'] = np.clip(perturbed_data['BMI'], 15, 45)
        perturbed_data['预计到达时间'] = np.clip(perturbed_data['预计到达时间'], 8, 25)

        return perturbed_data


class EnhancedMedicalRiskFunction:
    """改进的医学风险函数：过早检测用一次函数，过晚检测用分段二次函数"""

    def __init__(self,
                 early_linear_coeff=0.05,  # 一次函数系数
                 late_quad_coeff_1=0.008,  # 第1段二次函数系数
                 late_quad_coeff_2=0.015,  # 第2段二次函数系数
                 late_quad_coeff_3=0.055,  # 第3段二次函数系数
                 base_risk=1.0):
        """
        风险函数参数：

        过早检测（test_time < reach_time）：
        - early_linear_coeff: 一次函数系数，风险 = base_risk + coeff * (reach_time - test_time)

        过晚检测（test_time > reach_time）分段二次函数：
        - 第1段 (test_time ≤ 12): coeff_1 * delay²
        - 第2段 (12 < test_time ≤ 27): 连续拼接的二次函数
        - 第3段 (test_time > 27): 连续拼接的二次函数

        - base_risk: 基础风险
        """
        self.early_linear_coeff = early_linear_coeff
        self.late_quad_coeff_1 = late_quad_coeff_1
        self.late_quad_coeff_2 = late_quad_coeff_2
        self.late_quad_coeff_3 = late_quad_coeff_3
        self.base_risk = base_risk

    def calculate_individual_risk(self, test_time, reach_time):
        """计算单个孕妇的风险"""

        if test_time <= reach_time:
            # 过早检测：一次函数风险
            time_diff = reach_time - test_time
            early_risk = self.early_linear_coeff * time_diff - 0.1
            return self.base_risk + early_risk

        else:
            # 过晚检测：分段二次函数风险（基于test_time分段）
            delay = test_time - reach_time

            if test_time <= 13:
                # 第1段：温和的二次增长
                late_risk = self.late_quad_coeff_1 * (delay ** 2)

            elif test_time <= 27:
                # 第2段：中等的二次增长
                # 确保连续性：计算第1段在边界点的风险值
                if reach_time >= 13:
                    # 如果reach_time >= 13，第1段不存在，直接从第2段开始
                    late_risk = self.late_quad_coeff_2 * ((delay + 4) ** 2)
                else:
                    # 第1段在test_time=12处的风险值
                    boundary_1_delay = 13 - reach_time
                    boundary_1_value = self.late_quad_coeff_1 * (boundary_1_delay ** 2)

                    # 第2段的增量风险（从test_time=12开始的额外延迟）
                    segment_2_delay = test_time - 13
                    segment_2_risk = self.late_quad_coeff_2 * ((segment_2_delay + 4) ** 2)
                    late_risk = boundary_1_value + segment_2_risk

            else:
                # 第3段：较强的二次增长
                # 计算前两段的累积风险
                if reach_time >= 27:
                    # 如果reach_time >= 27，前两段都不存在
                    late_risk = self.late_quad_coeff_3 * ((delay + 4) ** 2)
                elif reach_time >= 13:
                    # 只有第2段存在
                    boundary_2_delay = 27 - reach_time
                    boundary_2_value = self.late_quad_coeff_2 * (boundary_2_delay ** 2)
                    segment_3_delay = test_time - 27
                    segment_3_risk = self.late_quad_coeff_3 * ((segment_3_delay + 8) ** 2)
                    late_risk = boundary_2_value + segment_3_risk
                else:
                    # 三段都存在
                    # 第1段在test_time=12处的风险
                    boundary_1_delay = 13 - reach_time
                    boundary_1_value = self.late_quad_coeff_1 * (boundary_1_delay ** 2)

                    # 第2段在test_time=27处的风险（从12到27的增量）
                    segment_2_length = 27 - 13  # 15周
                    boundary_2_increment = self.late_quad_coeff_2 * (segment_2_length ** 2)

                    # 第3段的风险（从27开始）
                    segment_3_delay = test_time - 27
                    segment_3_risk = self.late_quad_coeff_3 * ((segment_3_delay + 8) ** 2)

                    late_risk = boundary_1_value + boundary_2_increment + segment_3_risk

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

    def plot_risk_function(self, reach_time=11, test_time_range=(8, 35), save_path=None):
        """绘制风险函数形状"""
        test_times = np.linspace(test_time_range[0], test_time_range[1], 500)
        risks = [self.calculate_individual_risk(t, reach_time) for t in test_times]

        plt.figure(figsize=(16, 12))

        # 创建子图
        gs = plt.GridSpec(3, 2, hspace=0.4, wspace=0.3)

        # 主图：完整的风险函数
        ax1 = plt.subplot(gs[0, :])
        plt.plot(test_times, risks, 'b-', linewidth=3, label='Risk Function (Linear + Quadratic)')

        # 标记达标时间点
        plt.axvline(x=reach_time, color='red', linestyle='--', linewidth=2,
                    label=f'Reach Time = {reach_time} weeks')

        # 分别绘制早期和晚期区域
        early_times = test_times[test_times < reach_time]
        late_times = test_times[test_times > reach_time]

        if len(early_times) > 0:
            early_risks = [self.calculate_individual_risk(t, reach_time) for t in early_times]
            plt.fill_between(early_times, early_risks, alpha=0.3, color='green',
                             label='Early Detection (Linear)')

        if len(late_times) > 0:
            late_risks = [self.calculate_individual_risk(t, reach_time) for t in late_times]
            plt.fill_between(late_times, late_risks, alpha=0.3, color='orange',
                             label='Late Detection (Piecewise Quadratic)')

        # 标记分段边界（基于test_time）
        plt.axvline(x=13, color='orange', linestyle=':', alpha=0.7,
                    label='Segment 1→2 (test_time=13w)')
        plt.axvline(x=27, color='red', linestyle=':', alpha=0.7,
                    label='Segment 2→3 (test_time=27w)')

        plt.xlabel('Test Time (weeks)', fontsize=12)
        plt.ylabel('Risk', fontsize=12)
        plt.title(
            f'Medical Risk Function: Linear (Early) + Piecewise Quadratic (Late)\n(Reach Time = {reach_time} weeks)',
            fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 设置合理的y轴范围
        plt.ylim(0, max(risks) * 1.1)

        # 子图1：早期检测部分详细显示（一次函数）
        ax2 = plt.subplot(gs[1, 0])
        early_range = np.linspace(max(reach_time - 8, test_time_range[0]), reach_time, 100)
        early_risks = [self.calculate_individual_risk(t, reach_time) for t in early_range]
        plt.plot(early_range, early_risks, 'g-', linewidth=3, label='Linear Risk')
        plt.axvline(x=reach_time, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Test Time (weeks)')
        plt.ylabel('Risk')
        plt.title('Early Detection: Linear Risk')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：晚期检测部分详细显示（分段二次函数）
        ax3 = plt.subplot(gs[1, 1])
        late_range = np.linspace(reach_time, min(reach_time + 25, test_time_range[1]), 200)
        late_risks = [self.calculate_individual_risk(t, reach_time) for t in late_range]

        # 分别绘制不同段的颜色
        for i, t in enumerate(late_range):
            if t <= 13:
                color = 'lightblue'
            elif t <= 27:
                color = 'orange'
            else:
                color = 'red'
            if i == 0 or (i > 0 and t <= 13 and late_range[i - 1] > 13) or (
                    i > 0 and t <= 27 and late_range[i - 1] > 27):
                plt.plot(t, late_risks[i], 'o', color=color, markersize=4, alpha=0.7)

        plt.plot(late_range, late_risks, 'purple', linewidth=3, label='Piecewise Quadratic')
        plt.axvline(x=reach_time, color='red', linestyle='--', alpha=0.7)
        plt.axvline(x=13, color='orange', linestyle=':', alpha=0.7, label='Boundary 13w')
        plt.axvline(x=27, color='red', linestyle=':', alpha=0.7, label='Boundary 27w')
        plt.xlabel('Test Time (weeks)')
        plt.ylabel('Risk')
        plt.title('Late Detection: Piecewise Quadratic Risk')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图3：不同reach_time下的风险对比
        ax4 = plt.subplot(gs[2, 0])
        test_range = np.linspace(8, 30, 200)
        for rt in [9, 11, 13]:
            rt_risks = [self.calculate_individual_risk(t, rt) for t in test_range]
            plt.plot(test_range, rt_risks, linewidth=2, label=f'reach_time={rt}')

        plt.axvline(x=13, color='orange', linestyle=':', alpha=0.7)
        plt.axvline(x=27, color='red', linestyle=':', alpha=0.7)
        plt.xlabel('Test Time (weeks)')
        plt.ylabel('Risk')
        plt.title('Risk Comparison for Different Reach Times')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图4：风险函数的导数（增长率）分析
        ax5 = plt.subplot(gs[2, 1])
        # 计算数值导数来显示风险增长率
        dt = 0.1
        test_range_deriv = np.arange(reach_time + 0.1, min(30, test_time_range[1]), dt)
        derivatives = []

        for t in test_range_deriv:
            risk_1 = self.calculate_individual_risk(t - dt / 2, reach_time)
            risk_2 = self.calculate_individual_risk(t + dt / 2, reach_time)
            deriv = (risk_2 - risk_1) / dt
            derivatives.append(deriv)

        plt.plot(test_range_deriv, derivatives, 'purple', linewidth=2, label='Risk Growth Rate')
        plt.axvline(x=13, color='orange', linestyle=':', alpha=0.7, label='Boundary 13w')
        plt.axvline(x=27, color='red', linestyle=':', alpha=0.7, label='Boundary 27w')
        plt.xlabel('Test Time (weeks)')
        plt.ylabel('Risk Growth Rate (dRisk/dt)')
        plt.title('Risk Growth Rate Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

        # 打印关键风险值用于验证
        print(f"\n=== 风险值验证 (reach_time = {reach_time}) ===")
        test_points = [7, 8, 9, 10, 11, 12, 15, 18, 21, 24, 27, 30, 35]
        print(f"{'检测时间':>6} {'状态':>12} {'风险值':>8} {'函数类型':>15}")
        print("-" * 50)

        for t in test_points:
            risk = self.calculate_individual_risk(t, reach_time)
            if t < reach_time:
                status = f"提前{reach_time - t}周"
                func_type = "线性函数"
            elif t == reach_time:
                status = "准时"
                func_type = "基础风险"
            else:
                status = f"延迟{t - reach_time}周"
                # 显示分段信息
                if t <= 13:
                    func_type = "二次函数(段1)"
                elif t <= 27:
                    func_type = "二次函数(段2)"
                else:
                    func_type = "二次函数(段3)"

            print(f"{t:6d}周 {status:>12s} {risk:8.2f} {func_type:>15s}")

    def find_optimal_test_time(self, reach_times, test_time_range=(10, 30)):
        """为给定的达标时间分布找到最优检测时间"""
        test_times = np.linspace(test_time_range[0], test_time_range[1], 200)
        risks = [self.calculate_group_risk(t, reach_times)[0] for t in test_times]

        optimal_idx = np.argmin(risks)
        optimal_time = test_times[optimal_idx]
        optimal_risk = risks[optimal_idx]

        return optimal_time, optimal_risk


class RobustBMIGroupingOptimizer:
    """快速版BMI分组优化器 - 保持完整功能但优化速度"""

    def __init__(self, data, risk_function, min_group_size=6, max_groups=4):
        self.data = data.copy()
        self.risk_function = risk_function
        self.min_group_size = min_group_size
        self.max_groups = max_groups

        # 数据预处理
        self.bmi_values = self.data['BMI'].values
        self.reach_times = self.data['预计到达时间'].values
        self.bmi_min = self.bmi_values.min()
        self.bmi_max = self.bmi_values.max()

        # 可行性检查
        max_feasible_groups = len(self.data) // self.min_group_size
        if self.max_groups > max_feasible_groups:
            self.max_groups = min(self.max_groups, max_feasible_groups)

        self.best_solution = None

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
        """评估解决方案的风险"""
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
                    # 样本量不足的严重惩罚
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

            if total_weight == 0:
                return 50000.0

            avg_risk = total_risk / total_weight + penalty

            # 添加轻微的复杂度惩罚
            complexity_penalty = 0.001 * (num_groups - 2) ** 1.5
            avg_risk += complexity_penalty

            return avg_risk

        except Exception as e:
            return 50000.0

    def optimize_for_k_groups(self, num_groups):
        """为指定数量的组进行优化（快速版本）"""
        # 检查可行性
        min_required_samples = num_groups * self.min_group_size
        if len(self.data) < min_required_samples:
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

        # 快速优化参数
        result = differential_evolution(
            lambda x: self.evaluate_solution(x, num_groups),
            bounds,
            seed=42,
            maxiter=80,  # 减少迭代次数
            popsize=15,  # 减少种群大小
            polish=False,  # 关闭局部优化
            atol=1e-6,
            tol=1e-6,
            workers=1
        )

        if result.success and result.fun < 40000:
            boundaries = result.x[:num_groups - 1]
            test_times = result.x[num_groups - 1:num_groups - 1 + num_groups]

            # 确保边界点单调递增
            boundaries = np.sort(boundaries)
            boundaries = np.clip(boundaries, self.bmi_min + 0.5, self.bmi_max - 0.5)
            test_times = np.clip(test_times, 10.0, 28.0)

            # 验证解的可行性
            group_assignments = self.assign_groups(boundaries)
            group_sizes = [np.sum(group_assignments == i) for i in range(num_groups)]

            if all(size >= self.min_group_size for size in group_sizes):
                return {
                    'num_groups': num_groups,
                    'boundaries': boundaries,
                    'test_times': test_times,
                    'risk': result.fun,
                    'success': True,
                    'optimization_result': result,
                    'group_sizes': group_sizes
                }
            else:
                return {'success': False, 'num_groups': num_groups}
        else:
            return {'success': False, 'num_groups': num_groups}

    def run_optimization(self, verbose=False):
        """运行完整的优化过程"""
        if verbose:
            print("开始快速BMI分组优化")

        best_solution = None
        best_risk = float('inf')
        all_solutions = []

        # 尝试不同的分组数
        for k in range(2, self.max_groups + 1):
            if verbose:
                print(f"  优化 {k} 组...")
            solution = self.optimize_for_k_groups(k)

            if solution['success']:
                all_solutions.append(solution)
                if verbose:
                    print(f"    成功! 风险 = {solution['risk']:.4f}")

                if solution['risk'] < best_risk:
                    best_risk = solution['risk']
                    best_solution = solution
            else:
                if verbose:
                    print(f"    失败")

        if best_solution is not None:
            self.best_solution = best_solution
            if verbose:
                print(f"最佳方案: {best_solution['num_groups']}组, 风险 = {best_risk:.4f}")
            return best_solution
        else:
            if verbose:
                print("所有优化方案都失败了！")
            return None

    def get_detailed_results(self):
        """获取详细的优化结果"""
        if self.best_solution is None:
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


class SimplePerturbationImpactAnalyzer:
    """扰动影响分析器 - 保持完整功能，只优化算法速度"""

    def __init__(self, original_data):
        self.original_data = original_data
        self.perturbation_analyzer = SimplePerturbationAnalyzer(original_data)
        self.risk_function = EnhancedMedicalRiskFunction()
        self.analysis_results = []
        self.original_solution = None

    def run_single_perturbation_analysis(self, bmi_std=0.5, time_std=0.8):
        """运行单次扰动分析"""
        # 生成扰动数据
        perturbed_data = self.perturbation_analyzer.add_perturbation(bmi_std, time_std)

        # 优化扰动后的数据
        optimizer = RobustBMIGroupingOptimizer(
            data=perturbed_data,
            risk_function=self.risk_function,
            min_group_size=6,
            max_groups=4
        )

        solution = optimizer.run_optimization(verbose=False)

        if solution is not None:
            detailed_results = optimizer.get_detailed_results()
            return {
                'success': True,
                'bmi_std': bmi_std,
                'time_std': time_std,
                'solution': solution,
                'detailed_results': detailed_results,
                'optimizer': optimizer,
                'perturbed_data': perturbed_data
            }
        else:
            return {
                'success': False,
                'bmi_std': bmi_std,
                'time_std': time_std
            }

    def run_perturbation_analysis(self):
        """运行扰动分析 - 保持完整分析功能"""
        print("=" * 50)
        print("扰动影响分析 - 完整版")
        print("=" * 50)

        # 1. 原始数据优化
        print("1. 分析原始数据...")
        original_optimizer = RobustBMIGroupingOptimizer(
            data=self.original_data,
            risk_function=self.risk_function,
            min_group_size=6,
            max_groups=4
        )

        self.original_solution = original_optimizer.run_optimization(verbose=False)

        if self.original_solution is None:
            print("❌ 原始数据优化失败！")
            return None

        # 获取原始解的详细结果
        self.original_detailed = original_optimizer.get_detailed_results()

        print(f"✓ 原始数据最佳方案: {self.original_solution['num_groups']}组, "
              f"风险 = {self.original_solution['risk']:.4f}")

        # 2. 扰动分析 - 测试多种扰动参数
        print("\n2. 运行扰动测试...")

        # 扰动参数设置
        bmi_stds = [0.2, 0.5, 0.8]  # BMI扰动标准差
        time_stds = [0.3, 0.6, 1.0]  # 时间扰动标准差
        n_replications = 2  # 每种参数组合重复次数

        total_scenarios = len(bmi_stds) * len(time_stds) * n_replications
        scenario_count = 0

        print(f"总共 {total_scenarios} 个扰动场景...")

        for bmi_std in bmi_stds:
            for time_std in time_stds:
                scenario_results = []

                for rep in range(n_replications):
                    scenario_count += 1

                    # 设置随机种子确保可重复性
                    np.random.seed(42 + scenario_count)

                    print(
                        f"  场景 {scenario_count}/{total_scenarios}: BMI±{bmi_std}, 时间±{time_std}周, 重复#{rep + 1}")

                    result = self.run_single_perturbation_analysis(bmi_std, time_std)

                    if result['success']:
                        scenario_results.append(result)
                        print(f"    成功! 风险 = {result['solution']['risk']:.4f}")
                    else:
                        print(f"    失败")

                # 汇总当前参数组合的结果
                if scenario_results:
                    self._summarize_scenario_results(bmi_std, time_std, scenario_results)

        print(f"\n✓ 扰动分析完成! 共分析了 {len(self.analysis_results)} 个有效场景组合")

        return self.analysis_results

    def _summarize_scenario_results(self, bmi_std, time_std, results):
        """汇总单个场景的多次重复结果"""

        if not results:
            return

        # 提取关键指标
        risks = [r['solution']['risk'] for r in results]
        num_groups_list = [r['solution']['num_groups'] for r in results]

        # 计算统计量
        risk_mean = np.mean(risks)
        risk_std = np.std(risks)
        risk_min = np.min(risks)
        risk_max = np.max(risks)

        # 计算与原始解的偏差
        original_risk = self.original_solution['risk']
        risk_deviation_mean = risk_mean - original_risk
        risk_deviation_std = risk_std

        # 分组数变化
        original_groups = self.original_solution['num_groups']
        group_changes = [g - original_groups for g in num_groups_list]
        group_change_freq = {}
        for change in group_changes:
            group_change_freq[change] = group_change_freq.get(change, 0) + 1

        # 存储汇总结果
        summary = {
            'bmi_std': bmi_std,
            'time_std': time_std,
            'n_successful_runs': len(results),
            'original_risk': original_risk,
            'original_groups': original_groups,
            'risk_statistics': {
                'mean': risk_mean,
                'std': risk_std,
                'min': risk_min,
                'max': risk_max,
                'deviation_mean': risk_deviation_mean,
                'deviation_std': risk_deviation_std,
                'relative_deviation': risk_deviation_mean / original_risk * 100
            },
            'group_changes': group_change_freq,
            'detailed_results': results
        }

        self.analysis_results.append(summary)

    def generate_stability_report(self):
        """生成详细的稳定性分析报告"""

        if not self.analysis_results:
            print("❌ 没有可用的分析结果！")
            return

        print("\n" + "=" * 80)
        print("扰动稳定性分析报告")
        print("=" * 80)

        print(f"原始解决方案: {self.original_solution['num_groups']}组, "
              f"风险 = {self.original_solution['risk']:.4f}")

        # 显示原始解的详细信息
        if hasattr(self, 'original_detailed') and self.original_detailed:
            print(f"\n原始解决方案详细信息:")
            print("-" * 50)
            for group in self.original_detailed['group_details']:
                print(f"组 {group['group_id']}: BMI {group['bmi_range']}, "
                      f"样本数 {group['size']}, 最佳检测时间 {group['optimal_test_time']:.1f}周")

        print(f"\n分析了 {len(self.analysis_results)} 个扰动场景组合")

        # 按扰动强度分析
        print(f"\n扰动场景分析:")
        print("-" * 50)

        for result in self.analysis_results:
            bmi_std = result['bmi_std']
            time_std = result['time_std']
            n_success = result['n_successful_runs']
            risk_stats = result['risk_statistics']
            group_changes = result['group_changes']

            print(f"\n扰动参数: BMI±{bmi_std:.1f}, 时间±{time_std:.1f}周")
            print(f"  成功运行: {n_success} 次")
            print(f"  风险变化: {risk_stats['relative_deviation']:+.2f}% "
                  f"(标准差: {risk_stats['deviation_std']:.4f})")
            print(f"  风险范围: {risk_stats['min']:.4f} - {risk_stats['max']:.4f}")

            # 分组变化分析
            print(f"  分组变化分布:")
            total_runs = sum(group_changes.values())
            for change in sorted(group_changes.keys()):
                freq = group_changes[change]
                percentage = freq / total_runs * 100
                print(f"    变化{change:+d}组: {freq}次 ({percentage:.1f}%)")

        # 计算总体稳定性指标
        self._calculate_overall_stability()

    def _calculate_overall_stability(self):
        """计算总体稳定性指标"""

        # 计算总体稳定性指标
        all_relative_deviations = []
        all_group_changes = []

        for result in self.analysis_results:
            all_relative_deviations.append(abs(result['risk_statistics']['relative_deviation']))

            # 计算分组变化的权重
            for change, freq in result['group_changes'].items():
                all_group_changes.extend([abs(change)] * freq)

        # 风险稳定性评分 (0-100)
        avg_risk_deviation = np.mean(all_relative_deviations)
        risk_stability_score = max(0, 100 - avg_risk_deviation * 2)

        # 分组稳定性评分 (0-100)
        avg_group_change = np.mean(all_group_changes) if all_group_changes else 0
        group_stability_score = max(0, 100 - avg_group_change * 20)

        # 综合稳定性评分
        overall_stability = (risk_stability_score + group_stability_score) / 2

        print(f"\n【总体稳定性评估】")
        print(f"风险稳定性评分: {risk_stability_score:.1f}/100")
        print(f"分组稳定性评分: {group_stability_score:.1f}/100")
        print(f"综合稳定性评分: {overall_stability:.1f}/100")

        # 稳定性等级
        if overall_stability >= 85:
            stability_level = "A级 (非常稳定)"
        elif overall_stability >= 70:
            stability_level = "B级 (较稳定)"
        elif overall_stability >= 55:
            stability_level = "C级 (中等稳定)"
        elif overall_stability >= 40:
            stability_level = "D级 (不够稳定)"
        else:
            stability_level = "E级 (很不稳定)"

        print(f"稳定性等级: {stability_level}")

        # 给出详细建议
        print(f"\n【详细建议】")
        if overall_stability >= 70:
            print("✓ 优化方案对数据扰动具有良好的鲁棒性")
            print("✓ 可以放心应用于临床实践")
            print("✓ 建议定期监控数据质量以维持性能")
        elif overall_stability >= 55:
            print("⚠ 优化方案具有中等稳定性")
            print("⚠ 建议在应用时注意以下事项:")
            print("  - 加强BMI和孕周测量的质量控制")
            print("  - 建立数据异常值检测机制")
            print("  - 定期重新评估和调整分组策略")
        else:
            print("❌ 优化方案稳定性较差，强烈建议:")
            print("  1. 提高数据测量精度，特别是BMI和孕周")
            print("  2. 增加样本量以提高模型稳定性")
            print("  3. 考虑使用更保守的分组策略")
            print("  4. 建立更严格的数据质量控制流程")

        return overall_stability

    def plot_detailed_results(self, save_path=None):
        """绘制详细的扰动分析结果"""

        if not self.analysis_results:
            print("❌ 没有可用的分析结果！")
            return

        # 确保使用微软雅黑字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('扰动影响分析详细结果', fontsize=16, fontweight='bold')

        # 提取数据
        bmi_stds = [r['bmi_std'] for r in self.analysis_results]
        time_stds = [r['time_std'] for r in self.analysis_results]
        risk_deviations = [r['risk_statistics']['relative_deviation'] for r in self.analysis_results]

        # 1. BMI扰动 vs 风险偏差
        ax = axes[0, 0]
        scatter = ax.scatter(bmi_stds, risk_deviations, c=time_stds, cmap='viridis',
                             alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('BMI 扰动标准差', fontfamily='Microsoft YaHei')
        ax.set_ylabel('风险相对偏差 (%)', fontfamily='Microsoft YaHei')
        ax.set_title('BMI扰动对风险的影响', fontfamily='Microsoft YaHei')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='时间扰动标准差')

        # 2. 时间扰动 vs 风险偏差
        ax = axes[0, 1]
        scatter2 = ax.scatter(time_stds, risk_deviations, c=bmi_stds, cmap='plasma',
                              alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('时间扰动标准差 (周)', fontfamily='Microsoft YaHei')
        ax.set_ylabel('风险相对偏差 (%)', fontfamily='Microsoft YaHei')
        ax.set_title('时间扰动对风险的影响', fontfamily='Microsoft YaHei')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.colorbar(scatter2, ax=ax, label='BMI扰动标准差')

        # 3. 风险偏差分布
        ax = axes[1, 0]
        ax.hist(risk_deviations, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='原始风险')
        ax.axvline(x=np.mean(risk_deviations), color='green', linestyle='--', linewidth=2, label='平均偏差')
        ax.set_xlabel('风险相对偏差 (%)', fontfamily='Microsoft YaHei')
        ax.set_ylabel('频次', fontfamily='Microsoft YaHei')
        ax.set_title('风险偏差分布', fontfamily='Microsoft YaHei')
        ax.legend(prop={'family': 'Microsoft YaHei'})
        ax.grid(True, alpha=0.3)

        # 4. 分组变化统计
        ax = axes[1, 1]
        all_changes = []
        for result in self.analysis_results:
            for change, freq in result['group_changes'].items():
                all_changes.extend([change] * freq)

        if all_changes:
            change_counts = {}
            for change in all_changes:
                change_counts[change] = change_counts.get(change, 0) + 1

            changes = list(change_counts.keys())
            counts = list(change_counts.values())

            colors = ['lightgreen' if c == 0 else 'lightcoral' for c in changes]
            bars = ax.bar([f'{c:+d}' for c in changes], counts, alpha=0.7, color=colors)
            ax.set_xlabel('分组数变化', fontfamily='Microsoft YaHei')
            ax.set_ylabel('频次', fontfamily='Microsoft YaHei')
            ax.set_title('分组数变化分布', fontfamily='Microsoft YaHei')
            ax.grid(True, alpha=0.3)

            # 在柱子上添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontfamily='Microsoft YaHei')
        else:
            ax.text(0.5, 0.5, '无分组变化数据', ha='center', va='center',
                    transform=ax.transAxes, fontfamily='Microsoft YaHei')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('perturbation_analysis_detailed.png', dpi=300, bbox_inches='tight')

        plt.show()


def run_perturbation_analysis_with_full_report(prediction_results_file='reach_time_results.csv'):
    """运行完整的扰动分析主函数"""
    try:
        print("读取数据...")
        data = pd.read_csv(prediction_results_file)
        print(f"数据加载完成，共 {len(data)} 个样本")

        # 创建分析器
        analyzer = SimplePerturbationImpactAnalyzer(data)

        # 运行分析
        results = analyzer.run_perturbation_analysis()

        if results:
            # 生成详细的稳定性报告
            analyzer.generate_stability_report()

            # 生成详细的可视化图表
            print("\n生成扰动分析图表...")
            save_filename = analyzer.plot_detailed_results()

            print(f"✓ 合并图表已保存到: {save_filename}")

            # 保存详细结果
            print("\n保存扰动分析结果...")

            # 创建详细汇总数据框
            summary_data = []
            for result in analyzer.analysis_results:
                summary_data.append({
                    'BMI扰动标准差': result['bmi_std'],
                    '时间扰动标准差': result['time_std'],
                    '成功运行次数': result['n_successful_runs'],
                    '原始风险': result['original_risk'],
                    '平均风险': result['risk_statistics']['mean'],
                    '风险标准差': result['risk_statistics']['std'],
                    '最小风险': result['risk_statistics']['min'],
                    '最大风险': result['risk_statistics']['max'],
                    '风险偏差': result['risk_statistics']['deviation_mean'],
                    '风险相对偏差百分比': result['risk_statistics']['relative_deviation'],
                    '原始分组数': result['original_groups'],
                    '分组变化统计': str(result['group_changes'])
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('perturbation_analysis_detailed_summary.csv', index=False, encoding='utf-8-sig')

            print("✓ 详细结果已保存到: perturbation_analysis_detailed_summary.csv")
            print("✓ 详细图表已保存到: perturbation_analysis_detailed.png")

        return analyzer

    except Exception as e:
        print(f"分析出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    analyzer = run_perturbation_analysis_with_full_report()