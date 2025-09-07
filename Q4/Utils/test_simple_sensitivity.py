"""
测试详细参数敏感性分析 - 每个参数只做正向10%扰动
"""

import sys
import os
import pandas as pd

# 将当前目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 将Q4目录添加到Python路径  
q4_dir = os.path.join(current_dir, 'Q4')
if q4_dir not in sys.path:
    sys.path.insert(0, q4_dir)

def test_detailed_sensitivity():
    """测试详细敏感性分析"""
    try:
        from sensitivity_analysis import ModelSensitivityAnalysis
        print("✅ 成功导入模块")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return
    
    print("🔍 开始详细参数敏感性分析测试")
    print("="*60)
    
    # 创建分析器
    analyzer = ModelSensitivityAnalysis()
    
    try:
        # 1. 加载模型
        if not analyzer.load_model_components():
            return
        
        # 2. 加载和准备数据
        X, processed_data = analyzer.load_and_prepare_data('Q4/clean_girls_data_Q4.csv')
        
        # 3. 运行详细敏感性分析 (只做正向30%扰动)
        print("\n开始详细敏感性分析...")
        detailed_df = analyzer.detailed_parameter_sensitivity_analysis(X, perturbation_ratio=0.3)
        
        if detailed_df is not None:
            print(f"\n🎉 分析完成！生成了 {len(detailed_df)} 条记录")
            
            # 保存结果
            output_file = 'Q4/Results/detailed_parameter_sensitivity_analysis.csv'
            os.makedirs('Q4/Results', exist_ok=True)
            detailed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"📁 结果已保存到: {output_file}")
            
            # 显示前10个最敏感的参数
            print("\n📊 前10个最敏感的参数:")
            print("-" * 80)
            
            # 转换敏感性数值为float用于排序
            detailed_df['敏感性数值_float'] = detailed_df['敏感性数值'].astype(float)
            top_10 = detailed_df.nlargest(10, '敏感性数值_float')
            
            # 创建格式化的显示表格
            for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                print(f"{i:2d}. {row['参数名称']:<25} | {row['参数类型']:<10} | "
                      f"{row['变化幅度']:<6} | {row['基准值']:<10} | {row['扰动后值']:<10} | "
                      f"{row['目标值变化']:<12} | {row['敏感性等级']:<6}")
            
            # 统计敏感性等级分布
            print(f"\n📈 敏感性等级分布:")
            level_counts = detailed_df['敏感性等级'].value_counts()
            for level, count in level_counts.items():
                percentage = (count / len(detailed_df)) * 100
                print(f"  {level}: {count} 个参数 ({percentage:.1f}%)")
            
            # 统计参数类型分布
            print(f"\n🏷️  参数类型分布:")
            type_counts = detailed_df['参数类型'].value_counts()
            for param_type, count in type_counts.items():
                percentage = (count / len(detailed_df)) * 100
                print(f"  {param_type}: {count} 个参数 ({percentage:.1f}%)")
            
            # 按敏感性等级分组显示
            print(f"\n🔍 按敏感性等级分组的参数:")
            for level in ['高敏感', '中敏感', '低敏感', '不敏感']:
                level_params = detailed_df[detailed_df['敏感性等级'] == level]
                if len(level_params) > 0:
                    print(f"\n{level} ({len(level_params)}个):")
                    for _, row in level_params.head(5).iterrows():  # 只显示前5个
                        print(f"  • {row['参数名称']} (目标值变化: {row['目标值变化']})")
                    if len(level_params) > 5:
                        print(f"  ... 还有 {len(level_params) - 5} 个参数")
            
            return detailed_df
            
        else:
            print("❌ 详细敏感性分析失败")
            return None
            
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_detailed_sensitivity()
    if result is not None:
        print(f"\n🎉 测试成功完成！共分析了 {len(result)} 个参数")
        print("📋 生成的表格包含以下列:")
        print("  - 参数名称: 特征的名称")
        print("  - 参数类型: 特征的分类(Z值指标、临床指标等)")
        print("  - 变化幅度: 统一为+10%")
        print("  - 基准值: 原始参数的平均值")
        print("  - 扰动后值: 加上扰动后的值")
        print("  - 目标值变化: 模型预测概率的变化量")
        print("  - 敏感性等级: 高敏感/中敏感/低敏感/不敏感")
    else:
        print("💥 测试失败")
