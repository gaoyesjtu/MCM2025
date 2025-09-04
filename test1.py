import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(0, 2*np.pi, 200)
plt.plot(x, np.sin(x))
plt.title("中文标题测试：正弦曲线")
plt.xlabel("横轴：x")
plt.ylabel("纵轴：sin(x)")
plt.grid(alpha=0.3)
plt.show()
