import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
data = np.random.rand(10, 12)

# 创建热力图
sns.heatmap(data, annot=True, cmap='viridis')

# 显示图形
plt.show()