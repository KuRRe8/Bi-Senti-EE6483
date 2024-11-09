from skelm import ELMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建一个Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('elm', ELMClassifier())       # ELM分类器
])

# 定义参数网格
param_grid = {
    'elm__alpha': [0.00000002, 0.0000002, 0.000002, 0.00001, 0.0001, 0.001, 0.01, 0.1],
}

# 定义 GridSearchCV，并设置 n_jobs=-1 以使用所有可用的CPU核心
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

# 拟合模型
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳评分
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)

# 在测试集上评估模型
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test accuracy score: ", test_score)