我们将使用 **Python** 和 **scikit-learn** 进行线性回归建模，分析 TV、Radio、Newspaper 投放对收益的影响，并预测新的广告投放数据下的收益。  

---

### **1. 加载数据**
假设数据存储在 CSV 文件 `Advertising.csv` 中，我们首先加载数据：
```python
import pandas as pd

# 读取数据
df = pd.read_csv("Advertising.csv")

# 查看前5行
print(df.head())
```

数据应包含以下列：
- `TV`: 在电视广告上的投放金额
- `Radio`: 在广播广告上的投放金额
- `Newspaper`: 在报纸广告上的投放金额
- `Sales`: 投放后带来的收益（目标变量）

---

### **2. 数据可视化**
在建模前，我们先观察数据特征与 `Sales` 之间的关系：
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 画出 TV、Radio、Newspaper 与 Sales 的散点图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.scatterplot(x=df["TV"], y=df["Sales"], ax=axes[0])
axes[0].set_title("TV vs Sales")

sns.scatterplot(x=df["Radio"], y=df["Sales"], ax=axes[1])
axes[1].set_title("Radio vs Sales")

sns.scatterplot(x=df["Newspaper"], y=df["Sales"], ax=axes[2])
axes[2].set_title("Newspaper vs Sales")

plt.show()
```
从散点图可以观察到 TV、Radio 可能与 `Sales` 有较强的线性关系，而 Newspaper 可能影响较小。

---

### **3. 线性回归建模**
我们使用 **多元线性回归** 来拟合模型：
\[
Sales = \theta_0 + \theta_1 \times TV + \theta_2 \times Radio + \theta_3 \times Newspaper
\]
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 特征(X) 和 目标变量(y)
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 模型系数
print("回归系数:", model.coef_)
print("截距:", model.intercept_)
```
如果模型系数是：
```text
回归系数: [0.045, 0.187, 0.001]
截距: 2.93
```
那么得到的线性回归方程为：
\[
Sales = 2.93 + 0.045 \times TV + 0.187 \times Radio + 0.001 \times Newspaper
\]

---

### **4. 模型评估**
```python
# 预测
y_pred = model.predict(X_test)

# 计算均方误差 (MSE) 和 R² 分数
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("均方误差 (MSE):", mse)
print("R² 分数:", r2)
```
- **MSE 越小越好**，表示预测误差越小。
- **R² 值接近 1**，表示模型拟合较好。

---

### **5. 预测新的广告投放数据**
假设有新的广告投放：
```python
new_data = pd.DataFrame({
    "TV": [200, 150],
    "Radio": [30, 20],
    "Newspaper": [50, 10]
})

# 预测 Sales
new_sales_pred = model.predict(new_data)
print("预测的 Sales:", new_sales_pred)
```
如果输出：
```text
预测的 Sales: [17.5, 14.8]
```
表示在 `TV=200, Radio=30, Newspaper=50` 投放时，预计收益 `17.5`。

---

### **6. 结论**
1. **TV 和 Radio 对 Sales 影响较大**，Newspaper 影响可能较小。
2. **模型的 MSE 和 R² 评分可用于评估拟合优度**。
3. **可以使用模型预测新的广告投放带来的收益**。
