import numpy as np
import matplotlib.pyplot as plt

# 读取数据
file = np.load("\data\homework.npz")
X = file['X']
d = file['d']
# 观察输入矩阵形式
print("X Shape:{}; d Shape:{}".format(np.shape(X), np.shape(d)))
# 绘图
# plt.scatter(X[:, 0], d[:, 0])

def f(x, w):
    a, b, c = w
    return a*x**2 + b*x + c

def grad_f(x, d, w):
    y = f(x, w)
    grad_y = 2 * (y - d)
    grad_a = grad_y * x ** 2
    grad_b = grad_y * x
    grad_c = grad_y
    return grad_a, grad_b, grad_c

# 初始参数w
w = [0,0,0]
# 学习率
eta = 0.03
batchsize=10
for itr in range(250):
    sum_ga, sum_gb, sum_gc = 0, 0, 0
    for _ in range(batchsize):
        idx = np.random.randint(0, len(X))
        inx = X[idx]
        ind = d[idx]
        ga, gb, gc = grad_f(inx, ind, w)
        sum_ga += ga
        sum_ga += gb
        sum_gc += gc
    w[0] -= eta * sum_ga / batchsize
    w[1] -= eta * sum_gb / batchsize
    w[2] -= eta * sum_gc / batchsize

x = np.linspace(-2, 4, 100)
y = f(x, w)
# plt.scatter(X[:, 0], d[:, 0])
# plt.plot(x, y)
plt.scatter(X[:, 0], d[:, 0], color = 'g',s=20, alpha=0.4, label="数据散点")
plt.plot(x, y, lw=5, color="b", alpha=0.5, label="预测关系")
plt.legend()
plt.show()