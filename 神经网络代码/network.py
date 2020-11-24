import json
import sys
import numpy as np


# 交叉熵损失函数
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """交叉熵损失函数"""

        return np.sum(np.nan_to_num(-y * np.log(a)))

    @staticmethod
    def delta(z, a, y):
        """BP1:输出层误差"""
        return a - y


def ReLU(z):
    """ReLU函数"""
    return np.where(z < 0, 0, z)


def ReLU_prime(z):
    """ReLU函数的导数"""
    return np.where(z < 0, 0, 1)


def vectorized_result(j):
    """将x转换为向量类型的形式"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# 神经网络类
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        '''
        :param sizes: 以列表形式表示网络，如[5, 10, 3]表示三层网络
        :param cost: 使用哪种损失函数，默认采用交叉熵损失
        '''
        # 权重初始化
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """使用高斯分布随机初始化"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """前向传播"""
        for b, w in zip(self.biases, self.weights):
            a = ReLU(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lambda1=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        '''
        采用小批量梯度下降训练神经网络，并采用正则化
        :param training_data: 训练数据
        :param epochs: 迭代周期
        :param mini_batch_size: 每次采用样本个数
        :param eta:
        :param lambda1: 正则化参数
        :param evaluation_data: 评估数据
        :param monitor_evaluation_cost:
        :param monitor_evaluation_accuracy:
        :param monitor_training_cost:
        :param monitor_training_accuracy:
        :return:
        '''

        if evaluation_data:
            n_evaluation_data = len(evaluation_data)
        n_train_data = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            np.random.shuffle(training_data)  # 随机打乱数据
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n_train_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lambda1, len(training_data))
            print("第 %s 次迭代" % j)
            # 训练误差和精度
            if monitor_training_cost:
                cost = self.total_cost(training_data, lambda1)
                training_cost.append(cost)
                print("测试集的损失函数: {}".format(cost[0]))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("在测试集上的精度: {} / {}".format(accuracy, n_train_data))
            # 验证误差和精度
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lambda1, convert=True)
                evaluation_cost.append(cost)
                print("验证集的损失函数: {}".format(cost[0]))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("验证集的精度: {} / {}".format(self.accuracy(evaluation_data), n_evaluation_data))
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lambda1, n):
        '''
        通过梯度下降更新神经网络的权重和偏置
        :param mini_batch: 一个列表，每个元素为元组(x, y)
        :param eta: 学习率
        :param lambda1: 正则化参数
        :param n: 训练集的个数
        :return:
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lambda1 / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        反向传播
        :param x: 输入
        :param y: 输出
        :return: 元组(nabla_b, nabla_w)表示梯度
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # 存储每层的激活函数
        zs = []  # 存储每层的z
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = ReLU(z)
            activations.append(activation)
        # 反向传播
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = ReLU_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        '''
        返回精度
        :param data: 数据集
        :param convert: True表示训练集，False表示测试集或验证集
        :return:
        '''
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lambda1, convert=False):
        '''
        计算损失大大小
        :param data: 数据集
        :param lambda1: 正则化参数
        :param convert: 表示训练集还是测试集
        :return:
        '''
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lambda1 / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """将结果保存到filename中"""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
