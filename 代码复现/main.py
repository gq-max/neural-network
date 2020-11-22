# coding=gbk
import load_data
import network

# 读取训练、测试和验证数据
training_data, validation_data, test_data = load_data.load_data_wrapper()
net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
net.large_weight_initializer()
x = net.SGD(training_data, 30, 10, 0.5,
            evaluation_data=test_data, monitor_evaluation_accuracy=True)
print(x)
