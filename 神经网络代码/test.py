# coding=gbk
import load_data
import network
import matplotlib.pyplot as plt
import numpy as np

# training_data, validation_data, test_data = load_data.load_data_wrapper()
# training_data = training_data[0:5000]
# validation_data = validation_data[0:1000]
# test_data = test_data[0:1000]
# net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
# y = net.SGD(training_data, 30, 9, 0.05, lambda1=2,
#             evaluation_data=test_data,
#             monitor_training_cost=True,
#             monitor_evaluation_cost=True)
# x = np.arange(30)
# y1 = [i for i in y[0]]
# y2 = [i for i in y[2]]
# plt.plot(x, y1, 'g', label='evaluation_cost')
# plt.plot(x, y2, 'b', label='training_cost')
# plt.legend()
# plt.show()


# 网格搜索大致好的超参数
# x_scatter = np.arange(8, 15, 1)
# y_scatter = np.linspace(0.08, 0.13, 11)
# results = {}
# x_scatter = np.arange(5, 7, 1)
# y_scatter = np.linspace(0.08, 0.10, 3)
# results = {}

# for x in x_scatter:
#     for y in y_scatter:
#         training_data, validation_data, test_data = load_data.load_data_wrapper()
#         training_data = training_data[0:5000]
#         test_data = test_data[0:1000]
#         net = network.Network([784, 15, 10], cost=network.CrossEntropyCost)
#         res = net.SGD(training_data, 10, x, y,
#                       evaluation_data=test_data,
#                       monitor_training_accuracy=True,
#                       monitor_evaluation_accuracy=True)
#         accuracy = max([i / 1000 for i in res[1]])
#         results[(x, y)] = accuracy
# marker_size = 100  # default: 20
# best_point = max(results, key=results.get)
# best_acc = max(results.values())
# worst_acc = min(results.values())
# colors = [results[x] for x in results.keys()]
# colors = np.array(colors)
# colors = 1-((best_acc - colors)/(best_acc - worst_acc))
# [Y, X] = np.meshgrid(y_scatter, x_scatter)
#
# X = X.reshape(len(X.reshape(-1, 1)))
# Y = Y.reshape(len(Y.reshape(-1, 1)))
# plt.figure()
# plt.scatter(X, Y, marker_size, c=colors, cmap=plt.cm.coolwarm)
# plt.annotate('(%.2f,%.2f,%.2f%%)' % (best_point[0], best_point[1], best_acc*100),
#              xy=best_point, xytext=(-30, 30), textcoords='offset pixels',
#              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
# plt.colorbar()
# plt.xlabel('mini_batch_size')
# plt.ylabel('eta')
# plt.title('search best parameter')
# plt.show()


training_data, validation_data, test_data = load_data.load_data_wrapper()
net = network.Network([784, 25, 10], cost=network.CrossEntropyCost)
y = net.SGD(training_data, 30, 12, 0.08, lambda1=15,
            evaluation_data=test_data,
            monitor_training_cost=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_evaluation_accuracy=True)
x = np.arange(30)
y0 = [i for i in y[0]]
y1 = [i/10000 for i in y[1]]
y2 = [i for i in y[2]]
y3 = [i/50000 for i in y[3]]
plt.subplot(1, 2, 1)
plt.plot(x, y0, 'g', label='evaluation_cost')
plt.plot(x, y2, 'b', label='training_cost')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, y1, 'r', label='evaluation_acc')
plt.plot(x, y3, 'y', label='training_acc')
plt.legend()
plt.show()
