import numpy as np
import matplotlib.pyplot as plt

x_scatter = np.arange(5, 15, 1)
y_scatter = np.linspace(0.08, 0.18, 11)
# Visualize validation accuracies
# results[(lr, reg)] = (train_acc, val_acc)
# best_point = (math.log10(lr), math.log10(reg))  # for labeling
results = {}
for x in x_scatter:
    for y in y_scatter:
        results[(x, y)] = x / y
# x_scatter = [math.log10(x[0]) for x in results]
# y_scatter = [math.log10(x[1]) for x in results]
marker_size = 100  # default: 20
best_point = max(results, key=results.get)
best_acc = max(results.values())
worst_acc = min(results.values())
colors = [results[x] for x in results.keys()]
colors = np.array(colors)
colors = 1-((best_acc - colors)/(best_acc - worst_acc))
[Y, X] = np.meshgrid(y_scatter, x_scatter)

X = X.reshape(len(X.reshape(-1, 1)))
Y = Y.reshape(len(Y.reshape(-1, 1)))
print(colors)
print(X)
print(Y)
plt.figure()
plt.scatter(X, Y, marker_size, c=colors, cmap='coolwarm')
plt.annotate('(%.2f,%.2f,%.2f%%)'% (best_point[0],best_point[1],best_acc),
             xy=best_point, xytext=(-30, 30), textcoords='offset pixels',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.colorbar()
plt.xlabel('10^lr')
plt.ylabel('10^reg')
plt.title('CIFAR-10 validation accuracy')
plt.show()
