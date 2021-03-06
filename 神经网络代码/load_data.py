import pickle
import numpy as np
import gzip


def load_data():
    """从文件中加载数据特征x， 标签y"""
    f = gzip.open(r'mnist.pkl.gz', mode='rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """将特征x和标签y绑定在一起"""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return list(training_data), list(validation_data), list(test_data)


def vectorized_result(j):
    """将x转换为向量类型的形式"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
