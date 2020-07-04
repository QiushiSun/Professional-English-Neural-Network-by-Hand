import numpy as np
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# 导入损失函数，形式仿造了nndl中处理MINST的代码
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        #"""Return the cost associated with an output `a` and desired output`y`.
        return 0.5*np.linalg.norm(a-y)**2
    
    @staticmethod
    def delta(z, a, y):
        #Return the error delta from the output layer.
        return (a-y) * sigmoid_prime(z)

        #同样地，仿造nndl中处理MINST的代码引入交叉熵损失函数
class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        '''Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).'''
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

    #仿造nndl中的神经网络结构
class Network(object):
    #传入sizes参数构建神经网络的全连接层，默认使用交叉熵代价函数和默认权重初始化
    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        
        #需要实现对权重的初始化
        #这里使用两种权重初始化方式，默认权重初始化为均值为0，标准差为1的高斯分布随机分布。第二种权重初始化后均值为0，标准差为(1/n)^{1/2}，避免隐藏神经元饱和
    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        #实现前馈函数->用训练好的权重和偏置来计算网络输出（Test Dataset）
    def Feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    #接下来是神经网络的核心函数：随机梯度下降
    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            Lambda = 0.0, 
            evaluation_data=None,
            monitor_evaluation_accuracy=False):
        
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        
        #每个迭代期将训练集随机打乱，将数据集分成多个mini-batch
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, Lambda, len(training_data))
            print("Epoch %s" % j)
            
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))
                
        return evaluation_accuracy
    
    
    def update_mini_batch(self, mini_batch, eta, Lambda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(Lambda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    #反向传播算法计算权重和偏置
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    #判断网络输出和数据集相同的个数
    def accuracy(self, data):
        results = [(np.argmax(self.Feed_forward(x)), np.argmax(y))
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    def total_cost(self, data, Lambda):
        cost = 0.0
        for x, y in data:
            a = self.Feed_forward(x)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5*(Lambda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    #把神经网络的结构，权重，偏置和代价函数保存到json格式文件中
    """Save the neural network to the file ``filename``."""
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f) #写入json
        f.close()

        #### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def vectorized(i):
    e = np.zeros((2,1))
    e[i] = 1
    return e
#向量化


def data_loader():
    # preprocessing
    data = pd.read_csv('mushrooms.csv')
    encoder = preprocessing.LabelEncoder()
    for Colum in data.columns:
        data[Colum] = encoder.fit_transform(data[Colum])
    data = np.array(data)
    
    train_dataset, test_dataset = train_test_split(data, test_size = 0.25)#split test data and train data
    
    #process input and output and label
    train_output = [x[0] for x in train_dataset]
    train_in = np.array([x[1:] for x in train_dataset]).astype('float')
    test_out = [x[0] for x in test_dataset]
    test_in = np.array([x[1:] for x in test_dataset]).astype('float')
    
    
    # practice vectorization
    train_out_vec = [vectorized(y) for y in train_output]
    train_in_vec = [np.reshape(x, (22,1)) for x in train_in]
    
    test_out_vec = [vectorized(y) for y in test_out]
    test_in_vec = [np.reshape(x, (22,1)) for x in test_in]
    
    train_datas = list(zip(train_in_vec, train_out_vec))
    test_datas = list(zip(test_in_vec, test_out_vec))
    
    #divided datasets
    return train_datas, test_datas

# if __name__ == "__main__":
#     train_datas, test_datas = data_loader()
#     print(train_datas[0])