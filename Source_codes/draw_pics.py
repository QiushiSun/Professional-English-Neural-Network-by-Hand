import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号

# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号

def mkplot(epochs, ls1, ls2, label1, label2, title):
    
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(1,epochs+1,1), ls1, 'rx--', color = 'red', label = label1)
    ax.plot(np.arange(1,epochs+1,1), ls2, 'rx--', color = 'blue', label = label2)
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    
    plt.grid(linestyle='-.')
    plt.legend(loc = 'best')
    plt.savefig(title)
    
    plt.show()

# 对比1 分别使用交叉熵和二次代价函数
def Cost_Function_Comparison():
    
    net_cost_compare_quadratic = Network(layers, cost= QuadraticCost)
    net_cost_compare_crossentropy= Network(layers, cost= CrossEntropyCost)
    
    net_cost_compare_quadratic.large_weight_initializer()
    net_cost_compare_crossentropy.large_weight_initializer()
    
    evaluation_cost_function_accuracy1 = net_cost_compare_quadratic.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_cost_function_accuracy2 = net_cost_compare_crossentropy.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    
    #画图，用来对比交叉熵损失函数和二次损失函数
    mkplot(epochs,evaluation_cost_function_accuracy1, evaluation_cost_function_accuracy2, "二次代价函数", "交叉熵代价函数", "代价函数对准确度的影响")


    # 隐藏层调参数
    # 三组数据还算是有点代表性，但事实上训练量够了以后准确度都很高，亲测！
def Layers_Comparison():
    layer1 = [22,15,2]
    layer2 = [22,35,2]
    layer3 = [22,45,66,2]
    
    net_layer1 = Network(layer1, cost=mynetwork.QuadraticCost)
    net_layer2 = Network(layer2, cost=mynetwork.QuadraticCost)
    net_layer3 = Network(layer3, cost=mynetwork.QuadraticCost)
    
    net_layer1.large_weight_initializer()
    net_layer2.large_weight_initializer()
    net_layer3.large_weight_initializer()

    evaluation_layer_accuracy1 = net_layer1.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_layer_accuracy2 = net_layer2.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_layer_accuracy3 = net_layer3.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(1,epochs+1,1), evaluation_layer_accuracy1, 'rx--', color='steelblue', label = 'Layer:[22,15,2]')
    ax.plot(np.arange(1,epochs+1,1), evaluation_layer_accuracy2, 'rx--', color='blue', label = 'Layer:[22,35,2]')
    ax.plot(np.arange(1,epochs+1,1), evaluation_layer_accuracy3, 'rx--', color='yellow', label = 'Layer:[22,45,66,2]')
    
    #设置下图像参数
    ax.set_xlim([1,20])
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('不同隐藏层对比')
    
    plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号
    plt.grid(linestyle='-.')
    plt.legend(loc='best')
    #plt.savefig('CompareofLayers')
    
    plt.show()


#Mini_batch 对比
def Mini_Batch_Comparison():
    batch_size1 = 10
    batch_size2 = 20
    batch_size3 = 30
    batch_size4 = 40
    layers=[22,15,120,2]
    
    batch_test_net1 = Network(layers, cost=QuadraticCost)
    batch_test_net2 = Network(layers, cost=QuadraticCost)
    batch_test_net3 = Network(layers, cost=QuadraticCost)
    batch_test_net4 = Network(layers, cost=QuadraticCost)
    
    batch_test_net1.large_weight_initializer()
    batch_test_net2.large_weight_initializer()
    batch_test_net3.large_weight_initializer()
    batch_test_net4.large_weight_initializer()

    evaluation_batchsize_accuracy1 = batch_test_net1.SGD(train_datas, epochs, batch_size1, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_batchsize_accuracy2 = batch_test_net2.SGD(train_datas, epochs, batch_size2, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_batchsize_accuracy3 = batch_test_net3.SGD(train_datas, epochs, batch_size3, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_batchsize_accuracy4 = batch_test_net4.SGD(train_datas, epochs, batch_size4, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(1,epochs+1,1), evaluation_batchsize_accuracy1, 'rx--', color='red', label = 'MiniBatchSize=10')
    ax.plot(np.arange(1,epochs+1,1), evaluation_batchsize_accuracy2, 'rx--', color='blue', label = 'MiniBatchSize=20')
    ax.plot(np.arange(1,epochs+1,1), evaluation_batchsize_accuracy3, 'rx--', color='olive', label = 'MiniBatchSize=30')
    ax.plot(np.arange(1,epochs+1,1), evaluation_batchsize_accuracy4, 'rx--', color='green', label = 'MiniBatchSize=40')
    
    ax.set_xlim([1,25])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Batch大小对准确度的影响')
    
    plt.grid(linestyle='-.')
    plt.legend(loc='best')
    plt.savefig('CompareofMinibatch')
    plt.show()


def Reg_Analysis():
# 调试正则化参数

    lambda1 = 0.1
    lambda2 = 0.5
    lambda3 = 5
    lambda4 = 25
    lambda5 = 100
    
#     net_reg_1 = Network(layers, cost= CrossEntropyCost)
#     net_reg_2 = Network(layers, cost= CrossEntropyCost)
#     net_reg_3 = Network(layers, cost= CrossEntropyCost)
#     net_reg_4 = Network(layers, cost= CrossEntropyCost)
#     net_reg_5 = Network(layers, cost= CrossEntropyCost)
    
    net_reg_1 = Network(layers, cost= QuadraticCost)
    net_reg_2 = Network(layers, cost= QuadraticCost)
    net_reg_3 = Network(layers, cost= QuadraticCost)
    net_reg_4 = Network(layers, cost= QuadraticCost)
    net_reg_5 = Network(layers, cost= QuadraticCost)
    
    net_reg_1.large_weight_initializer()
    net_reg_2.large_weight_initializer()
    net_reg_3.large_weight_initializer()
    net_reg_4.large_weight_initializer()
    net_reg_5.large_weight_initializer()
    
    evaluation_reg_ccuracy1 = net_reg_1.SGD(train_datas, epochs, mini_batch, eta, lmbda = lambda1, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_reg_accuracy2 = net_reg_2.SGD(train_datas, epochs, mini_batch, eta, lmbda = lambda2, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_reg_accuracy3 = net_reg_3.SGD(train_datas, epochs, mini_batch, eta, lmbda = lambda3, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_reg_accuracy4 = net_reg_4.SGD(train_datas, epochs, mini_batch, eta, lmbda = lambda4, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_reg_accuracy5 = net_reg_4.SGD(train_datas, epochs, mini_batch, eta, lmbda = lambda5, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(1,epochs+1,1), evaluation_reg_accuracy1, 'x-', color='olive', label = r'$\lambda=0.05$')
    ax.plot(np.arange(1,epochs+1,1), evaluation_reg_accuracy2, 'x-', color='steelblue', label = r'$\lambda=0.5$')
    ax.plot(np.arange(1,epochs+1,1), evaluation_reg_accuracy3, 'x-', color='violet', label = r'$\lambda=5$')
    ax.plot(np.arange(1,epochs+1,1), evaluation_reg_accuracy4, 'x-', color='orange', label = r'$\lambda=25$')
    ax.plot(np.arange(1,epochs+1,1), evaluation_reg_accuracy5, 'x-', color='teal', label = r'$\lambda=100$')
    
    ax.set_xlim([1,20])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('正则化参数对准确率的影响')
    plt.grid(linestyle='-.')
    plt.legend(loc='best')
    
   # plt.savefig('Analysis Of lambda')
    plt.show()


# 学习速率对比

def Eta_analysis():
    eta1 = 0.1
    eta2 = 0.5
    eta3 = 0.8
    eta4 = 1.0
    
    net_eta_1 = Network(layers, cost= QuadraticCost)
    net_eta_1.large_weight_initializer()
    net_eta_2 = Network(layers, cost= QuadraticCost)
    net_eta_2.large_weight_initializer()
    net_eta_3 = Network(layers, cost= QuadraticCost)
    net_eta_3.large_weight_initializer()
    net_eta_4 = Network(layers, cost= QuadraticCost)
    net_eta_4.large_weight_initializer()

    evaluation_eta_accuracy1 = net_eta_1.SGD(train_datas, epochs, mini_batch, eta1, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_eta_accuracy2 = net_eta_2.SGD(train_datas, epochs, mini_batch, eta2, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_eta_accuracy3 = net_eta_3.SGD(train_datas, epochs, mini_batch, eta3, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    evaluation_eta_accuracy4 = net_eta_4.SGD(train_datas, epochs, mini_batch, eta4, evaluation_data = test_datas, monitor_evaluation_accuracy = True)
    
    fig, ax = plt.subplots()
    
    ax.plot(np.arange(1,epochs+1,1), evaluation_eta_accuracy1, 'x-', color='orange', label = 'Eta=0.1')
    ax.plot(np.arange(1,epochs+1,1), evaluation_eta_accuracy2, 'x-', color='red', label = 'Eta=0.5')
    ax.plot(np.arange(1,epochs+1,1), evaluation_eta_accuracy3, 'x-', color='dodgerblue', label = 'Eta=0.8')
    ax.plot(np.arange(1,epochs+1,1), evaluation_eta_accuracy4, 'x-', color='teal', label = 'Eta=1.0')
   
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Eta对准确度的影响')
    ax.set_xlim([1,20])
        
    plt.legend(loc='best')
    plt.grid(linestyle='-.')
    
    plt.show()


# 最优神经网络的参数

# layers = [22,65,100,2]
# epochs = 30
# mini_batch = 10
# eta = 0.5
# lmbda = 0.05

# best_fit_paras = Network(layers, cost=QuadraticCost)
# best_fit_paras.large_weight_initializer()#先要初始化
# test_of_accuracy_one = best_fit_paras.SGD(train_datas, epochs, mini_batch, eta, evaluation_data = test_datas, monitor_evaluation_accuracy = True)