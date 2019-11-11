import random
import scipy.stats as stats
import math
import numpy as np
from sklearn import datasets
from numpy import *
import matplotlib.pyplot as plt
#1相关系数   方差计算是除以n-1  还是n
class Related_analysis():
    #求均值
    def mean(self,x):
          return sum(x) / len(x)
    # 计算每一项数据与均值的差
    def de_mean(self,x):
      x_bar = self.mean(x)
      return [x_i - x_bar for x_i in x]
    # 辅助计算函数 dot product 、sum_of_squares
    def dot(self,v, w):
      return sum(v_i * w_i for v_i, w_i in zip(v, w))
    def sum_of_squares(self,v):
      return self.dot(v, v)
    # 方差
    def variance(self,x):
      n = len(x)
      deviations = self.de_mean(x)
      return self.sum_of_squares(deviations) / (n - 1)
    # 标准差
    import math
    def standard_deviation(self,x):
      return math.sqrt(self.variance(x))

    # 协方差
    def covariance(self,x, y):
      n = len(x)
      return self.dot(self.de_mean(x), self.de_mean(y)) / (n -1)

    # 相关系数
    def correlation(self,x, y):
      stdev_x = self.standard_deviation(x)
      stdev_y = self.standard_deviation(y)
      if stdev_x > 0 and stdev_y > 0:
        return self.covariance(x, y) / stdev_x / stdev_y
      else:
        return 0

    #   相关系数矩阵
    def corrlation_matrix(self,v):
        corrlation_matrix=[]
        corrlation_matrix_row=[]
        for i in v:
            for j in v:
                corrlation_matrix_row.append(self.correlation(i,j))
            corrlation_matrix.append(corrlation_matrix_row)
            corrlation_matrix_row=[]
        return corrlation_matrix
    #2秩相关系数    当有相同的数时如何解决？
    def rank_correlation(self,a,b):
        n=len(a)
        c=stats.rankdata(a,method='average')
        d=stats.rankdata(b,method='average')
        diffs=c-d
        r_s=1-6*sum(diffs*diffs)/(n*(n**2-1))
        return r_s
    #偏相关系数  仅仅计算一阶偏相关系数 3个变量
    def partial_correlation(self,a,b,c):
        r_ab=self.correlation(a,b)
        r_ac=self.correlation(a,c)
        r_bc=self.correlation(b,c)
        r_ab_c = (r_ab - r_ac * r_bc) / (((1 - r_ac ** 2) ** 0.5) * ((1 - r_bc ** 2) ** 0.5))
        return r_ab_c

    #复相关系数   仅仅能计算有两个自变量时的情形
    def complex_correlation(self,y,x1,x2):
        r_y1=self.correlation(y,x1)
        r_y2_1=self.partial_correlation(y,x2,x1)
        r_y_12=math.sqrt(1-(1-r_y1**2)*(1-r_y2_1**2))
        return r_y_12

class Regression_analysis():
    def unary_linear_regression(self,Dependent,Independent):
        related_analysis=Related_analysis()
        y_mean=related_analysis.mean(Dependent)
        x_mean=related_analysis.mean(Independent)
        stdev_x=related_analysis.standard_deviation(Independent)
        b=related_analysis.covariance(Dependent, Independent) / stdev_x/stdev_x
        a=y_mean-b*x_mean
        return a,b
    def Multiline_linear_regression(self,Dependent,Independent):
        #将矩阵转化为np数组
        #构造矩阵
        list1=[]
        n=len(Dependent)
        for i in range(n):
            list1.append(1)
        x=[]
        x.append(list1)
        for i in Independent:
            x.append(i)
        X=np.array(x)
        X_Transpose=X.T
        Y=np.array(Dependent)
        b=np.dot(np.dot(np.linalg.inv(np.dot(X,X_Transpose)),X),Y)
        return b

class System_clustering():
    def totalization_standardization(self,variable):
        sum_stand=[]
        sum_stand_variable=[]
        for i in variable:
            sumi=sum(i)
            for j in  i:
                sum_stand_variable.append(j/sumi)
            sum_stand.append(sum_stand_variable)
            sum_stand_variable=[]
        return sum_stand
    def standard_deviation_standardization(self,variable):
        dev_stand = []
        dev_stand_variable = []
        related=Related_analysis()
        for i in variable:
            average=related.mean(i)
            stddev=related.standard_deviation(i)
            for j in i:
                dev_stand_variable.append((j -average)/stddev)
            dev_stand.append(dev_stand_variable)
            dev_stand_variable=[]
        return dev_stand
    def max_standardization(self,variable):
        max_stand=[]
        max_stand_variable = []
        for i in variable:
            maxi=max(i)
            for j in i:
                max_stand_variable.append(j /maxi)
            max_stand.append(max_stand_variable)
            max_stand_variable = []
        return max_stand
    def extremely_poor_standardization(self,variable):
        extremely_poor_stand = []
        extremely_poor_stand_variable = []
        for i in variable:
            maxi = max(i)
            mini=min(i)
            for j in i:
                extremely_poor_stand_variable.append((j-mini)/(maxi-mini))
            extremely_poor_stand.append(extremely_poor_stand_variable)
            extremely_poor_stand_variable = []
        return extremely_poor_stand
    def caculate_absolute_distance(self,variable):
        absolute_distance=[]
        distance_list=[]
        distance=[]
        for a in variable:
            for b in variable:
                for i in range(len(a)):
                    distance.append(abs(a[i] - b[i]))
                distance_list.append(sum(distance))
                distance=[]
            absolute_distance.append(distance_list)
            distance_list=[]
        return absolute_distance
    def euclidean_distance(self,vector1,vector2):
        return sqrt((vector1-vector2)*(vector1-vector2).T)
    def direct_clustering(self,cluster_object):
        distance_matrix=np.array(self.caculate_absolute_distance(cluster_object))
        cluster_result=[]  #聚类结果  一开始各自为一类  0到n-1
        for i in range(len(cluster_object)):
            cluster_result.append([i])
        #需找矩阵中最小的类
        zeros =[]
        zero=[]
        for i in range(len(cluster_object)):
            zero.append(0)
        for i in range(len(cluster_object)):
            zeros.append(zero)
        zeros=np.array(zeros)
        n=len(distance_matrix)
        while(n>1):
            min_distance=np.where(distance_matrix == np.max(distance_matrix))
            min_distance=min_distance[0]
            arr=[]
            for j in cluster_result:
                if min_distance[0] in j:
                    arr.extend(j)
                    cluster_result.remove(j)
                    break

            for j in cluster_result:
                if min_distance[1] in j:
                    arr.extend(j)
                    cluster_result.remove(j)
                    break

            cluster_result.append(arr)
            print("当前聚类结果是",cluster_result)
            distance_matrix[min_distance[1]]=zero
            distance_matrix[:,min_distance[1]]=zeros[:,1]
            n=n-1

class Kmeans():
    ##随机挑选一个数据点作为种子点
    def select_seed(self,Xn):
        idx = np.random.choice(range(len(Xn)))
        return idx

    ##计算数据点到种子点的距离
    def cal_dis(self,Xn, Yn, idx):
        dis_list = []
        for i in range(len(Xn)):
            d = np.sqrt((Xn[i] - Xn[idx]) ** 2 + (Yn[i] - Yn[idx]) ** 2)
            dis_list.append(d)
        return dis_list

    ##随机挑选另外的种子点
    def select_seed_other(self,Xn, Yn, dis_list):
        d_sum = sum(dis_list)
        rom = d_sum * np.random.random()
        idx = 0
        for i in range(len(Xn)):
            rom -= dis_list[i]
            if rom > 0:
                continue
            else:
                idx = i
        return idx

    ##选取所有种子点
    def select_seed_all(self,seed_count):
        ##种子点
        Xk = []  ##种子点x轴列表
        Yk = []  ##种子点y轴列表

        idx = 0  ##选取的种子点的索引
        dis_list = []  ##距离列表

        ##选取种子点
        # 因为实验数据少，有一定的几率选到同一个数据，所以加一个判断
        idx_list = []
        flag = True
        for i in range(seed_count):
            if i == 0:
                idx = self.select_seed(Xn)
                dis_list = self.cal_dis(Xn, Yn, idx)
                Xk.append(Xn[idx])
                Yk.append(Yn[idx])
                idx_list.append(idx)
            else:
                while flag:
                    idx = self.select_seed_other(Xn, Yn, dis_list)
                    if idx not in idx_list:
                        flag = False
                    else:
                        continue
                dis_list = self.cal_dis(Xn, Yn, idx)
                Xk.append(Xn[idx])
                Yk.append(Yn[idx])
                idx_list.append(idx)

        ##列表转成数组
        Xk = np.array(Xk)
        Yk = np.array(Yk)

        return Xk, Yk

    def start_class(self,Xk, Yk):
        ##数据点分类
        cls_dict = {}
        ##离哪个分类点最近，属于哪个分类
        for i in range(len(Xn)):
            temp = []
            for j in range(len(Xk)):
                d1 = np.sqrt((Xn[i] - Xk[j]) * (Xn[i] - Xk[j]) + (Yn[i] - Yk[j]) * (Yn[i] - Yk[j]))
                temp.append(d1)
            min_dis = np.min(temp)
            min_inx = temp.index(min_dis)
            cls_dict[sign_n[i]] = sign_k[min_inx]
        # print(cls_dict)
        return cls_dict

    ##重新计算分类的坐标点
    def recal_class_point(self,Xk, Yk, cls_dict):
        num_k1 = 0  # 属于k1的数据点的个数
        num_k2 = 0  # 属于k2的数据点的个数
        x1 = 0  # 属于k1的x坐标和
        y1 = 0  # 属于k1的y坐标和
        x2 = 0  # 属于k2的x坐标和
        y2 = 0  # 属于k2的y坐标和

        ##循环读取已经分类的数据
        for d in cls_dict:
            ##读取d的类别
            kk = cls_dict[d]
            if kk == 'k1':
                # 读取d在数据集中的索引
                idx = sign_n.index(d)
                ##累加x值
                x1 += Xn[idx]
                ##累加y值
                y1 += Yn[idx]
                ##累加分类个数
                num_k1 += 1
            else:
                # 读取d在数据集中的索引
                idx = sign_n.index(d)
                ##累加x值
                x2 += Xn[idx]
                ##累加y值
                y2 += Yn[idx]
                ##累加分类个数
                num_k2 += 1
        ##求平均值获取新的分类坐标点
        k1_new_x = x1 / num_k1  # 新的k1的x坐标
        k1_new_y = y1 / num_k1  # 新的k1的y坐标

        k2_new_x = x2 / num_k2  # 新的k2的x坐标
        k2_new_y = y2 / num_k2  # 新的k2的y坐标

        ##新的分类数组
        Xk = np.array([k1_new_x, k2_new_x])
        Yk = np.array([k1_new_y, k2_new_y])
        return Xk, Yk

    def draw_point(self,Xk, Yk, cls_dict):
        # 画样本点
        plt.figure(figsize=(5, 4))
        plt.scatter(Xn, Yn, color="green", label="数据", linewidth=1)
        plt.scatter(Xk, Yk, color="red", label="分类", linewidth=1)
        plt.xticks(range(1, 6))
        plt.xlim([1, 5])
        plt.ylim([1, 6])
        plt.legend()
        for i in range(len(Xn)):
            plt.text(Xn[i], Yn[i], sign_n[i] + ":" + cls_dict[sign_n[i]])
            for i in range(len(Xk)):
                plt.text(Xk[i], Yk[i], sign_k[i])
        plt.show()

    def draw_point_all_seed(self,Xk, Yk):
        # 画样本点
        plt.figure(figsize=(5, 4))
        plt.scatter(Xn, Yn, color="green", label="数据", linewidth=1)
        plt.scatter(Xk, Yk, color="red", label="分类", linewidth=1)
        plt.xticks(range(1, 6))
        plt.xlim([1, 5])
        plt.ylim([1, 6])
        plt.legend()
        for i in range(len(Xn)):
            plt.text(Xn[i], Yn[i], sign_n[i])
        plt.show()

    pass

class PCA():

    """
    主成份分析算法PCA，非监督学习算法.
    """
    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None
        self.k = 2

    def shuffle_data(self,X, y, seed=None):
        if seed:
            np.random.seed(seed)

        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)

        return X[idx], y[idx]

    # 正规化数据集 X
    def normalize(self,X, axis=-1, p=2):
        lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
        lp_norm[lp_norm == 0] = 1
        return X / np.expand_dims(lp_norm, axis)

    # 标准化数据集 X
    def standardize(self,X):
        X_std = np.zeros(X.shape)
        mean = X.mean(axis=0)
        std = X.std(axis=0)

        # 做除法运算时请永远记住分母不能等于0的情形
        # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        for col in range(np.shape(X)[1]):
            if std[col]:
                X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]

        return X_std

    # 划分数据集为训练集和测试集
    def train_test_split(self,X, y, test_size=0.2, shuffle=True, seed=None):
        if shuffle:
            X, y = self.shuffle_data(X, y, seed)

        n_train_samples = int(X.shape[0] * (1 - test_size))
        x_train, x_test = X[:n_train_samples], X[n_train_samples:]
        y_train, y_test = y[:n_train_samples], y[n_train_samples:]

        return x_train, x_test, y_train, y_test

    # 计算矩阵X的协方差矩阵
    def calculate_covariance_matrix(self,X, Y=np.empty((0, 0))):
        if not Y.any():
            Y = X
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
        return np.array(covariance_matrix, dtype=float)

    # 计算数据集X每列的方差
    def calculate_variance(self,X):
        n_samples = np.shape(X)[0]
        variance = (1 / n_samples) * np.diag((X - X.mean(axis=0)).T.dot(X - X.mean(axis=0)))
        return variance

    # 计算数据集X每列的标准差
    def calculate_std_dev(self,X):
        std_dev = np.sqrt(self.calculate_variance(X))
        return std_dev

    # 计算相关系数矩阵
    def calculate_correlation_matrix(self,X, Y=np.empty([0])):
        # 先计算协方差矩阵
        covariance_matrix = self.calculate_covariance_matrix(X, Y)
        # 计算X, Y的标准差
        std_dev_X = np.expand_dims(self.calculate_std_dev(X), 1)
        std_dev_y = np.expand_dims(self.calculate_std_dev(Y), 1)
        correlation_matrix = np.divide(covariance_matrix, std_dev_X.dot(std_dev_y.T))

        return np.array(correlation_matrix, dtype=float)

    def transform(self, X):
        """ 
        将原始数据集X通过PCA进行降维
        """
        covariance = self.calculate_covariance_matrix(X)

        # 求解特征值和特征向量
        self.eigen_values, self.eigen_vectors = np.linalg.eig(covariance)

        # 将特征值从大到小进行排序，注意特征向量是按列排的，即self.eigen_vectors第k列是self.eigen_values中第k个特征值对应的特征向量
        idx = self.eigen_values.argsort()[::-1]
        eigenvalues = self.eigen_values[idx][:self.k]
        eigenvectors = self.eigen_vectors[:, idx][:, :self.k]

        # 将原始数据集X映射到低维空间
        X_transformed = X.dot(eigenvectors)

        return X_transformed

    def main(self):
        # Load the dataset
        data = datasets.load_iris()
        X = data.data
        y = data.target

        # 将数据集X映射到低维空间
        X_trans = PCA().transform(X)

        x1 = X_trans[:, 0]
        x2 = X_trans[:, 1]

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]

        class_distr = []
        # Plot the different class distributions
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        # Add a legend
        plt.legend(class_distr, y, loc=1)

        # Axis labels
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

class markov_chain():
    def markov_chain_show(self):
        # 转移矩阵    具体问题中一般采用频率近似概率的方式去计算
        trans_matrix = np.array(
            [[0.5, 0.1, 0.25, 0.05],
             [0.15, 0.5, 0.2, 0.05],
             [0.1, 0.2, 0.5, 0.4],
             [0.25, 0.2, 0.05, 0.5]])
        # 数据
        dt = np.array([[0], [1], [0], [0]])
        # 进行转换

        res = np.dot(trans_matrix, dt)
        for i in range(50):
            res = np.dot(trans_matrix, res)  # 50次
        #输出最终稳定的概率
        print(res)

if __name__ == "__main__":
    #数据准备

    # random.seed(12345)
    # a = [random.randint(0, 100,) for a in range(20)]
    # b = [random.randint(0, 100) for a in range(20)]
    # c = [random.randint(0, 100) for a in range(20)]
    # d = [random.randint(0, 100) for a in range(20)]
    # e = [random.randint(0, 100) for a in range(20)]
    # v=[a,b,c,d,e]
    # test=Related_analysis()
    # print("a b 之间的相关系数为",test.correlation(a,b))
    # print("相关系数矩阵为",test.corrlation_matrix(v))
    # print("a b 之间秩相关系数为",test.rank_correlation(a,b))
    # print('ab_c的一阶偏相关系数为：', test.partial_correlation(a,b,c))
    # print("a 为因变量，b,c为自变量的复相关系数",test.complex_correlation(a,b,c))
    #
    # x=[1,2,3,4,5]
    # y=[2,4,6,8,10]
    # regression=Regression_analysis()
    # a_estimated,b_estimated=regression.unary_linear_regression(y,x)
    # print("一元线性回归模型是y=",a_estimated,"+",b_estimated,"x")
    # print("多元线性回归模型的参数",regression.Multiline_linear_regression(a,v))
    #
    # system_cluster=System_clustering()
    # system_cluster.totalization_standardization(v)

    # #x y z m为聚类对象取值 不是变量值
    # x=[1,2,3,4,5]
    # y=[1,3,5,7,9]
    # z=[1,1,1,1,1]
    # m=[2,1,2,1,4]
    # v=[x,y,z,m]
    # system_cluster=System_clustering()
    # v=system_cluster.totalization_standardization(v)
    # distance=system_cluster.caculate_absolute_distance(v)
    # system_cluster.direct_clustering(v)
    # print(distance)
    #

    #kmeans
    # kmeans=Kmeans()
    # ##样本数据(Xi,Yi)，需要转换成数组(列表)形式
    # Xn = np.array([2, 3, 1.9, 2.5, 4])
    # Yn = np.array([5, 4.8, 4, 1.8, 2.2])
    #
    # # 标识符号
    # sign_n = ['A', 'B', 'C', 'D', 'E']
    # sign_k = ['k1', 'k2']
    # ##选取2个种子点
    # Xk, Yk =  kmeans.select_seed_all(2)
    # ##查看种子点
    # kmeans.draw_point_all_seed(Xk, Yk)
    # ##循环三次进行分类
    # for i in range(3):
    #     cls_dict =  kmeans.start_class(Xk, Yk)
    #     Xk_new, Yk_new =  kmeans.recal_class_point(Xk, Yk, cls_dict)
    #     Xk = Xk_new
    #     Yk = Yk_new
    #     kmeans.draw_point(Xk, Yk, cls_dict)

    #PCA
    # pca=PCA()
    # pca.main()

    #markov
    # markov= markov_chain()
    # markov.markov_chain_show()
    pass

