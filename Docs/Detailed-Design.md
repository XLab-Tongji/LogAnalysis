# 详细设计规约

## 背景

5G的出现，不仅伴随着数据传输速率的提高、延迟的减少以及设备连接的大规模化，同时也伴随着数据量的增多、微服务的增多等，由此将产生大量的日志数据，这些数据中记录了众多的5G设备运行信息，如何利用这些信息做到日志异常诊断智能化，对于5G时代系统的运维具有重要的意义。

系统发生故障前，其日志通常都具有许多特征反应出系统的异常状态，如果我们能提前发现系统异常，就能避免系统发生更严重的错误。然而在目前大多数系统中，管理员都是在系统出现故障之后，根据日志信息进行排错。这种事后检测会浪费大量的时间和精力，并且不一定能取得很好的效果。

传统的日志分析方法具有速度慢，鲁棒性差，难以适应大型系统等弊端，以迫切需要自动的、基于日志的系统异常检测方案。因此本项目使用机器学习方法来自动化学习，检测，定位日志中可能出现的异常。

## 过程流设计

- 用户异常检测总体过程流

![img](https://github.com/XLab-Tongji/LogAnalysis/raw/master/Docs/pics/user_seq.png)

- 异常检测算法过程流

![predict_seq.png](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/predict_seq.png?raw=true)

## 日志结构

[Log Structure文档](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/Log-Structure.md)从以下五个角度研究了日志的组成结构，具体日志结构的调研结果详见文档

- Who to log
- How to log
- Where to log
- What to log
- Whether to log

## 算法设计

算法部分，主要包含四部分的详细设计

- 聚类算法
- Logkey异常检测模型
- Value异常检测模型
- Workflow算法

### 聚类算法

#### LogCluster

以下为LogCluster的介绍，关于LogCluster的具体使用，注意事项等详见[LogCluster说明文档](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/LogCluster.pdf) 

- LogCluster简介<br>
  &emsp;&emsp;LogCluster是一个开源的基于perl语言的命令行日志分析工具，能够从大量的蕴含了事件的日志数据文件中挖掘出有意义的日志模式并对日志进行聚类，通过传入一系列的参数和参数值，来改变LogCluster的聚类算法分析效果。
- LogCluster算法流程图

![img](<https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/logcluster/LogCluster.png>)

#### Sequence

以下为Sequence的介绍，关于Sequence的具体使用，注意事项等详见[Sequence说明文档](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/Sequence.md) 

- 算法简介<br><br>
  它基于一个基本概念，对于多个日志消息，如果同一个位置共享一个相同的父节点和一个相同的子节点，然后在这个位置可能是变量字符串，这意味着我们可以提取它。<br><br>
  例如，查看以下两条messages：<br><br>
  Jan 12 06:49:42 irc sshd[7034]: Accepted password for root from 218.161.81.238 port 4228 ssh2<br>
  Jan 12 14:44:48 jlz sshd[11084]: Accepted publickey for jlz from 76.21.0.16 port 36609 ssh2<br><br>
  每条message的第一个token是一个时间戳，并且第三个token是一个字面量sshd,对于字面量irz和jlz他们共享同一个父节点时间戳，共享同一个孩子节点sshd, 这意味着在这两者之间的token也就是每个消息中的第二个令牌，可能表示此message类型中的变量token。在这种情况下，“irc”和“jlz”碰巧表示系统日志主机。
- 程序命令analyze算法流程图

![img](<https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/sequence.png>)

#### FT-tree

以下为FT-tree算法介绍，关于FT-tree的具体使用，注意事项等详见[FT-tree说明文档](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/Fttree.md)

1. 读取日志文件，将其存储在log_list变量中

2. 遍历log_list，找出所有日志类型，并构造索引表log_type_index（索引表只是为了加速计算）

3. 提取log_list中的detailed message存储在log_message中

4. 统计所有日志中所有单词及其出现次数，存储在字典word_support中，key为单词，value为频数

5. 将word_support根据value值进行排序，将排序结果存储在word_list中

6. 分别对每一种类型的日志计算其词频排序，并构造FT-tree

7. 根据k值对FT-tree进行减枝

8. 将聚类结果输出到指定文件中

   算法的具体实现难以概述，故结合下面案例对该算法进行较为清晰的介绍：

   <img src="pics/6.PNG" width=100%>

- 上图为读入的日志文件（部分），首先提取出Message type与Detailed meesage两个字段，如下图所示

  <img src="pics/7.PNG" width=90%>

- 统计所有日志中每个单词的出现频次，按大小排序，如上述日志中，单词出现频数排序如下：

  | changed | state | to   | Interface | Vlan-interface | down | up   | ae3  | ae1  | vlan22 | vlan20 |
  | ------- | ----- | ---- | --------- | -------------- | ---- | ---- | ---- | ---- | ------ | ------ |
  | 8       | 8     | 8    | 4         | 4              | 4    | 4    | 2    | 2    | 2      | 2      |

- 将所有日志按Message type分类，每一类对应一棵FT-tree，FT-tree根节点即为日志类型，如上述日志构造出两棵FT-tree，一颗根节点为SIF，另一棵根结点为OSPF

- 遍历每一条日志，根据上一步得到的词频排序表，对每条日志中的单词按词频顺序排列，并据此构造FT-tree

  如“Interface ae3, changed state to down”排序后变为：changed state to Interface down ae3

  构造的FT-tree为

  <img src="pics/8.PNG" width=25%>

  再遍历下一条同类型日志，不断对FT-tree作扩充：

  <img src="pics/10.PNG" width=80%>

- 根据阈值k，对FT-tree进行剪枝，如k等于5，则FT-tree包含根节点后最大深度为5+1=6，则深度大于6的结点都被舍弃

  <img src="pics/11.PNG" width=60%>

- 最后的FT-tree中有多少个叶子结点，就说明得到了多少种聚类，把所有Message Type对应的FT-tree聚类数相加即为聚类总数

- 最后将聚类结果与日志进行匹配，得到每一条日志的聚类类型

<br><br>

#### Drain

以下为Drain的介绍，关于Sequence的具体使用，注意事项等详见[Drain说明文档](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/Drain.md)
Drain是一个日志解析工具，主要用于web service management  可以将raw log message 解析为log event

下面是完整的Drain流程

- 预处理 <br>
  &emsp;&emsp;新到来的log message用domain knowledge进行预处理.就是用简单的正则表达式，将log message 中的诸如IP address的token移除。
- 通过log message length进行搜索<br>
  &emsp;&emsp;得到预处理之后的log message 之后，计算token数作为log Length进入第二层节点   
- 通过log message前几个token进行搜索  穿过depth-2层<br>
  &emsp;&emsp;比如“Receive from node  4”会进入上图中的Receive节点<br>
  &emsp;&emsp;为了防止分支爆炸，将所有数字匹配到一个独特的节点“*”中   并且当达到maxChild之后，其他未能匹配的log message全部去匹配“*”  <br>
  &emsp;&emsp;如果未匹配且未达到maxChild   创建对应节点
- 通过相同的token搜索<br>
  &emsp;&emsp;经过之前的步骤，现在已经到了一个叶子节点<br>
  &emsp;&emsp;这一步要去从log group list中选择要将message归于那个group<br>
  &emsp;&emsp;计算simSeq<br>
  ![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/Drain1.png) <br>
  &emsp;&emsp;Seq1(i)和seq2(i)分别代表log message和log event的第i个token<br>
  &emsp;&emsp;Log event 应该是指每个group的pattern   n为log length比较log message和每个group的log event的token是否一样<br>
     ![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/Drain2.png)<br>
   &emsp;&emsp;比较simSeq是否超过阈值st<br>
   &emsp;&emsp;若超过，返回simSeq最高的group<br>
   &emsp;&emsp;若没有超过 返回一个flag(eg None in python)<br>
- 更新解析树
  - 如果在第四步中匹配成功<br>
    &emsp;&emsp;将log ID加入group的log IDs,更新log event 扫描log message和log event，如果相同位置的token相同，则不做修改，如果不同，用通配符(wildcard)即”*”更新那个位置
  - 如果在第四步中没匹配成功<br>
    &emsp;&emsp;创建一个新的log group  log IDs仅仅包含这个message logID <br>
    &emsp;&emsp;Log event 就是log message 
    <br><br>

#### Louvain社区发现算法

Louvain社区发现算法是一种基于图论的聚类算法，Louvain算法思想如下，其具体说明详见[Louvain社区发现算法](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/Louvain-Algorithm.md):

1. 将图中的每个节点看成一个独立的社区，次数社区的数目与节点个数相同；
2. 对每个节点i，依次尝试把节点i分配到其每个邻居节点所在的社区，计算分配前与分配后的模块度变化ΔQ，并记录ΔQ最大的那个邻居节点，如果maxΔQ>0，则把节点i分配ΔQ最大的那个邻居节点所在的社区，否则保持不变；
3. 重复2，直到所有节点的所属社区不再变化；
4. 对图进行压缩，将所有在同一个社区的节点压缩成一个新节点，社区内节点之间的边的权重转化为新节点的环的权重，社区间的边权重转化为新节点间的边权重；
5. 重复1，直到整个图的模块度不再发生变化。<br><br>

### Model1：log key anomaly detection model算法设计

模型一是通过每条log的key值来对日志的异常与否进行判断。该模型的训练数据是从日志文件中提取出来的每条日志的key值数字化以后的一系列数字流，每个key值都有对应的正常（1）或异常（0）标签。我们选取正常日志的key值流来对模型一进行训练。

- 原理<br>
  &emsp;&emsp;该模型可被视作一个多分类模型，每一个不同的log key代表了不同的类。令K={k1, k2, …, kn}，代表的是从日志中提取出来的不同的log key值的集合。<br>
  &emsp;&emsp;原始日志的key值流反映了被测系统的特定的事件执行顺序和状态，利用这个特点可以基于LSTM神经网络训练一个基于上下文的异常日志检测模型。设mi是日志的key值流中出现在i位置的log key，则miK，且mi的值对之前出现的key值流有很强的依赖性，设mt为我们要进行异常检测的log key，我们取一个长度为h的窗口，w = {mt-h, …, mt-2, mt-1}，则mt的值可以由w中的值来进行预测，将预测结果与mt的实际值进行比对，便能判断mt是否正常。

- 模型结构<br>
  &emsp;&emsp;模型一为多层Lstm结构，下面是模型结构图：<br>
  ![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/model1/model1_structure.png)

  ```python
  class Model(nn.Module):
      def __init__(self, input_size, hidden_size, num_of_layers, num_of_keys):
          super(Model, self).__init__()
          self.hidden_size = hidden_size
          self.num_of_layers = num_of_layers
          self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True)
          self.fc = nn.Linear(hidden_size, num_of_keys)
  ```

- 训练阶段<br>
  &emsp;&emsp;供训练的log key值流会被分成长度为h的子流，每个子流包含两部分含义：历史log key值流和当前log key值。例如，有一个正常的log key值流为{k23, k6, k12, k5, k26, k12}，设窗口长度h=3，则训练数据将被分成如下形式：{k23, k6, k12 -> k5}, {k6, k12, k5 -> k26}, {k12, k5, k26 -> k12}。

  对应训练的数据处理的代码为

  ```python
  def generate_logkey_label(file_path):
      num_of_sessions = 0
    input_data, output_data = [], []
      with open(file_path, 'r') as file:
          for line in file.readlines():
              num_of_sessions += 1
              line = tuple(map(lambda n: n, map(int, line.strip().split())))
              for i in range(len(line) - window_length):
                  input_data.append(line[i:i + window_length])
                  output_data.append(line[i + window_length])
      data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
      return data_set
  ```

- 检测阶段<br>
  &emsp;&emsp;为了检测一个log key （mt）是否异常，将向模型输入mt之前的h个log值流w={mt-h, …, mt-1}，输出输出是一个条件概率分布结果<br>
  &emsp;&emsp; Pr[mt = ki | w ] = { k1: p1, k2: p2, …, kn: pn }，ki属于K( I = 1, …, n )<br>
  &emsp;&emsp;若概率最高的前g个候选值中包含了mt，则mt被视为正常，否则为异常。

- 模型一的工作过程如下图所示：<br>
  ![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/model1/model1_flow.png)
  <br><br>

### Model2：parameter value anomaly detection model for each log key 算法设计

&emsp;&emsp;模型一对于系统事件流中的异常检测非常有帮助，但是还有一些异常不能由这些key值直接检测到，它们隐藏在每条log的其他参数值当中。模型二能解决这个问题，其是针对每个log key训练的异常日志检测模型。在该部分，对于每个不同的log key，都会训练一个单独的模型出来。

&emsp;&emsp;模型二的训练数据是针对某个特定的log key而言的，这些数据都是与时间序列有关的一系列参数组成的向量集。下图是一个日志的key值和其他参数值（parameter value）的提取示例：

![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/model2/model2_value.png)

- 模型结构<br>
  &emsp;&emsp;模型二为多层Lstm结构，下面是模型结构图：<br>
  ![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/model2/model2_structure.png)

  ```python
  class Model(nn.Module):
      def __init__(self, input_size, hidden_size, num_of_layers, out_size):
          super(Model, self).__init__()
          self.hidden_size = hidden_size
          self.num_of_layers = num_of_layers
          self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True)
          self.fc = nn.Linear(hidden_size, out_size)
  ```

- 训练阶段<br>
  &emsp;&emsp;由于针对于每个log key，其parameter value vector同时间序列有关，例如，对于k2，构造出来的用于训练的向量集可表示如下：{[t2 – t1, 0.2], [t2’ – t1’, 0.7], … … }，因此我们可以再次利用LSTM来搭建用于训练的神经网络。在对训练数据进行预处理的时候，需要对数据进行归一化处理，我们的处理办法是：对于属于同一个log key的所有参数值向量，将在同一位置出现的参数值（parameter value），通过计算均值和标准差，用Z-score标准化方法对数据进行归一化处理。模型二的输出是一个对于下一个参数值向量的预测。该预测结果以之前的历史日志数据为基础，是一个向量。这里我们也能用到模型一中窗口长度的思想来对模型进行训练。

  ```
  对应训练的数据处理的代码为
  ```

  ```python
  def generate_value_label(file_path):
      num_sessions = 0
    inputs = []
      outputs = []
      vectors = []
      with open(file_path, 'r') as f:
          for line in f.readlines():
              num_sessions += 1
              line = tuple(map(lambda n: n, map(float, line.strip().split())))
              vectors.append(line)
      for i in range(len(vectors) - window_length):
          inputs.append(vectors[i: i + window_length])
          outputs.append(vectors[i + window_length])
      data_set = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
      if len(vectors) > 0:
          return data_set, len(vectors[0])
      else:
          return None, 0
  ```

- 检测阶段<br>
  &emsp;&emsp;在利用模型二对日志的异常与否进行检测的时候，我们采用均方误差(Mean square error, MSE)的方法来计算预测出来的向量和真实的向量之间的差异。在这里判断一条日志是否是异常的时候，我们并没有设置阈值来判断。我们的方法是：将训练数据分为训练集（training set）和验证集 (validation set)两部分，用训练集来训练我们的模型，而对于验证集中的每个参数值向量v，利用训练出来的模型计算预测出的参数值向量（该预测结果需要用到该验证集中位于每个v之前的向量）和v之间的MSE。在每一个时间步中，使用验证集得到的这些MSE服从高斯分布。于是，在进行异常检测的时候，如果预测参数值向量和真实参数值向量之间的MSE位于得到的高斯分布的置信区间内，则该日志是正常的，否则被视为异常。

综上，综合模型一和模型二，异常检测的作过程如下图所示：

![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/model2/flow.png)

&emsp;&emsp;当捕获到一条新的日志时，系统会将该日志解析成一个kog key和对应的一个参数值组成的向量（parameter value vector）。系统首先利用模型一对这个log key进行检测，看其是否正常，如果正常，系统会利用模型二对参数值向量做进一步的异常检测。若两个步骤的检测结果都表明该日志是正常的日志，则该日志正常，否则该日志异常。

<br><br>

### Workflow算法设计

以下为Workflow算法介绍，关于Workflow的具体使用，注意事项等详见[Workflow说明文档](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/Workflow.md)

#### 将数据从文件读取到dataset中

对应函数为：

```python
def loadData(infile):
```

源数据文件为由字符串，空格与换行符构成的文本文件，需要将其处理为统一格式并存储在数组dataset中

如源数据文件为：

<img src="pics/12.PNG" width=30%>

处理后得到的dataset为：

<img src="pics/13.PNG" width=90%>

#### 构建data_tree

对应函数为：

```
def buildTree(window_size,type_num):
```

为了起到节省空间的目的，设计了一了class Node：

```python
class Node:
    def __init__(self,_base_pattern,_next_pattern,_next_frequency,_next_pattern3,_next_frequency3):
        self.base_pattern = _base_pattern          # 扫描dataset得到的长度为window_size所有模式
        self.next_pattern = _next_pattern          # 向后扫描一个长度得到的不同模式
        self.next_frequency = _next_frequency      # next_pattern中各模式出现的频数
        self.next_pattern3 = _next_pattern3        # 向后扫描三个长度得到的不同模式
        self.next_frequency3 = _next_frequency3    # next_pattern3中各模式出现的频数
```

data_tree则是由多个Node组成的一个列表。

比如我们的dataset为[1,2,3,3,3,1,2,3,1,2,3,3,2,2,1,3]，window_size=3

- ```
  data_tree[0].base_pattern=[1,2,3]
  data_tree[0].next_pattern=[1,3]
  data_tree[0].next_frequency=[1,2] #表示base[1,2,3]后出现1次数为1 出现3次数为2
  data_tree[0].next_pattern3=[[3,3,1],[1,2,3],[3,2,2]]
  data_tree[0].next_frequency3=[1,1,1] #表示[1,2,3]后出现上述三种魔术的次数都是1
  
  ```

- ```
  data_tree[1].base_pattern=[2,3,3]
  data_tree[1].next_pattern=[2,3]
  data_tree[1].next_frequency=[1,1] 
  data_tree[1].next_pattern3=[[3,1,2],[2,2,1]]
  data_tree[1].next_frequency3=[1,1] 
  
  ```

  依此类推构建出data_tree，使得data_tree列表中结点的base_pattern涵盖dataset中出现的所有长度为window_size的模式

  我们将会使用next_pattern3进行并发事件检查，使用next_pattern进行新任务检查

#### 检查并发事件

对应函数为：

```
def checkConcurrency(window_size,type_num):

```

在检查并发事件时，我们只考虑两个事件的并发检查，未进行多个事件的并发检查（只要考虑到多个事件并发出现的频率不高，且多事件并发检查效率较慢）。

遍历data_tree中每一个结点data_tree[i]:

```
若data_tree[i]的next_pattern3为[[1,2,3],[3,2,4],[2,1,3]]，则[1,2,3]与[2,1,3]中的事件2与事件1就是一组并发事件，即存在j，k满足：

```

```
if (data_tree[i].next_pattern3[j][0] == data_tree[i].next_pattern3[k][1]) and \
        (data_tree[i].next_pattern3[j][1] == data_tree[i].next_pattern3[k][0]) and \
        (data_tree[i].next_pattern3[j][2] == data_tree[i].next_pattern3[k][2]):

```

则data_tree[i].next_pattern3[j]的第零个与第一个元素是一组并发事件。

将并发事件合并为一个新事件：如12 53为两个并发事件，则将其合并后的新事件为12053，计算方法为事件1*1000+事件2

#### 检查新任务

对应函数为：

```
def checkNewTask(window_size,type_num):

```

完成并发事件检查后，dataset发生了改变，所以在检查新任务之前，要重新构建data_tree（直接调用第3步的函数即可）

新任务检查需要使用data_tree结点中的next_pattern与next_frequency

假设data_tree[0].next_pattern=[1,2,3,4,5,6,7]，data_tree[0].next_frequency=[856,2,3,1,523,123,3]

很明显，1 5 6三个事件经常发生在该模式之后，而2 3 4 7四个事件就很少发生，这种情况，我就认为1 5 6是这段程序执行后的三个不同分支，而这段程序也可能是一个任务的终点（因为一个任务终止后，下一个事件就是另一个任务的起始事件，而新任务起始实践是不确定的，所以出现概率很低）。

所以对dataset中所有data_tree[0].base_pattern后跟随2 3 4 7的模式，在2 3 4 7前作截断。对dataset中所有data_tree[0].base_pattern后跟随1 5 6的模式，认为这是一个正常任务流，不做处理。

#### 输出，重构dataset

对应函数为：

```python
def outputDataset(infile):

```

检查出所有新任务起点后，将结果输出到命名格式为"new"+infilename+".txt"的文本文件中。每一行代表一个任务。

并构建new_dataset，new_dataset是一个二维列表，new_dataset[0]表示第一个任务中的所有事件流

#### 检查循环事件并输出最终结果

对应函数为：

```python
def checkCycle(infile):

```

最后遍历new_dataset，对每一个任务中的事件流进行循环事件检查。

如1 3 5 7 2 6 2 6 2 6 3 终2 6是一个循环单元，那么只保留一个循环单元，并将循环起始位置的数字设置为-2*1000-2，最终结果为1 3 5 7 -2002 3。具体算法为：只保留一个循环单元，并将循环起始位置数字$m$替换为$-1 \times m \times 1000 - n$，其中$n$为循环单元的长度。如5 -12004 5 6 2 1 2 则表示5  (12 5 6 2) 1 2括号中为循环部分。

然后将结果输出到命名格式为"new2"+infilename+".txt"的文本文件中。这就是我们程序的最终运行结果。
