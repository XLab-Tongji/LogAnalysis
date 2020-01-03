## workflow构建

### 使用

修改代码最后一行后即可直接运行

```python
mainFlow(para1,para2,para3=None)
```
- para1 源数据文件文件名

- para2 窗口大小 建议取值范围为2-5

- para3 日志种类数，可选参数，默认值None。若未给出此参数则取源文件最大日志序号作为日志种类数

示例：

```python
mainFlow('vectorize',3,None)
```

### 流程

##### 1.将数据从文件读取到dataset中

对应函数为：

```python
def loadData(infile):
```

源数据文件为由字符串，空格与换行符构成的文本文件，需要将其处理为统一格式并存储在数组dataset中

如源数据文件为：

<img src="pics/12.PNG" width=30%>

处理后得到的dataset为：

<img src="pics/13.PNG" width=90%>

##### 2.构建data_tree

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

#####  3.检查并发事件

对应函数为：

```
def checkConcurrency(window_size,type_num):
```

在检查并发事件时，我们只考虑两个事件的并发检查，未进行多个事件的并发检查（只要考虑到多个事件并发出现的频率不高，且多事件并发检查效率较慢）。

遍历data_tree中每一个结点data_tree[i]:

​	若data_tree[i]的next_pattern3为[[1,2,3],[3,2,4],[2,1,3]]，则[1,2,3]与[2,1,3]中的事件2与事件1就是一组并发事件，即存在j，k满足：

```
if (data_tree[i].next_pattern3[j][0] == data_tree[i].next_pattern3[k][1]) and \
        (data_tree[i].next_pattern3[j][1] == data_tree[i].next_pattern3[k][0]) and \
        (data_tree[i].next_pattern3[j][2] == data_tree[i].next_pattern3[k][2]):
```

则data_tree[i].next_pattern3[j]的第零个与第一个元素是一组并发事件。

将并发事件合并为一个新事件：如12 53为两个并发事件，则将其合并后的新事件为12053，计算方法为事件1*1000+事件2

##### 4.检查新任务

对应函数为：

```
def checkNewTask(window_size,type_num):
```

完成并发事件检查后，dataset发生了改变，所以在检查新任务之前，要重新构建data_tree（直接调用第3步的函数即可）

新任务检查需要使用data_tree结点中的next_pattern与next_frequency

假设data_tree[0].next_pattern=[1,2,3,4,5,6,7]，data_tree[0].next_frequency=[856,2,3,1,523,123,3]

很明显，1 5 6三个事件经常发生在该模式之后，而2 3 4 7四个事件就很少发生，这种情况，我就认为1 5 6是这段程序执行后的三个不同分支，而这段程序也可能是一个任务的终点（因为一个任务终止后，下一个事件就是另一个任务的起始事件，而新任务起始实践是不确定的，所以出现概率很低）。

所以对dataset中所有data_tree[0].base_pattern后跟随2 3 4 7的模式，在2 3 4 7前作截断。对dataset中所有data_tree[0].base_pattern后跟随1 5 6的模式，认为这是一个正常任务流，不做处理。

##### 5.输出，重构dataset

对应函数为：

```python
def outputDataset(infile):
```

检查出所有新任务起点后，将结果输出到命名格式为"new"+infilename+".txt"的文本文件中。每一行代表一个任务。

并构建new_dataset，new_dataset是一个二维列表，new_dataset[0]表示第一个任务中的所有事件流

##### 6.检查循环事件并输出最终结果

对应函数为：

```python
def checkCycle(infile):
```

最后遍历new_dataset，对每一个任务中的事件流进行循环事件检查。

如1 3 5 7 2 6 2 6 2 6 3 终2 6是一个循环单元，那么只保留一个循环单元，并将循环起始位置的数字设置为-2*1000-2，最终结果为1 3 5 7 -2002 3。具体算法为：只保留一个循环单元，并将循环起始位置数字$m$替换为$-1 \times m \times 1000 - n$，其中$n$为循环单元的长度。如5 -12004 5 6 2 1 2 则表示5  (12 5 6 2) 1 2括号中为循环部分。

然后将结果输出到命名格式为"new2"+infilename+".txt"的文本文件中。这就是我们程序的最终运行结果。

### 可能的问题

1.只考虑两个事件并发，没有考虑多并发的情形。

2.目前程序中将并发事件作为一个新事件进行处理，但这种方法可能未必妥当。

3.目前新任务的检查方式可能会导致某些任务只含有一个或两个事件（这是由于不必要的截断导致的），但暂时没有想到好的避免方法。

4.检查新任务时候，需要设定一个阈值，当频率小于阈值时，视为是一个新任务起点。目前阈值的计算公式为0.02*window_size，因为没有没有办法进行验证，所以无法判断这个阈值的选取是否合理。



