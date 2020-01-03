### Drain: An Online Log Parsing Approach with Fixed Depth Tree     

#### 一.Drain简述

Drain是一个日志解析工具，主要用于web service management  可以将raw log message 解析为log event

#### 二.Drain主要特点

​       1.online   不需要完所有日志再进行训练

​       2.高准确率和高速运行

#### 三.Drain所采用的方法

​       构造一个定深树 如下

![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/Drain0.png)

##### 3.1这是一个三层的定深树    有三种节点类型     

①Root表示树的根节点   

②内部节点:

Length:4,Length:5,Length:10  表示预处理过的log message的长度（token数）

③叶节点:每个叶节点存储一个log groups 的lists  

说根节点和内部节点encode特殊设计的规则去引导搜索过程

##### 3.2 两个参数

depth:表示定深树的深度

maxChild：每个节点的最大子节点数   防止分支爆炸

3.3定深树工作过程   

个人理解：将前几个token相同的log聚为分到不同的group里

​       当有一个新的日志到来的时候，会先经过简单的预处理，接下来会经过特殊的规则进行搜索，目的是到达一个叶节点找到一个log group ，如果找到，将其匹配并记录下来，否则创建一个新的log group

#### 四.完整的Drain流程

#####        4.1预处理  

​	新到来的log message用domain knowledge进行预处理.就是用简单的正则表达式，将log message 中的诸如IP address的token移除。

#####        4.2通过log message length进行搜索

​        得到预处理之后的log message 之后，计算token数作为log Length进入第二层节点   

#####        4.3通过log message前几个token进行搜索  穿过depth-2层

​              比如“Receive from node  4”会进入上图中的Receive节点

​              为了防止分支爆炸，将所有数字匹配到一个独特的节点“*”中   并且当达到maxChild之后，其他未能匹配的log message全部去匹配“*”  

​        如果未匹配且未达到maxChild   创建对应节点

#####        4.4通过相同的token搜索

​              经过之前的步骤，现在已经到了一个叶子节点

​             这一步要去从log group list中选择要将message归于那个group

​              计算simSeq

![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/Drain1.png) 

Seq1(i)和seq2(i)分别代表log message和log event的第i个token

Log event 应该是指每个group的pattern   n为log length

比较log message和每个group的log event的token是否一样   

​       ![img](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/Drain2.png)

比较simSeq是否超过阈值st    但没有说这个阈值从哪里来的

若超过，返回simSeq最高的group

若没有超过 返回一个flag(eg None in python)

#####        4.5 更新解析树

​       ①如果在第四步中匹配成功

​              将log ID加入group的log IDs

​              更新log event 扫描log message和log event，如果相同位置的token相同，则不做修改，如果不同，用通配符(wildcard)即”*”更新那个位置    

​       ②如果在第四步中没匹配成功

​              创建一个新的log group  log IDs仅仅包含这个message logID 

​                                                    Log event 就是log message 

#### 五.评估

在后面的评估中将maxChild设为100

用Drain做异常检测：    

用不同的日志解析日志  发现Drain最后得到的异常检测效果最好

 
