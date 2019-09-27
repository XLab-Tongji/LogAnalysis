# LogAnalysis

## LogCluster

**说明：**

* 原始日志文件需放在*LogFile* 目录下，然后运行*loganalysis.py*

* 输出所有聚类结果，路径为*WriteFile/cluster

* 输出文件命名为*(1,2,3...n).log*, 按聚类下日志总数**降序**排列, **n**为总聚类数

  样例输出文件：

  ```
  //log key
      Dec 9 10:12:28 combo *{1,1} ttloop: read: Connection reset by peer  
      //该聚类下的日志总数
      Support: 3                                                           
  
      //log 在原文件的行数
      21705 21706 21707 
  ```




## Model1

This is for training a model by using all the keys.