# README

### 1. 项目简介
现代系统大规模发展，已经成为 IT 行业的核心部分，支持各种在线服务和智能应用。这些系统大多设计为全天候 24x7 运行，每小时就有约几百 GB（约120至2亿行）大量日志，在系统操作期间记录详细的运行时信息。

此外5G的出现，不仅伴随着数据传输速率的提高、延迟的减少以及设备连接的大规模化，同时也伴随着数据量的增多、微服务的增多等，由此将产生大量的日志数据，这些数据中记录了众多的5G设备运行信息，如何利用这些信息做到日志**异常诊断智能化**，对于5G时代系统的运维具有重要的意义。

系统发生故障前，其日志通常都具有许多特征反应出系统的异常状态，如果我们能提前发现系统异常，就能避免系统发生更严重的错误。然而在目前大多数系统中，管理员都是在系统出现故障之后，根据日志信息进行排错。这种事后检测会浪费大量的时间和精力，并且不一定能取得很好的效果。

传统的日志分析方法具有速度慢，鲁棒性差，难以适应大型系统等弊端，以迫切需要自动的、基于日志的系统异常检测方案。因此本项目使用**机器学习**方法来**自动化**学习，检测，定位日志中可能出现的异常。


### 2. 项目构建方法
#### 2.1 环境准备
* 操作系统：Windows/Linux/Mac
* python>=3.6
* Flask==1.0.2
* Flask-Cors==3.0.6
* numpy==1.18.0
* scikit-learn==0.19.2
* scipy==1.1.0
* tensorboardX==1.9
* pytorch==1.2.0
* node.js>=8.11.3

#### 2.2 获取项目
git@github.com:XLab-Tongji/LogAnalysis.git

#### 2.3 导入项目
Open folder "LogAnalysis" with your IDE

### 3. 项目运行方法
##### 3.1 后端运行
* cd {project folder}/SE
* python app.py

##### 3.2 前端运行
- cd {project folder}/Frontend
- npm install
- npm run serve
- open browser with http://localhost:8080

### 4. 项目基本功能
* 日志清洗
* 日志聚类
* 基于log key的异常日志检测
* 基于log value的异常日志检测
* 基于workflow的异常日志定位

### 5. 代码结构说明

> 5G  5G日志相关文件
>
> k8s  k8s日志相关文件
>
> logcluster  logcluster聚类工具
>
> Model1  模型一相关代码
>
> > log_key_LSTM_train.py     模型一训练
>
> Model2  模型二相关代码
> > variable_LSTM_train.py  模型二训练
>
> Frontend
> > 
> >
> > 
>
> Backend
> > app.py  后端运行
> >
> > log_processor 数据预处理
> >
> > log_key_LSTM_train.py  模型一训练
> >
> > variable_LSTM_train.py  模型二训练
> >
> > LogPredict.py  日志预测
> >
> > upload  日志上传
>
> Docs  项目文档

