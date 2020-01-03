# High-level Design (概要设计规约)

## Prototype Design (原型设计)

​	项目的UI原型

![训练](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/%E8%AE%AD%E7%BB%83.png)

![预测报表](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/%E9%A2%84%E6%B5%8B%E6%8A%A5%E8%A1%A8.png)

![预测结果](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.png)

## Business Architecture (业务架构)



## ![概要设计](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/%E6%A6%82%E8%A6%81%E8%AE%BE%E8%AE%A1.png)

## chnology Architecture (技术架构)

## ![架构](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/%E6%9E%B6%E6%9E%84.png)



## 接口规约

### *dictionary/main_dictionary*

后端设计了两个接口类管理所有接口

![绘图1](https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/interface.jpg)



***/uploadTrainLog***

#### 接口描述

上传训练文件

|                |          |
| -------------- | -------- |
| Request Method | POST     |
| Authorization  | Required |

#### 参数

| Name | Located in | Description  | Required | Schema |
| ---- | ---------- | ------------ | -------- | ------ |
| file | form       | 日志训练文件 | Yes      | file   |

#### 返回结果

| Code | Description         | Schema |
| ---- | ------------------- | ------ |
| 200  | Successful response | String |

#### 示例请求

```
post /uploadTrainLog
body: form_data : file=file
```

#### 示例结果

```
{
  code : 200
  message : upload successfully!
  data : null
}
```

### */startTrain*

#### 接口描述

开始训练

|                |          |
| -------------- | -------- |
| Request Method | GET      |
| Authorization  | Required |

#### 参数

null

#### 返回结果

| Code | Description         | Schema |
| ---- | ------------------- | ------ |
| 200  | Successful response | String |

#### 示例请求

```
GET /startTrain
```

#### 示例结果

```
{
  code : 200,
  message : "start successfully",
  data : null 
}
```

### */showTrain*

#### 接口描述

返回训练进度

|                |          |
| -------------- | -------- |
| Request Method | GET      |
| Authorization  | Required |

#### 参数

null

#### 返回结果

| Code | Description         | Schema |
| ---- | ------------------- | ------ |
| 200  | Successful response | String |

#### 示例请求

```
GET /showTrain
```

#### 示例结果

```
{
  code : 200,
  message : "show successfully",
  data :  50%
}
```

### *UploadAbnormalLog*

#### 接口描述

上传异常日志文件

|                |          |
| -------------- | -------- |
| Request Method | POST     |
| Authorization  | Required |

#### 参数

| Name | Located in | Description  | Required | Schema |
| ---- | ---------- | ------------ | -------- | ------ |
| file | form       | 日志预测文件 | Yes      | file   |

#### 返回结果

| Code | Description         | Schema |
| ---- | ------------------- | ------ |
| 200  | Successful response | String |

#### 示例请求

```
post /uploadTrainLog
body: form_data : file=file
```

#### 示例结果

```
{
  code : 200
  message : upload successfully!
  data : null
}
```

### */startPredict*

#### 接口描述

开始进行预测

|                |          |
| -------------- | -------- |
| Request Method | POST     |
| Authorization  | Required |

#### 参数

- 无

#### 返回结果

| Code | Description         | Schema  |
| ---- | ------------------- | ------- |
| 200  | Successful response | *string |

#### 示例请求

```
get /startPredict

```

#### 示例结果

```
[ 
	total_log=""//返回所有的日志,
	total_abnormal_num="50",
	model1_adnormal_num="30",
	model2_abnormal_num="20",
	consume_time="20s",
	abnormal=""//返回异常日志,
	model1_abnormal=""//返回model1检测的异常日志,
	model2_abnormal=""//返回model2检测的异常日志
]

```

#### 
