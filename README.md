# LogAnalysis

python版本：3.6

## Step 1、 Use LogCluster to parse log

**Description：**

* You can skip to Step 3 because the default dataset used has been already parsed.

* check the log path and output path in *loganalysis.py*

* **run  *loganalysis.py***

* all cluster result will be in your output path

* file is named ad (1,2,3...n).log*, according to the total number of logs under the cluster escending order **, ** n  is the total number of clusters

  eg 1.log

  ```
  //log key
      Dec 9 10:12:28 combo *{1,1} ttloop: read: Connection reset by peer  
      //Total number of logs under this cluster
      Support: 3                                                           
  
      //log line
      21705 21706 21707 
  ```



## Step 2、 Pre-process Log

**Description：**

* You can skip to Step 3 because the default dataset used has been already parsed.
* check the log path and output path in k8s/*log_preprocessor.py*
* **run *k8s/log_preprocessor.py* ** to get log key data
* check the log path and output path in k8s/*value.py, k8s/value_deal.py, k8s/value_extract.py
* **run k8s/*value.py, k8s/value_deal.py, k8s/value_extract.py*** to get variable data
* all data file is saved in the output path,  including train_dataset, validation_dataset, test_dataset...

## Step 3、Train log key anomaly detection model

**Description：**

* check the log key data path(the path in step2.2) and model output path in *log_key_LSTM_train.py*
* **run *Model1/log_key_LSTM_train.py* ** to train the log key anomaly detection model
* model is saved in the output path

## Step 4、Train variable anomaly detection model

**Description：**

* check the viriable data path(the path in step2.4) and model output path in *variable_LSTM_train.py*
* **run *Model2/viriable_LSTM_train.py* ** to train the variable anomaly detection model
* model is saved in the output path

## Step 5、Predict and get evaluations

**Description：**

* check the model path(the path in step3 and step4) in *log_predict.py*
* **run *log_predict.py* ** to get prediction and get evaluations