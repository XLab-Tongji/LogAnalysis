from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template,Response
from werkzeug.utils import secure_filename
from tensorboardX import SummaryWriter
import os
import random
import shutil
from flask_cors import CORS
from abc import ABCMeta
from abc import abstractmethod
from flask.views import MethodView
app = Flask(__name__)
CORS(app, resources=r'/*')
app.config['UPLOAD_FOLDER'] = 'upload'

trainfiles = []
testfiles = []

#接口类
class train(metaclass=ABCMeta):
    @abstractmethod
    def uploadTrainLog(self):
        pass
    @abstractmethod
    def startTrain(self):
        pass
    @abstractmethod
    def showTrain(self):
        pass
class predict(metaclass=ABCMeta):
    @abstractmethod
    def startPredict(self):
        pass
    @abstractmethod
    def uploadAbnormalLog(self):
        pass

app.url_rule_class
app.add_url_rule
#实现类

class Training(train):
    def startTrain(self):
        if os.path.exists('output.txt'):
            os.remove('output.txt')
        print(trainfiles[-1])
        if trainfiles[-1][0] == '5':
            '''
                聚类
            '''
            os.system('python loganalysis_5G.py')
            '''
                拆log key
            '''
            os.system('python log_preprocessor_5G.py')
            '''
                拆log value
            '''
            os.system('python value_normalize.py')
            '''
                model1 train
            '''
            os.system('python log_key_LSTM_train_5G.py')
            '''
                model2 train
            '''
            os.system('python variable_LSTM_train_5G.py')
        else:
            '''
                聚类
            '''
            os.system('python loganalysis.py')
            '''
                拆log key
            '''
            os.system('python log_preprocessor.py')
            '''
                拆log value
            '''
            rootpath = '../k8s/'
            if os.path.exists(rootpath + 'dealedvalue.log'):
                os.remove(rootpath + 'dealedvalue.log')
            if os.path.exists(rootpath + 'normalize_value.log'):
                os.remove(rootpath + 'normalize_value.log')
            if os.path.exists(rootpath + 'value.log'):
                os.remove(rootpath + 'value.log')
            valuepath = rootpath + 'LogClusterResult-k8s/' + 'logvalue/'
            shutil.rmtree(valuepath + 'logvalue_test/')
            shutil.rmtree(valuepath + 'logvalue_abnormal/')
            shutil.rmtree(valuepath + 'logvalue_train/')
            shutil.rmtree(valuepath + 'logvalue_val/')
            os.mkdir(valuepath + 'logvalue_test')
            os.mkdir(valuepath + 'logvalue_abnormal')
            os.mkdir(valuepath + 'logvalue_train')
            os.mkdir(valuepath + 'logvalue_val')
            os.system('python deal.py')
            '''
                model1 train
            '''
            os.system('python log_key_LSTM_train.py')
            '''
                model2 train
            '''
            os.system('python variable_LSTM_train.py')
        return "fininsh training"
    def showTrain(self):
        try:
            outfile = './output.txt'
            meg = ['0', '0']
            with open(outfile, 'r') as file:
                lines = file.readlines()  # 读取所有行
                try:
                    first_line = lines[0]  # 取第一行
                    msg = first_line.split();
                except IndexError:
                    msg = ['0', '0']
        except FileNotFoundError:
            msg = ['0', '0']
        msg = dict(step=msg[0], progress=msg[1])
        return jsonify(msg)
    def uploadTrainLog(self):
        file = request.files['file']  # 获取上传的文件
        print(file.filename)
        if file:
            filename = secure_filename(file.filename)
            trainfiles.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'train', filename))
            return render_template("home.html", trainuploaded=True, testuploaded=False)
        else:
            return render_template("home.html", trainuploaded=False, testuploaded=False)

class Prediction(predict):
    def startPredict(self):
        if testfiles[-1][0] == '5':
            os.system('python LogPredict_5G.py')
        else:
            os.system('python LogPredict.py')
        msg9 = []
        try:
            outfile = './upload/test/' + testfiles[-1]
            print(outfile)
            msg1 = 0
            msg6 = []
            with open(outfile, 'r') as file:
                lines = file.readlines()  # 读取所有行
                msg1 = len(lines)
                msg6 = lines
        except FileNotFoundError:
            msg1 = 0
            msg6 = []
        try:
            outfile = './model1_predict.txt'
            msg4 = ''
            msg7 = []
            with open(outfile, 'r') as file:
                lines = file.readlines()  # 读取所有行
                try:
                    first_line = lines[0]  # 取第一行
                    msg4 = first_line.split();
                    for i in range(len(msg4)):
                        msg9.append(msg4[i])
                    msg7 = len(msg4)
                except IndexError:
                    msg4 = ''
                    msg7 = 0
        except FileNotFoundError:
            msg4 = ''
            msg7 = 0
        try:
            outfile = './model2_predict.txt'
            msg5 = ''
            msg3 = '0'
            with open(outfile, 'r') as file:
                lines = file.readlines()  # 读取所有行
                try:
                    first_line = lines[0]  # 取第一行
                    msg5 = first_line.split();
                    msg3 = lines[1]
                    msg8 = len(msg5)
                    for i in range(len(msg5)):
                        msg9.append(msg5[i])
                except IndexError:
                    msg5 = ''
                    msg3 = '0'
                    msg8 = 0
        except FileNotFoundError:
            msg5 = ''
            msg3 = '0'
            msg8 = 0
        msg2 = len(msg4) + len(msg5)
        msg = dict(total_log=msg1, total_abnormal_num=msg2, model1_adnormal_num=msg7, model2_abnormal_num=msg8,
                   consume_time=msg3 + 's', abnormal=msg9, model1_abnormal=msg4, model2_abnormal=msg5, log=msg6
                   )
        return jsonify(msg)

    def uploadAbnormalLog(self):
        file = request.files['file']  # 获取上传的文件
        print(file.filename)
        if file:
            filename = secure_filename(file.filename)
            testfiles.append(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'test', filename))
            return render_template("home.html", trainuploaded=True, testuploaded=True)
        else:
            return render_template("home.html", trainuploaded=True, testuploaded=False)
Train=Training()
prediction=Prediction()

@app.route('/uploadTrainLog',methods=['post'])
def uploadTrainLog():
    Train.uploadTrainLog()

@app.route('/startTrain',methods=['get'])
def train():
    Train.startTrain()


@app.route('/showTrain',methods=['get'])
def showTrain():
    Train.showTrain()



@app.route('/uploadAbnormalLog',methods=['post'])
def uploadAbnormalLog():
    prediction.uploadAbnormalLog()


@app.route('/startPredicate',methods=['get'])
def predict():
    prediction.startPredict()


@app.route("/")
def main():
    return render_template("home.html",trainuploaded = False,trainstarted=False,trainfinished =False
                           ,testuploaded=False,teststarted=False,testfinished=False)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000,debug=True)
