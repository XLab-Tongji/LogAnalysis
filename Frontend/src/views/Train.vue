<template>
  <el-card class="trainCard">
    <div slot="header">
      <span style="font-size:18px">模型训练阶段</span>
    </div>
    <div class="step-lt">
      <el-steps direction="vertical" :active="active" finish-status="success">
        <el-step title="上传训练文件">
          <template slot="description"></template>
        </el-step>
        <el-step title="数据预处理">
          <template v-if="active > 0" slot="description">
            <div v-if="showTip1">日志聚类已完成</div>
            <div v-if="showTip2">Log Key 提取完成</div>
            <div v-if="showTip3">Log Value 提取</div>
            <el-progress
              v-if="showTip3"
              :text-inside="true"
              :stroke-width="22"
              :percentage="progress3"
            ></el-progress>
          </template>
        </el-step>
        <el-step title="模型一训练">
          <br>
          <template slot="description">
            <el-progress
              v-if="active > 2"
              :text-inside="true"
              :stroke-width="22"
              :percentage="progress4"
            ></el-progress>
          </template>
        </el-step>

        <el-step title="模型二训练">
          <br>
          <template slot="description">
            <el-progress
              v-if="active > 3"
              :text-inside="true"
              :stroke-width="22"
              :percentage="progress5"
            ></el-progress>
          </template>
        </el-step>
        <el-step title="异常预测">
          <template slot="description">
            <el-progress
              v-if="active > 4"
              :text-inside="true"
              :stroke-width="22"
              :percentage="progress6"
            ></el-progress>
          </template>
        </el-step>
      </el-steps>
    </div>

    <el-row style="margin-top:40px">
      <el-col :span="2" :offset="20">
        <el-upload v-if="active==0" action :show-file-list="false" :http-request="uploadTrainLog">
          <el-button>上传训练文件</el-button>
        </el-upload>
        <el-button v-if="active==1" @click="startTrain" :disabled="isTrain">训练模型</el-button>
        <el-upload
          v-if="active==4"
          action
          :show-file-list="false"
          :http-request="uploadAbnormalLog"
        >
          <el-button>上传预测文件</el-button>
        </el-upload>
      </el-col>
    </el-row>
  </el-card>
</template>

<script>
import axios from "axios";
var baseUrl = "http://localhost:5000/";

export default {
  name: "train",

  data() {
    return {
      active: 0,
      showTip1: false,
      showTip2: false,
      showTip3: false,
      isTrain: false,
      timer: null,
      timer2: null,
      progress3: 5,
      progress4: 0,
      progress5: 0,
      progress6: 0
    };
  },

  methods: {
    next() {
      if (this.active++ > 4) this.active = 0;
    },

    uploadTrainLog(param) {
      var fileObj = param.file;
      var form = new FormData();
      form.append("file", fileObj);
      axios.post(baseUrl + "uploadTrainLog", form)
        .then(res => {
          this.$message({
            message: "上传成功",
            type: "success",
            duration: 1500
          });
        })
        .catch(() => {
          this.$message({
            message: "上传失败",
            type: "error",
            duration: 1500
          });
        });
      this.next();
    },

    showTrain() {
      axios.get(baseUrl + "showTrain")
        .then(res => {
          if (res.data.step == "1") {
            this.showTip1 = true;
          }
          if (res.data.step == "2") {
            this.showTip1 = true;
            this.showTip2 = true;
          }
          if (res.data.step == "3") {
            this.showTip1 = true;
            this.showTip2 = true;
            this.showTip3 = true;
            this.progress3 = parseInt(res.data.progress);
          }
          if (res.data.step == "4") {
            this.active = 3;
            this.showTip1 = true;
            this.showTip2 = true;
            this.showTip3 = true;
            this.progress4 = parseInt(res.data.progress);
          }
          if (res.data.step == "5") {
            this.active = 4;
            this.showTip1 = true;
            this.showTip2 = true;
            this.showTip3 = true;
            this.progress5 = parseInt(res.data.progress);
          }
        })
        .catch(err => {
          this.$message({
            message: err,
            type: "error",
            duration: 1500
          });
        });
    },

    startTrain() {
      var that = this;
      this.timer = setInterval(that.showTrain, 500);
      axios
        .get(baseUrl + "startTrain")
        .then(res => {
          console.log(res);
          clearInterval(this.timer);
          this.timer = null;
          this.isTrain = false;
        })
        .catch(err => {
          this.$message({
            message: err,
            type: "error",
            duration: 1500
          });
          return;
        });
    },

    uploadAbnormalLog(param) {
      var fileObj = param.file;
      var form = new FormData();
      form.append("file", fileObj);
      axios
        .post(baseUrl + "uploadAbnormalLog", form)
        .then(res => {
          this.$message({
            message: "上传成功",
            type: "success",
            duration: 1500
          });
        })
        .catch(() => {
          this.$message({
            message: "上传失败",
            type: "error",
            duration: 1500
          });
        });
      //上传异常文件后，立即开始异常预测
      this.startPredicate();
    },

    startPredicate() {
      //this.timer2 = setInterval(this.showPredict, 200);
      axios
        .get(baseUrl + "startPredicate")
        .then(res => {
          localStorage.setItem("predictResult", JSON.stringify(res.data));
          this.$router.push({
          path: "/predict"
        });
        })
        .catch(err => {
          this.$message({
            message: err,
            type: "error",
            duration: 1500
          });
        });
    },

    showPredict() {
      this.progress6 += 2;
      if (this.progress6 >= 100) {
        clearInterval(this.timer2);
        this.$router.push({
          path: "/predict"
        });
      }
    }
  }
};
</script>

<style>
.trainCard {
  max-width: 900px;
  margin: 120px auto 0;
  padding: 0 20px;
}

.trainCard .el-steps {
  height: 480px;
}
</style>