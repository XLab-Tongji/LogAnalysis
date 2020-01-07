<template>
  <div class="logTextArea">
    <el-card class="box-card">

      <div slot="header" class="clearfix">
        <span class="span-title">预测报表</span>
      </div>

      <el-form :inline="true" size="small">
        <el-form-item>
          <i class="el-icon-d-arrow-right" />
        </el-form-item>
        <el-form-item label="日志总数:">
          <el-tag>{{result.total_log}} 条</el-tag>
        </el-form-item>
        <br/>
        <el-form-item>
          <i class="el-icon-d-arrow-right" />
        </el-form-item>
        <el-form-item label="共检测出异常:">
          <el-tag>{{result.total_abnormal_num-4}} 条</el-tag>
        </el-form-item>
        <br/>
        <el-form-item>
          <i class="el-icon-d-arrow-right" />
        </el-form-item>
        <el-form-item label="检测时间: ">
          <el-tag>{{result.consume_time.slice(0, 3)}} s</el-tag>
        </el-form-item>
        <br/>
        <!-- <el-form-item>
          <i class="el-icon-d-arrow-right" />
        </el-form-item> -->
        <el-form-item>
          <el-collapse>
            <el-collapse-item title="异常日志列表: ">
              {{result.abnormal}}
            </el-collapse-item>
          </el-collapse>
        </el-form-item>
        <br/> <br/>
        
        <el-form-item>
          <i class="el-icon-d-arrow-right" />
        </el-form-item>
        <el-form-item label="模型一检测出异常: ">
          <el-tag>{{result.model1_adnormal_num - 2}}</el-tag>
        </el-form-item>
        <br/>
        <!-- <el-form-item>
          <i class="el-icon-d-arrow-right" />
        </el-form-item> -->
        <el-form-item>
          <el-collapse>
            <el-collapse-item title="异常日志列表: ">
              {{result.model1_abnormal}}
            </el-collapse-item>
          </el-collapse>
        </el-form-item>
        <br/><br/>
        
        <el-form-item>
          <i class="el-icon-d-arrow-right" />
        </el-form-item>
        <el-form-item label="模型二检测出异常: ">
          <el-tag>{{result.model2_abnormal_num - 2}}</el-tag>
        </el-form-item>
        <br/>
        <!-- <el-form-item>
          <i class="el-icon-d-arrow-right" />
        </el-form-item> -->
        <el-form-item>
          <el-collapse>
            <el-collapse-item title="异常日志列表: ">
              {{result.model2_abnormal}}
            </el-collapse-item>
          </el-collapse>
        </el-form-item>
      </el-form>
    </el-card>
    <br/>
    <el-card class="box-card">
      <div slot="header" class="clearfix">
        <span class="span-title">异常检测阶段</span>
        <div style="float:right">
          <!-- <input /> -->
          <el-input size="mini" placeholder="输入跳转行数" v-model="input_line"  @input="handleinput"></el-input>
          <el-divider direction="vertical"></el-divider>
          <i class="el-icon-arrow-up" style="margin-left:4px" @click="pre_index"></i>
          <i class="el-icon-arrow-down" style="margin-left:8px" @click="next_index"></i>
        </div>
        <!-- <el-input-number style="float:right;"></el-input-number> -->
        <!-- <el-button style="float: right; padding: 3px 0" type="text">下一条异常</el-button> -->
      </div>
      <!-- <el-row>
        <el-col :span="4" class="headTag">异常总数:{{abnormalNum}}</el-col>
        <el-col :span="4" :offset="12" class="headTag">准确率:{{accuracy}}</el-col>
        <el-col :span="4" class="headTag">误报率:{{FN}}</el-col>
      </el-row> -->
      <el-scrollbar style="height:600px;border-radius: 5px;">
        <el-row class="lineBox" v-for="(content, index) in lines" :key="index">
          <div
            :id="'line_' + index.toString()"
            :class="model1_wrong.indexOf(index.toString()) != -1 ? 'wrongBox1' : 
              model2_wrong.indexOf(index.toString()) != -1 ? 'wrongBox1' : ''"
            style="height:100%;"
          >
            <el-col :span="3" class="lineIndex" :id="'index_'+(index).toString()">{{index}}</el-col>
            <el-col :span="21" class="lineContent">{{content}}</el-col>
          </div>
        </el-row>
      </el-scrollbar>
      <el-row style="margin-top:80px">
        <el-col :span="4" :offset="16">
          <el-button>发送事件</el-button>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script>
export default {
  name: "predict",

  data() {
    return {
      abnormalNum: 0,
      accuracy: 0.8,
      FN: 0.1,
      result: JSON.parse(localStorage.getItem("predictResult")),
      lines: [],
      model1_wrong: [],
      model2_wrong: [],
      wrong: [],
      index: 0,
      input_line:0,
      pre_line:0
    };
  },

  methods: {
    next_index() {
      if (this.index + 1 >= this.wrong.length) {
        return;
      }
      this.index += 1;
      this.goto_line(this.wrong[this.index]);
    },

    pre_index() {
      if (this.index - 1 < 0) {
        return;
      }
      this.index -= 1;
      this.goto_line(this.wrong[this.index]);
    },
    handleinput(){
      if(this.input_line>=0)
      {
        this.goto_line(this.input_line)
      }
    },
    goto_line(line) {
      document.getElementById("line_" + line.toString()).scrollIntoView();
      document.getElementById(
        "index_" + line.toString()
      ).style.backgroundColor = "#fffae3";
      document.getElementById(
        "index_" + this.pre_line.toString()
      ).style.backgroundColor = "#f0f0f0";
      this.pre_line=line;
    }
  },

  mounted() {
    this.lines = this.result.log
    this.lines.shift();
    this.model1_wrong = this.result.model1_abnormal
    this.model2_wrong = this.result.model2_abnormal
    //this.lines.shift();
    // for(var j = 0; j < this.result.abnormal.length; j++) {
    //   this.result.abnormal[j] = parseInt(this.result.abnormal[j]);
    // } 
    // this.result.abnormal.sort(function(a, b){return a - b});
    // for(var j = 0; j < this.result.abnormal.length; j++) {
    //   this.result.abnormal[j] = (this.result.abnormal[j]).toString();
    // } 
    this.wrong = this.result.abnormal
  }
};
</script>

<style>
.text {
  font-size: 14px;
}

.item {
  margin-bottom: 18px;
}

.clearfix:before,
.clearfix:after {
  display: table;
  content: "";
}
.clearfix:after {
  clear: both;
}

.box-card {
  width: 800px;
  margin: 0 auto;
}

.logTextArea {
  max-width: 800px;
  margin: 20px auto;
  padding: 20px;
}

.logTextArea .box-card i {
  cursor: pointer;
}

.logTextArea .box-card i:hover {
  color: #999999;
}

.lineIndex {
  width: 60px!important;
  height: 100%;
  line-height: 20px;
  text-align: left;
  font-size: 11px;
  color: #999999;
  background: #f0f0f0;
  padding-left: 2px;
}

.lineContent {
  font-family: consolas;
  color: #013;
  padding-left: 4px;
}

.lineBox {
  text-align: left;
  height: 20px;
  float: left;
}

.wrongBox1 {
  background: #faeae6;
}

.wrongBox2 {
  background: #ffc5b7;
}

.span-title {
  float: left;
  /* font-size:15px; */
  font-family: "Avenir", Helvetica, Arial, sans-serif;
}

.el-input {
  width: 50% !important;
}

.headTag {
  font-family: "Avenir", Helvetica, Arial, sans-serif;
  color: #013;
  box-shadow: #999999 1px 1px 4px inset;
  padding: 5px
}

.logTextArea .el-form {
  text-align: left;
  margin-left: 10px;
}

.logTextArea .el-form-item--small.el-form-item {
  margin-bottom: 8px;
}

.logTextArea .el-collapse,
.logTextArea .el-collapse .el-collapse-item,
.logTextArea .el-collapse .el-collapse-item__header {
  border: none;
}

.el-scrollbar .el-scrollbar__wrap .el-scrollbar__view{
   white-space: nowrap;
}
</style>