import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Scanner;

public class deeplog {
    public static void main(String[] args) throws Exception {
        Scanner input = new Scanner(System.in);
        // 在同一行输入两个数字，用空格分开，作为传入Python代码的命令行参数
        String detect_file = "HDFS_detect.log";
        String detect_abnormal_label = "HDFS_detect_abnormal_label.txt";
        System.out.println("\nExecuting python script file now.");
        // 定义传入Python脚本的命令行参数，将参数放入字符串数组里
        String cmds = String.format("python C:\\study\\code\\LogAnalysis\\java\\deeplog_java.py %s %s",
                detect_file,detect_abnormal_label);
        // 执行CMD命令
        Process pcs = Runtime.getRuntime().exec(cmds);
        pcs.waitFor();
        // 定义Python脚本的返回值
        String result = null;
        // 获取CMD的返回流
        BufferedInputStream in = new BufferedInputStream(pcs.getInputStream());
        // 字符流转换字节流
        BufferedReader br = new BufferedReader(new InputStreamReader(in));
        // 这里也可以输出文本日志
        String lineStr = null;
        while ((lineStr = br.readLine()) != null) {
            System.out.println(lineStr);
        }
        // 关闭输入流
        br.close();
        in.close();
    }
}
