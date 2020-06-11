import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Scanner;
import java.io.IOException;

public class deeplog {
    public static void main(String[] args) throws Exception {
        String detect_file = "HDFS_detect.log";
        String use_model2 = "1";
        System.out.println("\nExecuting python script file now.");
        try {
            String cmds = String.format("python C:\\study\\code\\LogAnalysis\\java\\deeplog_java.py %s %s",
                    detect_file,use_model2);
            Process proc = Runtime.getRuntime().exec(cmds);
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
