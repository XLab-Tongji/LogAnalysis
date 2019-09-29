'''
提取value函数
参数tool表示使用的工具 0为sequence 1为logcluster
参数output表示输出到的文件
'''
def valueExtract(pattern, logs, tool=0, ouput="result.txt"):
    start_char = "%"
    if tool == 1:
        start_char = "*"
    pattern_arr = pattern.split()
    values = []
    # 遍历所有日志
    for log in logs:
        log_value = []
        log_arr = log.split()
        log_index = 0
        cur_log_str = log_arr[log_index]
        last_is_pattern = False
        # 遍历模式字符串进行匹配
        for pattern_str in pattern_arr:
            if pattern_str[0] == start_char and pattern_str[-1] == start_char:
                if pattern_str[1:-1] == "msgtime":
                    cur_log_str += (" " + log_arr[log_index+1] + " " + log_arr[log_index+2])
                    log_index += 2
                elif pattern_str[1:-1] == "time":
                    # To Do
                    pass
                log_value.append(cur_log_str)
                log_index += 1
                if (log_index < len(log_arr)):
                    cur_log_str = log_arr[log_index]
                last_is_pattern = True
            elif cur_log_str.lower() == pattern_str.lower():
                log_index += 1
                if (log_index < len(log_arr)):
                    cur_log_str = log_arr[log_index]
                last_is_pattern = False
            elif len(cur_log_str) >= len(pattern_str) and cur_log_str.lower()[0:len(pattern_str)] == pattern_str.lower():
                cur_log_str = cur_log_str[len(pattern_str):]
                last_is_pattern = False
            elif last_is_pattern:
                log_index -= 1
                cur_log_str = log_value.pop()
                index = cur_log_str.find(pattern_str)
                log_value.append(cur_log_str[0:index])
                if index+len(pattern_str) == len(cur_log_str):
                    log_index += 1
                    if (log_index < len(log_arr)):
                        cur_log_str = log_arr[log_index]
                else:
                    cur_log_str = cur_log_str[index+len(pattern_str):]
        values.append(log_value)
    lines = []
    for val in values:
        lines.append(", ".join(val) + "\n")
    with open(ouput, "w") as f:
        f.writelines(lines)
    return values

if __name__ == "__main__":
    pattern = r"%msgtime% combo %string% ( pam_unix ) [ %integer% ] : %action% %status% ; logname = uid = %srcuid% euid = %integer% tty = %string% ruser = rhost = %srcuser%"
    logs = [
        "Jun 10 17:16:34 combo sshd(pam_unix)[3353]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=itsc.iasi.astral.ro",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17422]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17432]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17421]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17423]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17428]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17420]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17427]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17431]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17433]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 13 20:13:11 combo sshd(pam_unix)[17429]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=under-eepc58.kaist.ac.kr",
        "Jun 15 14:53:32 combo sshd(pam_unix)[23661]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=061092085098.ctinets.com", 
        "Jun 15 14:53:32 combo sshd(pam_unix)[23663]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=061092085098.ctinets.com",
        "Jun 15 14:53:32 combo sshd(pam_unix)[23664]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=061092085098.ctinets.com",
        "Jun 15 14:53:33 combo sshd(pam_unix)[23665]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=061092085098.ctinets.com",
    ]
    valueExtract(pattern, logs)
    print("=====done=====")