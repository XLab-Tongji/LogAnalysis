#处理value.log  转换为时间差
from datetime import datetime 
with open("value.log","r")as valuedeal:
    values=valuedeal.readlines()

dealvalue=[]
for i in range(len(values)-1):
    v1=values[i][:-1].split(" ")
    v2=values[i+1][:-1].split(" ")
    time1=datetime(int(v1[0]),int(v1[1]),int(v1[2]),hour=int(v1[3]),minute=int(v1[4]),second=int(v1[5]),microsecond=int(v1[6]))
    time2=datetime(int(v2[0]),int(v2[1]),int(v2[2]),hour=int(v2[3]),minute=int(v2[4]),second=int(v2[5]),microsecond=int(v2[6]))
    timedays=(time2-time1).days
    timeseconds=(time2-time1).seconds
    timemsonds=(time2-time1).microseconds
    v=str(timedays)+" "+str(timeseconds)+" "+str(timemsonds)+" "+v1[7]+" "+v1[8]+" "+v1[9]+"\n"
    dealvalue.append(v)

with open("dealedvalue.log","a+")as dvalue:
    for i in dealvalue:
        dvalue.write(i)