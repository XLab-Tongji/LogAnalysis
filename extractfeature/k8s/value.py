with open("union.log",'r')as union:
 for line in union.readlines():
     valueline=line.split(" ")
     value=''
     value+=str(valueline[0][0:4])+" "+str(valueline[0][5:7])+" "+str(valueline[0][8:10])+" "
     value+=str(valueline[1][0:2])+" "+str(valueline[1][3:5])+" "+str(valueline[1][6:8])+" "+str(valueline[1][9:])+" "
     value+=str(valueline[5])+" "
     value+=str(valueline[7][5:7])+" "+str(valueline[7][13:15])+"\n"
     with open("value.log",'a+')as v:
         v.write(value)
