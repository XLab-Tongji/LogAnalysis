### 						Sequence说明文档

​	Sequence是一个开源的基于go语言的高性能的顺序日志扫描仪、分析器和解析器。它按顺序遍历日志消息，解析有意义的部分，而不使用正则表达式。它可以每秒解析超过100000条消息，而无需按日志源类型分离解析规则。

##### 一.使用限制

sequence不处理多行日志。每条日志消息必须显示为一行。因此，如果有多行日志，则必须首先将其转换为单行。

##### 二.使用说明

###### 2.1主要命令

2.1.1scan   扫描将标记日志文件或消息并输出tokens列表

例：

go run sequence.go scan -m "jan 14 10:15:56 testserver sudo:    gonner : tty=pts/3 ; pwd=/home/gonner ; user=root ; command=/bin/su - ustream"

输出：

```
#   0: { Field="%funknown%", Type="%ts%", Value="jan 14 10:15:56" }
#   1: { Field="%funknown%", Type="%literal%", Value="testserver" }
#   2: { Field="%funknown%", Type="%literal%", Value="sudo" }
#   3: { Field="%funknown%", Type="%literal%", Value=":" }
#   4: { Field="%funknown%", Type="%literal%", Value="gonner" }
#   5: { Field="%funknown%", Type="%literal%", Value=":" }
#   6: { Field="%funknown%", Type="%literal%", Value="tty" }
#   7: { Field="%funknown%", Type="%literal%", Value="=" }
#   8: { Field="%funknown%", Type="%string%", Value="pts/3" }
#   9: { Field="%funknown%", Type="%literal%", Value=";" }
#  10: { Field="%funknown%", Type="%literal%", Value="pwd" }
#  11: { Field="%funknown%", Type="%literal%", Value="=" }
#  12: { Field="%funknown%", Type="%string%", Value="/home/gonner" }
#  13: { Field="%funknown%", Type="%literal%", Value=";" }
#  14: { Field="%funknown%", Type="%literal%", Value="user" }
#  15: { Field="%funknown%", Type="%literal%", Value="=" }
#  16: { Field="%funknown%", Type="%string%", Value="root" }
#  17: { Field="%funknown%", Type="%literal%", Value=";" }
#  18: { Field="%funknown%", Type="%literal%", Value="command" }
#  19: { Field="%funknown%", Type="%literal%", Value="=" }
#  20: { Field="%funknown%", Type="%string%", Value="/bin/su" }
#  21: { Field="%funknown%", Type="%literal%", Value="-" }
#  22: { Field="%funknown%", Type="%literal%", Value="ustream" }

```



2.1.2analyze  analyze将分析日志文件并输出与所有日志消息匹配的模式列表

例：

go run sequence.go analyze -i ../../data/sshd.all -o sshd.analyze

输出：Analyzed 212897 messages, found 45 unique patterns, 45 are new.：

 

2.1.3parse   parse将解析一个日志文件，并为每个日志消息输出一个解析的tokens列表

###### 2.2主要参数

-h, --help: sequence的帮助信息

-i, --infile="": 输入的日志文件

-o, --outfile="": 输出pattern的输出文件

-m, --msg="": 单条日志

##### 三.算法流程

###### 3.1算法简介

它基于一个基本概念，对于多个日志消息，如果同一个位置共享一个相同的父节点和一个相同的子节点，然后在这个位置可能是变量字符串，这意味着我们可以提取它。

例如，查看以下两条messages：

Jan 12 06:49:42 irc sshd[7034]: Accepted password for root from 218.161.81.238 port 4228 ssh2

Jan 12 14:44:48 jlz sshd[11084]: Accepted publickey for jlz from 76.21.0.16 port 36609 ssh2

每条message的第一个token是一个时间戳，并且第三个token是一个字面量sshd,对于字面量irz和jlz他们共享同一个父节点时间戳，共享同一个孩子节点sshd, 这意味着在这两者之间的token也就是每个消息中的第二个令牌，可能表示此message类型中的变量token。在这种情况下，“irc”和“jlz”碰巧表示系统日志主机。

再往下看消息，文字“password”和“publickey”也共享一个父节点，“accepted”，和一个孩子节点“for”。所以这意味着在此位置的token也是一个变量token（tokenstring类型）。

可以在这两个message中找到共享共同父和子的几个token，这意味着可以提取这些token中的每一个。最后，我们可以确定将两者匹配的单个模式是：

%time% %string% sshd [ %integer% ] : Accepted %string% for %string% from %ipv4% port %integer% ssh2

如果稍后我们将另一条message添加到此组合中：

Jan 12 06:49:42 irc sshd[7034]: Failed password for root from 218.161.81.238 port 4228 ssh2

第一个message中的“Accepted”字和第3条消息中的“failed”共享一个公共父级“：”和一个公共子级“password”，因此它将确定此位置的标记也是一个变量标记。匹配模式将是：

%time% %string% sshd [ %integer% ] : %string% %string% for %string% from %ipv4% port %integer% ssh2

###### 3.2程序命令analyze算法流程图

![img](<https://github.com/XLab-Tongji/LogAnalysis/blob/master/Docs/pics/sequence.png>)

 

##### 四.主要函数简介

###### 4.1func (this *Scanner) Scan(s string) (Sequence, error)

scan返回所提供数据字符串的序列或token列表。

在scan中函数中，对传入的字符串遍历调用Tokenize函数。

在Tokenize函数中，对于字符串先跳过前导空格，接下来看这是不是一个标记标记，包含在两个“%”字符中，其至少还剩2个字符，第一个字符是“%”，然后调用scanToken函数，返回最后都得到的sequence列表。

###### 4.2func (this *Analyzer) Add(seq Sequence) error

add将单个message序列添加到分析树中。此时，它不会确定token是否共享一个公共父级或子级。在所有的序列都被添加之后，就应该调用 Finalize() .

在Add函数中，会先调用markSequenceKV函数，识别所有的等号，将等号之前的token类型置为key，等号之后的非单个字符置为value,然后创建一个足够大的二维分析节点数组和一个集合类型的一维数组，然后遍历sequence,判断token是否夹在两个%之间，然后进行 switch操作

如果tag是TagUnknown,这意味着tag是一个被了解到的tag类型，将它加入到二维数组的i，int(token.Tag)位置。

如果token.Type不是TokenUnknown并且token.Type != TokenLiteral  这意味着这是一个已知的token type但又不是Literal，即他可能包含不同的值将它加入到二维数组的i，Literal位置。

如果token.Tag == TagUnknown && token.Type == TokenLiteral 表示这是我们从message中解析的某种类型的字符串，不能确定它是一个变量还是字面量，如果不在map数组中，将其放入。

最后使用位集来跟踪父子关系。在这种情况下，为当前节点的索引设置父位，并设置父节点索引的子位

###### 4.3func (this *Analyzer) Finalize() error

 Finalize将通过分析树并确定哪些token共享公共父和子，合并所有共享至少1个父节点和1个子节点的所有节点，最后压缩树并删除所有死节点。

  Finalize函数建立两个for循环去遍历每一个二维数组的节点，对于这一个节点再去遍历其在这一行之后的节点，看那些能与之合并。得到能合并的序列将之合并并删除死节点。

###### 4.4func (this *Analyzer) Analyze(seq Sequence) (Sequence, error)

analyze分析提供的message序列，并返回与此消息匹配的唯一模式

analyze函数先调用analyzeMessage函数，将传入的sequence与二维数组中对应位置的进行对比，遍历所有的node，如果节点的类型等于token的类型并且token类型不是string或literal,则认为这是一个已经识别的token，如时间戳，认为完全匹配，权值加2,

如果node和token的类型都是literal并且值也相同，如为常量”creat”， 认为完全匹配，权值加2,如果token是一个变量，认为完全匹配，如果node.Type == TokenString && token.Type == TokenLiteral 认为是部分匹配，权值加1.

最后这样会在二维数组中找到一个权值最高的路径即是它的pattern

 

 

 

