# 第二章 Jupyter notebook 

## 2.1 Tab 补全 

## 2.2 内省 

（？）：显示一些关于对象的概要信息

（？？）：显示函数的源代码


```python
b = [1, 2, 3]
```


```python
b?
```

![0201.png](attachment:0201.png)


```python
print?
```

![0202.png](attachment:0202.png)


```python
def add_numbers(a, b):
    """
    Add two numbers together
    Returns
    -------
    the sum : type of arguments
    """
    return a + b
```


```python
add_numbers?
```

![0204.png](attachment:0204.png)


```python
add_numbers??
```

![0203.png](attachment:0203.png)

与一些字符通配符结合（星号*），会显示所有匹配通配符表达式的命名


```python
import numpy as np
np.*load*?
```

![0206.png](attachment:0206.png)

## 2.3 %run命令


```python
%run ipython_script_test.py.ipynb
```


```python
c
```




    7.5




```python
result
```




    1.4666666666666666



## 2.4 Python语言基础

### 2.4.1 isinstance函数：检查一个对象是否是特定类型的实例


```python
a = 5

isinstance(a, int)
```


```python
a = 5
b = 4.5
```


```python
isinstance(a, (int, float))
```




    True




```python
isinstance(b, (int, float))
```




    True



### 2.4.2 属性与方法


```python
a = 'foo'
```


```python
getattr(a, 'split')
```




    <function str.split(sep=None, maxsplit=-1)>



### 2.4.3 isiterable()验证是否可迭代


```python
def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError: # 不可遍历
        return False
```


```python
isiterable('a string')
```




    True




```python
isiterable([1,2,3])
```




    True




```python
isiterable(5)
```




    False



### 2.4.4 is关键字：检查两个引用是否指向同一个对象


```python
a = [1, 2, 3]
```


```python
b = a
```


```python
c = list(a)
```


```python
a is b
```




    True



is 和 == 是不同的 （list函数总是创建一个新列表（即一份拷贝））


```python
a is c
```




    False




```python
a == c
```




    True



is 和 is not的常用之处是检查一个变量是否为None


```python
a = None
```


```python
a is None
```




    True




```python

```

操作符 -------描述
a / b -------a除以b
a // b -------a整除b
a ** b -------a的b次方
a & b -------a或b；对于整数则是按为AND
a | b -------

### 2.4.5 数值类型

基础数值类型int和float。

int 可以存储任意大小数字：


```python
ival = 17239871
```


```python
ival ** 6
```




    26254519291092456596965462913230729701102721



float都是双精度64位数值，它们可以用科学计数法表示：


```python
fval = 7.243
```


```python
fval2 = 6,78e-5
```

整数除法会将结果自动转型为浮点数：


```python
3 / 2
```




    1.5



### 2.4.6 字符串

对于含有换行的多行字符串，可以使用三个单引号'''或三个双引号"""


```python
c = """
一顿火锅
两串烧烤
三杯可乐
四盆龙虾
"""
```


```python
c.count('\n')
```




    5



Python的字符串是不可变的，你无法修改一个字符串


```python
a = 'this is a string'
```


```python
a[10] = 'f'
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-55-2151a30ed055> in <module>
    ----> 1 a[10] = 'f'
    

    TypeError: 'str' object does not support item assignment



```python
a
```




    'this is a string'



字符串Unicode字符的序列，可以被看作是除了列表和元组外的另一种序列：


```python
s = 'python'
```


```python
list(s)
```




    ['p', 'y', 't', 'h', 'o', 'n']




```python
s[:3]
```




    'pyt'



反斜杠符号\是一种转义符

要在字符串中写反斜杠则需要将其转义：


```python
s = '12\\24'
```


```python
print(s)
```

    12\24
    


```python
s = '12\24'
```


```python
print(s)
```

    12
    

有位N先生，

指点我说这个应该是“贪心法”

每个字符应该包含更多的字符。

编译器将程序分解成符号的方法是，

从左到右一个字符一个字符的读入，

如果该字符可能组成一个符号，

就再读入下一个字符，

判断已经读入的两个字符组成的字符串

是否可能是一个符号的组成部分；

如果可能，继续读入下一个字符，

重复上述判断，

直到读入的字符组成的字符串

不再可能组成一个有意义的符号。

这种处理方法，又称为“贪心法”，或者“大嘴法””。

字符串前面加一个前缀符号r，表明这些字符为原生字符：

r是raw的简写，表示原生的


```python
s = r'this\has\no\special\characters'
```


```python
s
```




    'this\\has\\no\\special\\characters'



字符串格式化

字符串对象拥有一个format方法


```python
template = '{0:.2f} {1:s} are worth US${2:d}'
# {0:.2f}表示将第一个参数格式化为2位小数的浮点数
# {1：s}表示将第二个参数格式化为字符串
# {2：d}表示将第三个参数格式化为整数
```


```python
template.format(4.5560, 'Argentine Pesos', 1)
```




    '4.56 Argentine Pesos are worth US$1'



### 2.4.7 字节与Unicode


```python
val = 'Joyeux Noël'
```


```python
val
```




    'Joyeux Noël'



encode方法将Unicode字符串转换成UTF-8字节：


```python
val_utf8 = val.encode('utf-8')
```


```python
val_utf8
```




    b'Joyeux No\xc3\xabl'



decode方法解码


```python
val_utf8.decode('utf-8')
```




    'Joyeux Noël'




```python
type(val_utf8)
```




    bytes




```python
type(val)
```




    str



字符串前面加前缀b来定义字符文本


```python
bytes_val = b'po ma zhang fei'
```


```python
type(bytes_val)
```




    bytes




```python
decoded = bytes_val.decode('utf-8')
```


```python
type(decoded)
```




    str



### 2.4.8 日期和时间


```python
from datetime import datetime, date, time
```


```python
dt = datetime(2011, 10, 29, 20, 30, 21)
```


```python
dt.day
```




    29




```python
dt.minute
```




    30



对于datetime实例，还可以分别用date和time方法来获取它的date和time对象：


```python
dt.date()
```




    datetime.date(2011, 10, 29)




```python
dt.time()
```




    datetime.time(20, 30, 21)



strftime方法将datetime转换成字符串


```python
dt.strftime('%m/%d/%Y %H:%M')
```




    '10/29/2011 20:30'



strptime方法将字符串转换成datetime对象


```python
datetime.strptime('20091031', '%Y%m%d')
```




    datetime.datetime(2009, 10, 31, 0, 0)



替换datetime中的一些值，比如：


```python
dt.replace(minute=0, second=0)
```




    datetime.datetime(2011, 10, 29, 20, 0)



两个不同的datetime对象会产生一个datetime.timedelta类型的对象：


```python
dt2 = datetime(2011, 11, 15, 22, 30)
```


```python
delta = dt2 - dt
```


```python
delta
```




    datetime.timedelta(days=17, seconds=7179)



输出表示为17天又7179秒


```python
type(delta)
```




    datetime.timedelta



将timedelta加到一个datetime上将产生一个新的对象


```python
dt
```




    datetime.datetime(2011, 10, 29, 20, 30, 21)




```python
dt + delta
```




    datetime.datetime(2011, 11, 15, 22, 30)



### 2.4.9 三元表达式

Python中允许你将一个if-else代码联合起来，语法如下：

value = true-expr if condition else false-expr

它与一下详细的代码效果一致：

if condition:

----value = true-expr

else:
   
----value = false-expr


```python
x = 5
```


```python
'Non-negative' if x >= 0 else 'Negative'
```




    'Non-negative'


