# 第四章 NumPy基础：数组与向量化

下面的例子来展现 NumPy 的不同，

假设一个 NumPy 数组包含100万个整数，还有一个同样数据内容的 Python 列表：


```python
import numpy as np
```


```python
my_arr = np.arange(1000000)
```


```python
my_list = list(range(1000000))
```

现在我们同时对每个序列乘以2:


```python
%time for _ in range(10): my_arr2 = my_arr * 2
%time for _ in range(10): my_list2 = [x * 2 for x in my_list]
```

    Wall time: 24 ms
    Wall time: 854 ms
    

NumPy 的方法比 Python 方法要快10到100倍

使用的内存也更少

## 4.1 NumPy ndarray：多维数组对象

NumPy 的核心特征之一就是 N-维数组对象——ndarray。

ndarray 是Python 中一个快速、灵活的大型数据集容器。

感受下 NumPy 如何使用类似于 Python 内建对象的标量计算语法进行批量计算:


```python
import numpy as np
# 尽量使用标准的NumPy导入方式
# numpy 这个命名空间包含了大量与Python内建函数重名的函数（比如 min 和 max）

# 生成随机 2*3 数组
data = np.random.randn(2, 3)
```


```python
data
```




    array([[ 1.527459  , -0.52051521, -0.63328186],
           [ 0.58732376,  0.10422902, -2.20845316]])



然后给 data 加上一个数学操作：


```python
data * 10
```




    array([[ 15.27459   ,  -5.20515206,  -6.3328186 ],
           [  5.87323758,   1.04229018, -22.08453162]])




```python
data + data
```




    array([[ 3.054918  , -1.04103041, -1.26656372],
           [ 1.17464752,  0.20845804, -4.41690632]])



一个 ndarray 是一个通用的多维同类数据容器，也就是说，它包含的每一个元素均为相同类型。

每一个数组都有一个 shape 属性，表征数组每一维度的数量

每一个数组都有一个 dtype 属性，来描述数组的数据类型


```python
data.shape
```




    (2, 3)




```python
data.dtype
```




    dtype('float64')



### 4.1.1 生成 ndarray

array 函数：接受任意的序列型对象

例如：列表的转换：


```python
data = [6, 7.5, 8, 0, 1]
arr1 = np.array(data)

arr1
```




    array([6. , 7.5, 8. , 0. , 1. ])



嵌套序列，例如同等长度的列表，将会自动转换成多维数组：


```python
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)

arr2
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])



通过 ndim 和 shape 属性来确认这一点：


```python
arr2.ndim
```




    2




```python
arr2.shape
```




    (2, 4)



除非显示的指定，否则 np.array 会自动推断数组的数据类型。并存储在一个特殊的元数据的 dtype 中：


```python
arr1.dtype
```




    dtype('float64')




```python
arr2.dtype
```




    dtype('int32')



除 np.array，还有很多其他函数可以创建新数组。


```python
# zeros

np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
np.zeros((3, 6))
```




    array([[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]])




```python
# ones

np.ones((3, 6))
```




    array([[1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.]])




```python
# empty 创建一个没有初始化数值的数组

np.empty((2, 3, 2)) # 高维数组
```




    array([[[1.02977135e-311, 3.16202013e-322],
            [0.00000000e+000, 0.00000000e+000],
            [0.00000000e+000, 9.23471186e-071]],
    
           [[5.04070495e+174, 1.65948984e-076],
            [8.77692562e-071, 5.99410436e-066],
            [5.78751557e+174, 4.91225461e-062]]])



arange 是 Python 内建函数 range 的数组版：


```python
np.arange(15)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])



#### 标准数组的生成函数（数据类型默认为 float64（浮点型））

array 接受：列表元组数组等序列，复制

asarry 如果已经是ndarray则不再复制

arange Python内建函数range的数组版

ones 接受形状，全是1

ones_like 接受数组，生成形状一样的全是1 

zeros 全是0

zeros_like 接受数组，生成形状一样的全是0

empty 接受形状，未初始化

empty_like 

full 指定数值，指定类型

full_like

eye，identity n*n,对角线为1，其余位置为0

### 4.1.2 数据类型

数据类型，即 dtype，是一个特殊的对象

它包含了 ndarray 需要为某一种类型数据所申明的内存块信息

也称为元数据，即表示数据的数据


```python
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
print(arr1.dtype)
print(arr2.dtype)
```

    float64
    int32
    

dtype 是 NumPy 能够与其他系统数据灵活交互的原因。

你可以使用 astype 方法显是地转换数组的数据类型：


```python
arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)

float_arr = arr.astype(np.float64)
print(float_arr.dtype)
```

    int32
    float64
    


```python
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])

print(arr)

arr.astype(np.int32)## 转为整型
```

    [ 3.7 -1.2 -2.6  0.5 12.9 10.1]
    




    array([ 3, -1, -2,  0, 12, 10])




```python
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)

numeric_strings.astype(float)## 转为浮点型
```




    array([ 1.25, -9.6 , 42.  ])



可以使用另一个数组的 dtype 属性：


```python
int_array = np.arange(10)

calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)

int_array.astype(calibers.dtype)# 第二个的类型浮点型，指定给第一个数组
```




    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])



也可以使用类型代码来传入数据类型：


```python
empty_uint32 = np.empty(8, dtype='u4')
empty_uint32
```




    array([         0, 1075314688,          0, 1075707904,          0,
           1075838976,          0, 1072693248], dtype=uint32)



### 4.1.3 NumPy 数组算术


```python
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
arr * arr
```




    array([[ 1.,  4.,  9.],
           [16., 25., 36.]])




```python
arr - arr
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
1 / arr
```




    array([[1.        , 0.5       , 0.33333333],
           [0.25      , 0.2       , 0.16666667]])




```python
arr ** 0.5
```




    array([[1.        , 1.41421356, 1.73205081],
           [2.        , 2.23606798, 2.44948974]])




```python
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
arr2
```




    array([[ 0.,  4.,  1.],
           [ 7.,  2., 12.]])




```python
arr2 > arr # 对应位置 比较大小 
```




    array([[False,  True, False],
           [ True, False,  True]])



### 4.1.4 基础索引与切片

一维数组比较简单，看起来和 Python 的列表类似：


```python
arr = np.arange(10)
```


```python
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
arr[5]
```




    5




```python
arr[5: 8]
```




    array([5, 6, 7])




```python
arr[5: 8] = 12 # 这一步在Python的列表中是无法实现的，只能实现一个迭代
```


```python
arr
```




    array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])



区别于 Python 的内建列表，数组的切片是原数组的视图。

这意味着数据不是被复制了，任何对于视图的修改都会反映到原数组上。


```python
arr_slice = arr[5: 8]
```


```python
arr_slice
```




    array([12, 12, 12])



当我们改变arr_slice，变化也会体现在数组上：


```python
arr_slice[1] = 12345
arr
```




    array([    0,     1,     2,     3,     4,    12, 12345,    12,     8,
               9])




```python
arr_slice[:] = 64
arr
```




    array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])



如果你还是想要一份数组切片的拷贝而不是一份视图的话，你就必须显式地复制这个数组，例如 arr[5: 8].copy()

对于更高为维数的数组：


```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]
```




    array([7, 8, 9])



单个元素可以通过递归的方式获得：


```python
arr2d[0][2] # 可以理解为在嵌套里查找 
```




    3




```python
arr2d[0, 2] # 可以理解为行和列
```




    3




```python
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[0]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
old_values = arr3d[0].copy()
```


```python
arr3d[0] = 42
arr3d
```




    array([[[42, 42, 42],
            [42, 42, 42]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[0] = old_values
arr3d
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 12]]])




```python
arr3d[1, 0]
```




    array([7, 8, 9])



上面的表达式可以分解为下面两步：


```python
x = arr3d[1]
x
```




    array([[ 7,  8,  9],
           [10, 11, 12]])




```python
x[0]
```




    array([7, 8, 9])



#### 4.1.4.1 数组的切片索引


```python
arr
```




    array([ 0,  1,  2,  3,  4, 64, 64, 64,  8,  9])




```python
arr[1:6] # 一维，就是从1号到6号止
```




    array([ 1,  2,  3,  4, 64])




```python
arr2d
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
arr2d[:2]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
arr2d[:2, 1:]
```




    array([[2, 3],
           [5, 6]])




```python
arr2d[1, :2]
```




    array([4, 5])




```python
arr2d[:2, 2]
```




    array([3, 6])




```python
arr2d[:, :1]
```




    array([[1],
           [4],
           [7]])



当然对切片表达式赋值是，整个切片都会重新赋值：


```python
arr2d[:2, 1:] = 0
```


```python
arr2d
```




    array([[1, 0, 0],
           [4, 0, 0],
           [7, 8, 9]])



### 4.1.5 布尔索引


```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names
```




    array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')




```python
data
```




    array([[ 1.36226215, -1.69811467, -1.53070126, -0.80451784],
           [ 0.08907323,  1.52510099, -1.00506474, -0.05057588],
           [ 1.6129536 , -1.82848812,  0.95213646, -0.21713875],
           [-0.59978703,  0.55353455, -0.55569732, -0.10193827],
           [ 1.60661696,  1.5575756 ,  0.65237786, -0.47599793],
           [-2.16383688, -0.04137054,  1.23878676, -0.68583057],
           [ 0.4157422 ,  1.31276116,  0.25406215,  0.06513469]])




```python
names == 'Bob'
```




    array([ True, False, False,  True, False, False, False])




```python
data[names == 'Bob'] 
# 这组布尔值传入data
# 作为索引取数组的值
```




    array([[ 1.36226215, -1.69811467, -1.53070126, -0.80451784],
           [-0.59978703,  0.55353455, -0.55569732, -0.10193827]])




```python
data[names == 'Bob', 2:] 
# 竖着按布尔值取
# 横着按切片索引
```




    array([[-1.53070126, -0.80451784],
           [-0.55569732, -0.10193827]])




```python
data[names == 'Bob', 3]
```




    array([-0.80451784, -0.10193827])




```python
names != 'Bob'
```




    array([False,  True,  True, False,  True,  True,  True])




```python
data[~(names == 'Bob')] ## Bob之外的数据
```




    array([[ 0.08907323,  1.52510099, -1.00506474, -0.05057588],
           [ 1.6129536 , -1.82848812,  0.95213646, -0.21713875],
           [ 1.60661696,  1.5575756 ,  0.65237786, -0.47599793],
           [-2.16383688, -0.04137054,  1.23878676, -0.68583057],
           [ 0.4157422 ,  1.31276116,  0.25406215,  0.06513469]])




```python
cond = names == 'Bob' # 一个意思
data[~cond]
```




    array([[ 0.08907323,  1.52510099, -1.00506474, -0.05057588],
           [ 1.6129536 , -1.82848812,  0.95213646, -0.21713875],
           [ 1.60661696,  1.5575756 ,  0.65237786, -0.47599793],
           [-2.16383688, -0.04137054,  1.23878676, -0.68583057],
           [ 0.4157422 ,  1.31276116,  0.25406215,  0.06513469]])




```python
mask = (names == 'Bob') | (names == 'Will') # |或, &和
mask
```




    array([ True, False,  True,  True,  True, False, False])



注意：Python 的关键字 and 和 or 对布尔值数组并没有用，一定要使用 &（and）和 | （or）


```python
data[mask]
```




    array([[ 1.36226215, -1.69811467, -1.53070126, -0.80451784],
           [ 1.6129536 , -1.82848812,  0.95213646, -0.21713875],
           [-0.59978703,  0.55353455, -0.55569732, -0.10193827],
           [ 1.60661696,  1.5575756 ,  0.65237786, -0.47599793]])




```python
data_2 = np.random.randn(7, 4)
```


```python
data_2[data_2 < 0] = 0 ## 判断并赋值，这个本质上也是用了布尔值
data_2
```




    array([[0.        , 0.        , 0.7192389 , 0.        ],
           [0.7217638 , 0.        , 0.        , 0.78929928],
           [1.53556079, 0.        , 0.67078749, 0.        ],
           [0.        , 0.        , 0.        , 1.33226459],
           [1.65547905, 0.        , 0.        , 0.86242327],
           [1.23279714, 2.14372266, 0.        , 1.31395484],
           [0.19152459, 0.        , 0.59491741, 0.        ]])




```python
data_3 = np.random.randn(7, 4)
```


```python
data_3[names != 'Joe'] = 7
data_3
```




    array([[ 7.        ,  7.        ,  7.        ,  7.        ],
           [ 0.40445394, -0.80640425,  0.71811874, -0.72069617],
           [ 7.        ,  7.        ,  7.        ,  7.        ],
           [ 7.        ,  7.        ,  7.        ,  7.        ],
           [ 7.        ,  7.        ,  7.        ,  7.        ],
           [-0.16868219, -0.87383631,  0.37598706,  0.15223421],
           [ 0.69634151, -1.00877583, -0.11742624,  1.57075533]])



### 4.1.6 神奇索引

神奇索引是 NumPy 中的术语，用于描述使用整数数组进行数据索引。


```python
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i 
    ## 如arr[0]=0，就是0-7行里面的第0行，全部元素等于0
arr
```




    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [2., 2., 2., 2.],
           [3., 3., 3., 3.],
           [4., 4., 4., 4.],
           [5., 5., 5., 5.],
           [6., 6., 6., 6.],
           [7., 7., 7., 7.]])



为了选出一个符合特定顺序的子集，你传递一个包含指明所需顺序的<span class="burk">列表</span>或<span class="burk">数组</span>来完成：


```python
arr[[4, 3, 0, 6]] 
## 用列表或数组做索引
## 取的就是相应的行
## 这里取的是axis=0</div><i class="fa fa-lightbulb-o "></i>
```




    array([[4., 4., 4., 4.],
           [3., 3., 3., 3.],
           [0., 0., 0., 0.],
           [6., 6., 6., 6.]])




```python
arr[[-3, -5, -7]]
## 倒数也可以
```




    array([[5., 5., 5., 5.],
           [3., 3., 3., 3.],
           [1., 1., 1., 1.]])




```python
arr = np.arange(32).reshape((8, 4))
arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])




```python
arr[[1, 5, 7, 2], [0, 3, 1, 2]] # 先取行，再取列
```




    array([ 4, 23, 29, 10])




```python
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]] # 先取出子集，再对子集取全部行和相应的列
```




    array([[ 4,  7,  5,  6],
           [20, 23, 21, 22],
           [28, 31, 29, 30],
           [ 8, 11,  9, 10]])



<div class="burk">
请牢记神奇索引与切片不同，它总是将数据复制到一个新的数组中</div><i class="fa fa-lightbulb-o "></i>

### 4.1.7 数组转置和换轴

使用 .T 换轴是一个特殊案例。


```python
arr = np.arange(15).reshape((3, 5))
arr
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
arr.T
```




    array([[ 0,  5, 10],
           [ 1,  6, 11],
           [ 2,  7, 12],
           [ 3,  8, 13],
           [ 4,  9, 14]])




```python
arr_6 = np.arange(6).reshape((2, 3))
arr_6
```




    array([[0, 1, 2],
           [3, 4, 5]])




```python
arr_6.T
```




    array([[0, 3],
           [1, 4],
           [2, 5]])



计算矩阵内积会使用 np.dot:


```python
np.dot(arr_6, arr_6.T)
```




    array([[ 5, 14],
           [14, 50]])



对于高维度的数组，transpose 方法可以接受包含轴编号的元组，用于置换轴：

补充：

轴用来为超过一维的数组定义的属性，二维数据拥有两个轴：

第0轴沿着行的垂直往下，第1轴沿着列的方向水平延伸


```python
arr_7 = np.arange(24).reshape((2, 3, 4))
arr_7
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
arr_7.transpose((1, 0, 2)) # 将元素坐标（a, b, c）换成（b, a, c）
```




    array([[[ 0,  1,  2,  3],
            [12, 13, 14, 15]],
    
           [[ 4,  5,  6,  7],
            [16, 17, 18, 19]],
    
           [[ 8,  9, 10, 11],
            [20, 21, 22, 23]]])



ndarray 中有一个 swapaxes 方法，该方法接受一堆轴编号为参数，并对轴进行调整用于重组数据：


```python
arr_7
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
arr_7.swapaxes(1, 2) # (a, b, c)换成（a, c, b）
```




    array([[[ 0,  4,  8],
            [ 1,  5,  9],
            [ 2,  6, 10],
            [ 3,  7, 11]],
    
           [[12, 16, 20],
            [13, 17, 21],
            [14, 18, 22],
            [15, 19, 23]]])



swapaxes 返回的是数据的视图，而没有对数据进行复制。

## 4.2 通用函数：快速的逐元素数组函数

通用函数，也可以成为 ufunc， 是一种在 ndarray 数据中进行逐元素操作的函数。

有很多 ufunc 是简单的逐元素转换，比如 sqrt 或 exp 函数：


```python
arr = np.arange(10)
```


```python
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.sqrt(arr)
```




    array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
           2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])




```python
np.exp(arr)
```




    array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
           5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
           2.98095799e+03, 8.10308393e+03])



这些都是所谓的一元通用函数。

还有一些函数，比如 add 或 maximum 则会接收两个数组并返回一个数组作为结果，因此称为二元通用函数：


```python
x = np.random.randn(8)
y = np.random.randn(8)
```


```python
x
```




    array([ 0.61090096,  0.59892697,  0.6278289 , -0.40773563, -0.10761388,
            1.54215274, -0.1015939 , -2.01282381])




```python
y
```




    array([-0.3554516 ,  0.81646829, -0.06717243, -2.10134621, -0.305811  ,
           -0.16315587,  2.18749266,  0.54107663])




```python
np.maximum(x, y) # 逐个将 x 和 y 中的元素的最大值计算出来
```




    array([ 0.61090096,  0.81646829,  0.6278289 , -0.40773563, -0.10761388,
            1.54215274,  2.18749266,  0.54107663])



也有一些通用函数返回多个数组。

比如 modf，是 Python 内建函数 divmod 的向量化版本。

它返回一个浮点值数组的小数部分和整数部分：


```python
arr = np.random.randn(7) * 5
arr
```




    array([-3.71520839,  6.16905275, -2.19476926, -8.78335329,  1.31173405,
            2.61406342, -1.38248396])




```python
remainder, whole_part = np.modf(arr)
```


```python
remainder
```




    array([-0.71520839,  0.16905275, -0.19476926, -0.78335329,  0.31173405,
            0.61406342, -0.38248396])




```python
whole_part
```




    array([-3.,  6., -2., -8.,  1.,  2., -1.])



通用函数接受一个可选参数 out，允许对数组按位置操作：


```python
arr
```




    array([-3.71520839,  6.16905275, -2.19476926, -8.78335329,  1.31173405,
            2.61406342, -1.38248396])




```python
np.sqrt(arr)
```

    D:\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in sqrt
      """Entry point for launching an IPython kernel.
    




    array([       nan, 2.48375779,        nan,        nan, 1.14530959,
           1.61680655,        nan])




```python
arr # 原数组为改变
```




    array([-3.71520839,  6.16905275, -2.19476926, -8.78335329,  1.31173405,
            2.61406342, -1.38248396])




```python
np.sqrt(arr, arr)
```

    D:\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: RuntimeWarning: invalid value encountered in sqrt
      """Entry point for launching an IPython kernel.
    




    array([       nan, 2.48375779,        nan,        nan, 1.14530959,
           1.61680655,        nan])




```python
arr # 将计算的结果赋值给原数组
```




    array([       nan, 2.48375779,        nan,        nan, 1.14530959,
           1.61680655,        nan])



### 一元通用函数

abs、fabs 整数 浮点数取绝对值

sqrt 平方根

square 平方

exp e的n次方

log、log10、log2、log1p

sign 符号值，1，0，-1，正数0负数

ceil 大于等于给定数字的最小整数

floor 小于给定数字的最大整数

rint 保留整数，dtype不变

modf 小数部分与整数部分分开，分别作为数组返回

isnan 判断返回的东西，是不是nan，nan不是数值，得到布尔值

isfinite、isinf 判断是否有限，是否无限，得到布尔值

cos、cosh、sin、sinh、tan、tanh 三角函数

arccos、arccosh、arcsin、arcsinh、arctan、arctanh 反三角函数

logical_not 对数组的元素取反，相当于~arr

### 二元通用函数

add 数组的对应元素相加

subtract 第二个数组中提出第一个数组中的元素

multiply 对应元素相乘

divide，floor_divide 除或整除

power [1,2],[3,4] 得到 [1**3=1, 2**4=16]

maximum, fmax 最大值，fmax忽略nan

minimum，fmin 最小值，fmin忽略nan

mod 除法余数

copysign [2,1] [-1,2] 变成 [-2,1]

greater，greater_equal，less，less_equal，equal，not_equal 大于等于啥的

logical_and，logical_or，logical_xor 逻辑操作

## 4.3 使用数组进行面向数组编程

假设我们想要对一些网格数据来计算函数 sqrt(x^2 + y^2) 值。

np.meshgrid 函数接受两个以为数组，并根据两个数组的所有（x， y）对生成一个二维矩阵：


```python
points = np.arange(-5, 5, 0.1) 
## 100 equally spaced points
## 从-5到5，精度0.1，共100个数
points
```




    array([-5.00000000e+00, -4.90000000e+00, -4.80000000e+00, -4.70000000e+00,
           -4.60000000e+00, -4.50000000e+00, -4.40000000e+00, -4.30000000e+00,
           -4.20000000e+00, -4.10000000e+00, -4.00000000e+00, -3.90000000e+00,
           -3.80000000e+00, -3.70000000e+00, -3.60000000e+00, -3.50000000e+00,
           -3.40000000e+00, -3.30000000e+00, -3.20000000e+00, -3.10000000e+00,
           -3.00000000e+00, -2.90000000e+00, -2.80000000e+00, -2.70000000e+00,
           -2.60000000e+00, -2.50000000e+00, -2.40000000e+00, -2.30000000e+00,
           -2.20000000e+00, -2.10000000e+00, -2.00000000e+00, -1.90000000e+00,
           -1.80000000e+00, -1.70000000e+00, -1.60000000e+00, -1.50000000e+00,
           -1.40000000e+00, -1.30000000e+00, -1.20000000e+00, -1.10000000e+00,
           -1.00000000e+00, -9.00000000e-01, -8.00000000e-01, -7.00000000e-01,
           -6.00000000e-01, -5.00000000e-01, -4.00000000e-01, -3.00000000e-01,
           -2.00000000e-01, -1.00000000e-01, -1.77635684e-14,  1.00000000e-01,
            2.00000000e-01,  3.00000000e-01,  4.00000000e-01,  5.00000000e-01,
            6.00000000e-01,  7.00000000e-01,  8.00000000e-01,  9.00000000e-01,
            1.00000000e+00,  1.10000000e+00,  1.20000000e+00,  1.30000000e+00,
            1.40000000e+00,  1.50000000e+00,  1.60000000e+00,  1.70000000e+00,
            1.80000000e+00,  1.90000000e+00,  2.00000000e+00,  2.10000000e+00,
            2.20000000e+00,  2.30000000e+00,  2.40000000e+00,  2.50000000e+00,
            2.60000000e+00,  2.70000000e+00,  2.80000000e+00,  2.90000000e+00,
            3.00000000e+00,  3.10000000e+00,  3.20000000e+00,  3.30000000e+00,
            3.40000000e+00,  3.50000000e+00,  3.60000000e+00,  3.70000000e+00,
            3.80000000e+00,  3.90000000e+00,  4.00000000e+00,  4.10000000e+00,
            4.20000000e+00,  4.30000000e+00,  4.40000000e+00,  4.50000000e+00,
            4.60000000e+00,  4.70000000e+00,  4.80000000e+00,  4.90000000e+00])




```python
xs, ys = np.meshgrid(points, points)
ys
```




    array([[-5. , -5. , -5. , ..., -5. , -5. , -5. ],
           [-4.9, -4.9, -4.9, ..., -4.9, -4.9, -4.9],
           [-4.8, -4.8, -4.8, ..., -4.8, -4.8, -4.8],
           ...,
           [ 4.7,  4.7,  4.7, ...,  4.7,  4.7,  4.7],
           [ 4.8,  4.8,  4.8, ...,  4.8,  4.8,  4.8],
           [ 4.9,  4.9,  4.9, ...,  4.9,  4.9,  4.9]])




```python
xs
```




    array([[-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           ...,
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9],
           [-5. , -4.9, -4.8, ...,  4.7,  4.8,  4.9]])




```python
z = np.sqrt(xs ** 2 + ys ** 2)
z
```




    array([[7.07106781, 7.00071425, 6.93108938, ..., 6.86221539, 6.93108938,
            7.00071425],
           [7.00071425, 6.92964646, 6.85930026, ..., 6.78969808, 6.85930026,
            6.92964646],
           [6.93108938, 6.85930026, 6.7882251 , ..., 6.71788657, 6.7882251 ,
            6.85930026],
           ...,
           [6.86221539, 6.78969808, 6.71788657, ..., 6.64680374, 6.71788657,
            6.78969808],
           [6.93108938, 6.85930026, 6.7882251 , ..., 6.71788657, 6.7882251 ,
            6.85930026],
           [7.00071425, 6.92964646, 6.85930026, ..., 6.78969808, 6.85930026,
            6.92964646]])




```python
import matplotlib.pyplot as plt

plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()

plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
```




    Text(0.5, 1.0, 'Image plot of $\\sqrt{x^2 + y^2}$ for a grid of values')




![png](output_176_1.png)


### 4.3.1 将条件逻辑作为数组操作

np.where 函数是三元表达式 x if condition else y 的向量化版本。


```python
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
```

要求condition中的元素为True时，我们取 xarr 中的对应元素值，否则取 yarr 中的元素。

首先使用列表推导式来完成：


```python
result = [(x if c else y) ## if满足，要x，if不满足，要y
          for x, y, c in zip(xarr, yarr, cond)]
result
```




    [1.1, 2.2, 1.3, 1.4, 2.5]



这样就有产生几个问题。

首先，如果数组很大的话，速度会很慢（因为所有的工作都是通过解释器Python代码完成）。

其次，当数组为多维时，就无法奏效了。

而使用 np.where 时，就可以非常简单地完成：


```python
result = np.where(cond, xarr, yarr) ## 判断，如果是xarr，如果不是yarr
result
```




    array([1.1, 2.2, 1.3, 1.4, 2.5])



np.where 的第二个和第三个参数并不是需要数组，它们可以是标量。

where 在数据分析中的一个典型用法是根据一个数组来生成一个新的数组。


```python
arr = np.random.randn(4, 4)
arr
```




    array([[ 1.11382497, -0.56882187, -2.64307613,  0.08504018],
           [ 0.31993552, -1.83044535, -0.6199498 ,  0.30061171],
           [ 0.81729467,  1.6693836 , -0.15945171,  0.25866852],
           [-0.46382228, -1.10865797,  0.7173846 , -0.39235309]])




```python
arr > 0 # 这就是condition
```




    array([[ True, False, False,  True],
           [ True, False, False,  True],
           [ True,  True, False,  True],
           [False, False,  True, False]])




```python
np.where(arr > 0, 2, -2)
```




    array([[ 2, -2, -2,  2],
           [ 2, -2, -2,  2],
           [ 2,  2, -2,  2],
           [-2, -2,  2, -2]])



还可以将标量与数组结合，

例如，将 arr 中所有正值替换为常数2：


```python
np.where(arr > 0, 2, arr)
```




    array([[ 2.        , -0.56882187, -2.64307613,  2.        ],
           [ 2.        , -1.83044535, -0.6199498 ,  2.        ],
           [ 2.        ,  2.        , -0.15945171,  2.        ],
           [-0.46382228, -1.10865797,  2.        , -0.39235309]])



### 4.3.2 数学和统计方法

许多关于计算整个数组统计值或关于轴向数据的数学函数，可以作为数组类型的方法称为调用。

你可以使用聚合函数（缩减函数），比如sum、mean和 std（标准差），

既可以直接调用数组实例的方法，也可以使用顶层的 NumPy 函数。

此时我生成一些正态分布的随机数，并计算了部分聚合统计数据：


```python
arr = np.random.randn(5, 4)
arr
```




    array([[ 0.33866822,  0.25670654, -1.09321578, -1.51200092],
           [ 0.47011221,  2.00831998,  1.95196054,  1.09750339],
           [ 0.8374163 , -0.64938786, -1.98042032, -1.25061797],
           [ 0.75218917, -2.14526029,  0.32430664,  0.57853034],
           [-0.14056578, -0.17759475,  1.22827116, -1.9913405 ]])




```python
arr.mean()
```




    -0.05482098440722396




```python
np.mean(arr)
```




    -0.05482098440722396




```python
arr.sum()
```




    -1.0964196881444792



像mean、sum等函数可以接受一个可选参数axis，参数可以用于计算给定轴向上的统计值，形成一个下降一维度的数组：


```python
arr.mean(axis=1)
```




    array([-0.50246049,  1.38197403, -0.76075246, -0.12255853, -0.27030747])




```python
arr.sum(axis=0)
```




    array([ 2.25782012, -0.70721638,  0.43090224, -3.07792566])



其他函数，例如 cumsum 和 cumprod 并不会聚合，它们会产生一个中间结果：


```python
arr = np.arange(8)
arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7])




```python
arr.cumsum()
```




    array([ 0,  1,  3,  6, 10, 15, 21, 28], dtype=int32)



还可以在指定轴向上根据较低维度的切片进行部分聚合：


```python
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
arr.cumsum(axis=0)
```




    array([[ 0,  1,  2],
           [ 3,  5,  7],
           [ 9, 12, 15]], dtype=int32)




```python
arr.cumprod(axis=1)
```




    array([[  0,   0,   0],
           [  3,  12,  60],
           [  6,  42, 336]], dtype=int32)



#### 基础数组统计方法

sum 沿着轴方向计算所有元素的累和，0长度的数组，累计和为0

mean 数学平均，0长度的数组平均值为NaN

std,var 标准差和方差，可以选择自由调整度，默认分母是n

min,max 最小值和最大值

argmin,argmax 最小值和最大值的位置

cumsun 从0开始元素累积和

cumprod 从1开始元素累积积


### 4.3.3 布尔值数组的方法

sum 通常可以用于计算布尔值数组中的 True 的个数：


```python
arr = np.random.randn(100)
(arr > 0).sum() # 正数的个数
```




    48



any 方法检查数组中是否至少有一个 True；

all 方法检查是否每个值都是True


```python
bools = np.array([True, True, False, True])
```


```python
bools.any()
```




    True




```python
bools.all()
```




    False



这些方法也适用于非布尔值数组，所有非0元素都会按 True 处理

### 4.3.4 排序

和 Python 的内建列表类型相似，NumPy 数组可以使用 sort 方法按位置排序：


```python
arr17 = np.random.randn(6)
arr17
```




    array([-0.09559334, -0.98671866,  1.593457  ,  1.59425253, -0.2877424 ,
            0.38386557])




```python
arr17.sort()
arr17 ## 1*1 从小到大排序，改变本身
```




    array([-0.98671866, -0.2877424 , -0.09559334,  0.38386557,  1.593457  ,
            1.59425253])




```python
arr18 = np.random.randn(5, 3)
arr18 
```




    array([[-0.24351852, -0.6397246 ,  0.13097486],
           [ 0.77230555, -0.36228155,  1.94839886],
           [ 0.32735087,  0.86200479, -0.22114366],
           [ 0.38316247,  0.57985576, -0.62029429],
           [ 0.25688937, -1.82403165,  0.48868915]])




```python
arr18.sort(1) # 可以根据 axis 值进行排序
arr18
```




    array([[-0.6397246 , -0.24351852,  0.13097486],
           [-0.36228155,  0.77230555,  1.94839886],
           [-0.22114366,  0.32735087,  0.86200479],
           [-0.62029429,  0.38316247,  0.57985576],
           [-1.82403165,  0.25688937,  0.48868915]])




```python
arr18.sort(0) # 5*3，沿着纵轴排序
arr18
```




    array([[-1.82403165, -0.24351852,  0.13097486],
           [-0.6397246 ,  0.25688937,  0.48868915],
           [-0.62029429,  0.32735087,  0.57985576],
           [-0.36228155,  0.38316247,  0.86200479],
           [-0.22114366,  0.77230555,  1.94839886]])



可以发现，以上操作都将原数组改变；

顶层 np.sort方法返回的是已经排序好的数组拷贝。


```python
arr = np.random.randn(5)
arr
```




    array([-0.84132643,  0.73456422, -0.75911104,  0.76730715, -0.88579067])




```python
np.sort(arr)
```




    array([-0.88579067, -0.84132643, -0.75911104,  0.73456422,  0.76730715])




```python
arr # 原数组没有改变
```




    array([-0.84132643,  0.73456422, -0.75911104,  0.76730715, -0.88579067])



下面的例子计算的是一个数组的分位数：


```python
large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]  # 排序完了，长度0.05位置的元素取出来
# 5% quantile
# 5% 分为点
```




    -1.7102453297255942



### 4.3.5 唯一值与其他集合逻辑

NumPy 包含一些针对一堆 ndarray 的基础集合操作。

常用的一个方法 np.unique,返回的是数组中唯一值排序后形成的数组：


```python
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
```




    array(['Bob', 'Joe', 'Will'], dtype='<U4')




```python
ints = np.array([3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 15])
np.unique(ints)
```




    array([ 1,  2,  3, 15])



将 np.unique 和纯 Python 实现比较：


```python
sorted(set(names))
```




    ['Bob', 'Joe', 'Will']



另一个函数 np.in1d，检查一个数组中的值是否在另一个数值中，并返回一个布尔值数组：


```python
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6]) # 前者中的值是否在后者中
```




    array([ True, False, False,  True,  True, False,  True])



#### 数组的集合操作

unique(x) 计算x的唯一值，并排序

intersect1d(x, y) 计算x和y的交集，并排序

union1d(x, y) 计算x和y的并集，并排序

in1d(x, y) 计算x中的元素是否包含在y中，返回一个布尔值数值组

setdiff1d(x, y) 差集，在x中但不在y中的x的元素

setxor1d(x, y) 在x或在y，不同时在xy中

## 4.4 使用数组进行文件输入和输出

这个之后还会用pandas做，所以这里就是简单介绍Numpy的内建二进制模式

np.save 和 np.load 是高效存取硬盘数据的两大工具函数。

数组在默认情况下是以未压缩的格式进行存储的，后缀名是 .npy:


```python
arr = np.arange(10)
```


```python
np.save('some_array', arr) # 存放路径中没有写 .npy时，后缀会被自动加上
```

硬盘上的数组可以使用 np.load 进行载入：


```python
np.load('some_array.npy') # 载入时就必须要写 .npy
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.savez('array_archive.npz', a=arr, b=arr)# savez 默认未压缩，多个数组，后缀名.npz
```


```python
arch = np.load('array_archive.npz')  # load读取，是个字典型的对象
```


```python
arch['b']
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.savez_compressed('arrays_compressed.npz', a=arr, b=arr) # 如果数据压缩好了，存入压缩文件
```

## 4.5 线性代数

### 常用 numpy.linalg 函数

diag 将方阵的对角元素 -- 一位数组，之间转换，空白默认用0

dot 矩阵点乘

trace 对角元素和

det 矩阵的行列式

eig 特征值和特征向量

inv 逆矩阵

pinv 计算矩阵的Moore-Penrose伪逆向，什么鬼

qr 计算QR分解

svd 计算奇异值分解（SVD）

solve 求解x的线性系统 Ax = b， 其中A是方阵

lstsq 计算 Ax = b 的最小二乘解


```python
x = np.array([[1., 2., 3.], [4., 5., 6.]])# （2，3）
x 
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
y = np.array([[6., 23.], [-1, 7], [8, 9]]) # （3，2）
y
```




    array([[ 6., 23.],
           [-1.,  7.],
           [ 8.,  9.]])




```python
x.dot(y) 
```




    array([[ 28.,  64.],
           [ 67., 181.]])



x.dot(y) 等价于 np.dot(x, y):


```python
np.dot(x, y) 
```




    array([[ 28.,  64.],
           [ 67., 181.]])




```python
np.dot(x, np.ones(3)) 

# 这个np.ones(3)是啥，先看一眼
# 应该是2*3与3*1变成2*1
```




    array([ 6., 15.])




```python
x @ np.ones(3) ##一个意思，@用于点乘矩阵
```




    array([ 6., 15.])




```python
from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
X
```




    array([[-0.57525014, -1.60406114,  0.23862338,  0.41485475, -0.22466479],
           [-0.31954158,  0.10292421, -0.18466744, -0.42926119,  0.73914035],
           [ 0.36690802,  0.28700823, -0.48666467,  2.65833388,  0.26028593],
           [-0.22842198,  1.66332362, -1.57176859,  0.15627762,  1.05996425],
           [-0.18343953,  1.19407585, -0.01392484,  0.20550137, -0.03328424]])




```python
mat = X.T.dot(X) # 转置矩阵与 X点积(内积)
mat
```




    array([[ 0.6534677 ,  0.39617308,  0.10476048,  0.80049123, -0.24746012],
           [ 0.39617308,  6.85844185, -3.17243714,  0.55865444,  2.23447536],
           [ 0.10476048, -3.17243714,  2.79853609, -1.36394641, -1.98233243],
           [ 0.80049123,  0.55865444, -1.36394641,  7.48976216,  0.44024812],
           [-0.24746012,  2.23447536, -1.98233243,  0.44024812,  1.78918354]])




```python
inv(mat) ## 逆矩阵
```




    array([[ 2.56846446, -0.50632804, -0.36075545, -0.34194396,  0.67202329],
           [-0.50632804,  0.42277257,  0.52209827,  0.12055289, -0.04922454],
           [-0.36075545,  0.52209827,  2.68763931,  0.36049472,  2.18714237],
           [-0.34194396,  0.12055289,  0.36049472,  0.21802459,  0.14791412],
           [ 0.67202329, -0.04922454,  2.18714237,  0.14791412,  3.10019302]])




```python
mat.dot(inv(mat)) # 结果近似单位矩阵
```




    array([[ 1.00000000e+00,  1.00613962e-16, -3.33066907e-16,
             9.02056208e-17,  0.00000000e+00],
           [ 2.22044605e-16,  1.00000000e+00,  8.88178420e-16,
             0.00000000e+00, -8.88178420e-16],
           [-2.22044605e-16, -1.24900090e-16,  1.00000000e+00,
             0.00000000e+00,  0.00000000e+00],
           [-1.66533454e-16, -3.46944695e-18,  2.22044605e-16,
             1.00000000e+00,  0.00000000e+00],
           [ 0.00000000e+00,  8.32667268e-17,  0.00000000e+00,
            -1.11022302e-16,  1.00000000e+00]])




```python
q, r = qr(mat) # 计算QR分解
r
```




    array([[-1.13884618, -2.22850527,  1.3140313 , -5.69706124, -0.37364561],
           [ 0.        , -7.5893527 ,  4.32945675, -0.12486543, -3.28446578],
           [ 0.        ,  0.        , -1.79617764,  4.76964269,  1.16836014],
           [ 0.        ,  0.        ,  0.        , -1.97027837,  0.13826221],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.25931893]])




```python
q
```




    array([[-0.57379803,  0.11628645, -0.19780444,  0.76664022,  0.17426836],
           [-0.34787233, -0.80154484, -0.42029903, -0.24432864, -0.01276485],
           [-0.09198826,  0.4450226 , -0.55267665, -0.40787583,  0.56716741],
           [-0.70289671,  0.13278531,  0.56520379, -0.4091182 ,  0.03835693],
           [ 0.2172902 , -0.35822656,  0.39914324,  0.13720753,  0.80393872]])



## 4.6 伪随机数生成

### numpy.random 中的部分函数列表

seed 向随机数生成器传递随机状态种子

permutation 返回一个序列的随机排列，或返回一个乱序的整数范围序列

shuffle 随机排列一个序列

rand 从均匀分布中抽取样本

randint 根据给定的由低到高的范围抽取随机整数

randn 从均值0方差1的正太分布中抽取样本（MATLAB型接口）

binomial 从二项分布中抽取样本

normal 从正太分布中抽取样本

beta 从beta分布中抽取样本

chisqueare 从卡方分布中抽取样本

gamma 从伽马分布中抽取样本

uniform 从均匀[0,1)分布中抽取样本

numpy.random 模块填补了 Python 内建的 random 模块的不足，可以高效的生成多种概率分布下的完整样本值数组。

例如，可以使用 normal 来获取一个 4 * 4 的正态分布样本数组：


```python
samples = np.random.normal(size=(4, 4))
## 4乘4，正态分布
samples
```




    array([[ 1.29089621, -1.10064108, -0.63363085, -0.69436472],
           [ 0.42637877,  0.17601022,  0.22618884,  1.16137942],
           [ 0.32908252, -1.06767016,  2.14914183, -0.38072812],
           [ 0.04027688,  1.19528927,  0.85820602, -0.09267409]])



然而 Python 内建的 random 模块一次只能生成一个值。

下面示例中，numpy.random 在生成大型样本时比纯 Python 的方式快了一个数量级：


```python
from random import normalvariate # 正态变量，Python函数
N = 1000000
%timeit samples = [normalvariate(0, 1) for _ in range(N)]
```

    1.17 s ± 94.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    


```python
%timeit np.random.normal(size=N)# np函数比Python快了一个数量级，40倍
```

    33.9 ms ± 1.4 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    

这些就被称为伪随机数，因为它们是由具有确定性行为的算法根据随机数生成器中的随机数种子生成的。

可以通过 np.random.seed 更改 NumPy 的随机数种子：


```python
# np.random.seed(1234) ## 之所以没有运行，是因为它会生成一个全局随机数种子
```

为避免全局状态，可以使用 numpy.random.RandState 创建一个随机数生成器，使数据独立于其他的随机数状态。


```python
rng = np.random.RandomState(1234)
```


```python
rng.randn(10)
```




    array([ 0.47143516, -1.19097569,  1.43270697, -0.3126519 , -0.72058873,
            0.88716294,  0.85958841, -0.6365235 ,  0.01569637, -2.24268495])



## 4.7 示例：随机漫步

首先，考虑一个简单的随机漫步，从0开始，步进为1和-1，并且两种步进发生的概率相等。

以下是使用内建 random 模块利用纯 Python 实现的一个1000步的随机漫步：


```python
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
```


```python
plt.plot(walk[:100])
```




    [<matplotlib.lines.Line2D at 0x1e5523b99c8>]




![png](output_284_1.png)


你可以观察到 walk 只是对随机步进的累积和，并且可以通过一个数组表达式来实现。

我使用 np.random 模块一次性抽取1000次投硬币的结果，每次投掷的结果为1 或 -1，然后计算累计值：


```python
nsteps = 1000
```


```python
draws = np.random.randint(0, 2, size=nsteps)
```


```python
steps = np.where(draws > 0, 1, -1)
```


```python
walk = steps.cumsum()
```


```python
plt.plot(walk[:100])
```




    [<matplotlib.lines.Line2D at 0x1e555babdc8>]




![png](output_290_1.png)


由此我们还可以提取一些统计数据，比如最大值，最小值等：


```python
walk.max()
```




    37




```python
walk.min()
```




    -7




```python
np.abs(walk) >= 10
```




    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False,  True,
            True,  True, False,  True, False,  True, False,  True, False,
           False, False,  True, False,  True,  True,  True, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False,  True,  True,  True, False,
           False, False,  True, False, False, False,  True,  True,  True,
           False, False, False, False, False,  True,  True,  True, False,
           False, False, False, False,  True, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False,  True,  True,
            True,  True,  True, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True, False,
            True, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False,  True,
            True,  True, False, False, False, False, False,  True, False,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True])




```python
(np.abs(walk) >= 10).argmax()
# 就是最大值第一次出现的位置
# 也就是Ture第一次出现的位置
```




    35



### 4.7.1 一次性模拟多次随机漫步

下面我们来模拟5000次随机漫步。

如果传入一个2个元素的元组，numpy.random 中的函数可以生成一个二维的抽取数组，

并且我们也可以一次性计算5000个随机步的累积和：


```python
nwalks = 5000
```


```python
nsteps = 1000
```


```python
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
```


```python
steps = np.where(draws > 0, 1, -1)
```


```python
walks = steps.cumsum(1)
```


```python
walks
```




    array([[ -1,  -2,  -3, ..., -52, -51, -50],
           [  1,   2,   3, ..., -62, -61, -60],
           [ -1,  -2,  -1, ..., -28, -27, -26],
           ...,
           [  1,   2,   1, ..., -14, -15, -14],
           [  1,   0,   1, ...,   2,   3,   4],
           [  1,   2,   1, ..., -36, -35, -34]], dtype=int32)



计算随机步的最大值和最小值


```python
walks.max()
```




    116




```python
walks.min()
```




    -122



计算出30或-30的最小穿越时间。

并不是所有的都达到30，我们找出达到30的这些：


```python
hits30 = (np.abs(walks) >= 30).any(1)
```


```python
hits30
```




    array([ True,  True,  True, ...,  True,  True,  True])




```python
hits30.sum() # 达到30的数量
```




    3390




```python
walks[hits30]
```




    array([[ -1,  -2,  -3, ..., -52, -51, -50],
           [  1,   2,   3, ..., -62, -61, -60],
           [ -1,  -2,  -1, ..., -28, -27, -26],
           ...,
           [  1,   2,   1, ..., -14, -15, -14],
           [  1,   0,   1, ...,   2,   3,   4],
           [  1,   2,   1, ..., -36, -35, -34]], dtype=int32)




```python
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
```


```python
crossing_times
```




    array([809, 501, 957, ..., 613, 351, 763], dtype=int64)




```python
crossing_times.mean()
```




    505.57935103244836


