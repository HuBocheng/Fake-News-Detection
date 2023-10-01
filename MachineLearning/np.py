import numpy as np

# 属性查探

a=np.array([[1,2,3,4],[5,6,7,8]])
print(a.ndim) # 返回维度
print(a.shape) # 以（）形式返回返回行列数


# 数组or矩阵操作变换
'''
b=np.array([[1,2,3,4],[5,6,7,8]])
b.shape=(4,2);  # 成员函数.shape=来操作,非函数操作
print(b);
b=b.reshape(2,4)    # 成员函数reshape()函数操作 参数为*两个int*
print(b)

a = np.arange(8).reshape(2,4)
print(a)
print(a.flat[6])  # 返回矩阵展开后对应元素的下标
print(a.flatten(order="C"))  # 返回矩阵展开成一维后的数组，可设置横向展开or纵向展开
print(a.flatten(order="F"))
c=a.flatten()
print(c)

print(a.T)

'''
# 矩阵链接
a = np.arange(1, 7).reshape(2, 3)
print(a)
b = np.arange(7, 13).reshape(2, 3)
print(b)
c = np.concatenate((a, b), axis=0)  # axis=0纵向合并
d = np.concatenate((a, b), axis=1)  # axis=1横向合并
print(c)  # concatenate((a,b))参数是一个列表
print(d)
print()
print()
print(np.hstack((a, b)))  # hstack 水平堆叠   ***hstack和vstack参数是(a,b),有括号***
print(np.vstack((a, b)))  # vstack 垂直堆叠
print(np.append(a,[[10,10,10]],axis=0))  # append函数加行axis=0
print(np.append(a,[[10],[10]],axis=1))  # ***append函数加列axis=1，注意加进去的是个二维矩阵[[]],对于行列得匹配***

# 数组创建
'''
a=np.array([1,2,3])
b=np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[1,1,1,1]]])
print(b,type(b))
print(b.ndim)
print(b.shape)
c=np.arange(1,101)
print(c)
c=np.arange(1,101).reshape(10,10)
print(c)
d=np.linspace(1,5,10)
print(d)
d=np.logspace(0,5,5,endpoint=False,base=2.0)
print(d)
d=np.logspace(0,5,6,endpoint=True,base=2.0)  
print(d)
e=np.empty((2,3))
print(e)
'''

# 生成各种特殊阵
'''
print(np.eye(3, k=1))  # k 参数让对角阵右偏移k
print(np.eye(3, k=-1))
print(np.eye(3, k=1))
print(np.eye((3,4)))
print(np.identity(3))  # 只能创建方阵
print(np.ones(5))
print(np.ones((2,2),dtype="int"))
print(np.ones((3, 4)))
print(np.full((3, 4), 2))
print(np.diag([1, 2, 3, 4]))
print(np.zeros((3, 4)))
print(np.random.rand(3, 4))
print(np.random.randint(5, size=(3, 4)))  # 生成5以内随机整数

'''
# 切片操作
'''
ary2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# 切右下角2*2
print(ary2)
print(ary2[1::, 2::])

print(ary2[::2, 1::2])
print(ary2[..., ::2])  # 可以用省略号表示从头切到尾

print(ary2[[0, 1, 2, ], [1, 2, 3]])  # 高级引用[[]]，第一个【】里放行号，第二个【】里放列号，输出对于行列号元素
# 输出对角线元素
print(ary2[[0, 0, 2, 2], [0, 3, 0, 3]])

rows1 = np.array([[0, 0], [3, 3]])
cols1 = np.array([[0, 2], [0, 2]])  # 上下对0 0  0 2   3 0   3 2
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
y = x[rows1, cols1]
print(y)

rows2 = np.array([[0, 0], [2, 2]])
cols2 = np.array([[0, 3], [0, 3]])
print(ary2[[rows2, cols2]])
'''

# numpy广播
'''
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a * b
d = a * b.T
print(c)
print(d)
ary1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
ary2 = np.array([1, 10, 100, 1000])
print(ary1 * ary2)  # ***乘号*为对位乘法可以广播，不是矩阵乘法***
print(ary1+ary2)
'''

# 矩阵的循环遍历（输出） for x in np.nditer()
'''
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('修改后的数组是：')
for x in np.nditer(a):  # 默认按行输出C风格
    print(x)
b=a.copy(order='C')  # 成员函数copy()参数可设置输出风格，不建议在copy时定输出风格，在遍历时候定比较好
for x in np.nditer(b):
    print(x,end=' ')
b=a.copy(order='F')
for x in np.nditer(b):
    print(x,end=' ')
print()
b=a.T
for x in np.nditer(b):
    print(x,end=" ")  # b变成a转置之后，用迭代器输出默认按F风格输出
print()

for x in np.nditer(b,order="C"):  # ***可强制指定输出风格，推荐使用这个***
    print(x,end=' ')
print()
for x in np.nditer(b, order="F"):
    print(x, end=' ')
print()


for x in np.nditer(a,order="C",op_flags=["readwrite"]):  # ***想修改矩阵的值得把op_flags改成读写模式，记住细节“”和[]***
    x[...]*=2
    print(x)
'''

# 矩阵数学处理
