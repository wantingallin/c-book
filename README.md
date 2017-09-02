### 3第一个Pytorch实例：预测房价
为了更好地理解前面所讲的概念，我们来引入一个预测房价问题实例，实现一个线性回归模型，并用梯度更新算法来求解该模型从而给出预测直线。线性回归是指用找到一条使得总误差最小直线来对输入和输出变量进行线性拟合，从而更好地进行预测。

首先我们考虑这样一个现实问题：已经有一组随着时间变化的房价数据，如何来预测未来某一天的房价是多少呢?针对一个问题，我们通用的建模步骤应该是这样的：
1. 准备数据
1. 模型设计
1. 训练
1. 测试

![alt](http://img.blog.csdn.net/20170829152457815?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FudGluZzE1MTExODE0OTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 1.准备数据

为了简单起见，我们可以人为的生成一些样本点作为我们的数据。 
首先由linspace构造了0~100之间的均匀数字作为 Variable x

```Python
x = Variable(torch.linspace(0, 100).type(torch.FloatTensor)) 
```

然后利用rand随机生成100个满足标准正态分布的随机数，均值为0，方差为1.并将这个数字乘以10，标准方差变为10，构造噪声。

![alt](http://img.blog.csdn.net/20170829152457815?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FudGluZzE1MTExODE0OTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```Python
rand = Variable(torch.randn(100)) * 10 
```

将x和rand相加，得到伪造的标签数据y。所以(x,y)应能近似地落在y=x这条直线上
y = x + rand 
通常我们都会使用将我们已有的数据进行可视化，观察数据的形态从而决定采取哪种算法。在这里，我们同样的进行数据可视化的处理。 
首先导入进行画图的程序包，我们通常使用最好用的画图软件包matplotlib：

```Python
import matplotlib.pyplot as plt 
```

设定绘制窗口大小为10*8 inch：

```Python
plt.figure(figsize=(10,8)) 
```

绘制数据，考虑到x和y都是Variable，需要用.data获取它们包裹的Tensor，用.numpy指令将tensor转成numpy：

```python
plt.plot(x.data.numpy(), y.data.numpy(), 'o')
```


添加标注，并画出图形：

```python
plt.xlabel('X')
plt.ylabel('Y') 
plt.show() #将图形画在下面
```

最终得到的输出图像：

![alt](http://img.blog.csdn.net/20170829154613694?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FudGluZzE1MTExODE0OTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

观察x,y的散点图，可以看出x,y的走势呈现线性，所以可以用线性回归来进行拟合。


#### 2.模型设计


为了刻画拟合的程度，我们定义一个函数，可以叫做损失函数（loss function），在线性回归中，我们的损失函数通常都采用最小均方误差函数的形式（minimum square error function），函数表达式为： 
L(Y,f(X))=(Y−f(X))2
其中Y是真实值，f(X)是预测值。
因为我们的目标是求出最佳的拟合曲线，即对应的损失函数最小，从而可以确定我们的目标函数（target function）： 
minL(Y,f(X))=min(Y−f(X))2=1n∑i=1n(Yi−axi−b)2

由此我们成功地将房价的线性回归问题转化为：求参数a,b使得损失函数最小。

根据我们的数学基础知识，对线性函数而言，可以通过对损失函数求导，导数为0，来确定极值点，但是对计算机而言，求导是很困难的。于是人们为了使计算机能够找到极值点，设计了一种算法：梯度下降算法，这种算法通过迭代计算，可使计算机找到极值点。 
关于梯度下降算法的原理，可以参见数学知识补充部分。这里直接给出算法的迭代公式：
ai+1=ai−α∂L∂ai

bi+1=bi−α∂L∂bi

其中α表示学习率。
训练
将上述数学思想转化为Pytorch的代码如下： 
首先将a,b进行初始化:

```Python
a = Variable(torch.rand(1), requires_grad = True) 
b = Variable(torch.rand(1), requires_grad = True) 
print('Initial parameters:', parameters) 
```

然后设置学习率α：

```Python
learning_rate = 0.0001 
```


接着定义训练的过程，通过pridections函数计算出预测值，在计算预测值时，尤其需要注意expand_as和mul的使用。首先，a的维度为1，x的维度为100*1的tensor，这两者不能直接相乘，因为维度不同。 

所以，先要将a升维成1*1的Tensor。这就好比将原本在直线上的点被升维到了二维平面上，同时直线仍然在二维平面中。expand_as(x)可以将张量升维成与x同维度的张量。所以如果a=1, x为尺寸为100，那么， 

```Python
a.expand_as(x) =(1,1,⋅⋅⋅,1)T
```

x∗y为两个1维张量的乘积，计算结果： 
(x∗y)i=xi⋅yi

利用loss函数计算出误差；loss.backward()进行反向传播，通过 a.data.add_ 和b.data.add_进行参数的更新。这里需要注意我们无法改变一个Variable，而只能对Variable的data属性做更改，所有函数加._都意味着需要更新调用者的数值。

```Python
for i in range(1000):    
    predictions = a.expand_as(x).mul(x)+ b.expand_as(x)    
    loss = torch.mean((predictions - y) ** 2)    
    print('loss:', loss)    
    loss.backward()    
    a.data.add_(- learning_rate * a.grad.data)    
    b.data.add_(- learning_rate * b.grad.data)    
```


```Python
a.grad.data.zero_()     
b.grad.data.zero_()
```

因为backward()反向传播算法计算的是累加梯度，所以每次更新时需要利用.grad.data.zero_()将梯度信息清零。 
最后将图形画出来：

```Python
 x_data = x.data.numpy() 
 plt.figure(figsize = (10, 7))      
 xplot, = plt.plot(x_data, y.data.numpy(), 'o')      
 yplot, = plt.plot(x_data, a.data.numpy() * x_data +b.data.numpy())     
 plt.xlabel('X')      
 plt.ylabel('Y')      
 str1 = str(a.data.numpy()[0]) + 'x +' + str(b.data.numpy()[0])      
 plt.legend([xplot, yplot],['Data', str1])      
 plt.show()
 
```

通过画出图像，得到我们拟合的直线：

![alt](http://img.blog.csdn.net/20170830084236154?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FudGluZzE1MTExODE0OTg=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 4.测试
最后一步，进行测试，给定Variable x，在这个线性模型中指的是我们的时间，我们预测出对应的y，也就是房价。在这个问题中，我们给定x=[1,2,10,100,1000],将x输入predictions函数，计算出预测的y值。

```Python
x_test = Variable(torch.FloatTensor([1, 2, 10, 100, 1000])) 
predictions = a.expand_as(x_test) * x_test + b.expand_as(x_test)  
predictions
```

最终的输出结果为： 


```python
Variable containing: 
1.8748 
2.8295 
10.4669 
96.3885 
955.6038 
[torch.FloatTensor of size 5]
```
