# How to understand the Neural Network through Logistic Regression

[学习参考 - 李宏毅](https://www.youtube.com/watch?v=hSXFuypLukA&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=10)

最近在温习神经网络相关的知识，但是我一直对神经网络的运作有一种莫名的无知感，就好像神经网络就是一个黑盒，我只知道把这些数据放进去，然后结果自己就出来了

今日在看李宏毅老师关于 Machine Learning 的公开课时， 发现了老师对于神经网络有一个非常浅显易懂的解释，特此记录，与大家分享(ง •_•)ง

在文章中我尽量少用到数学推导，用最简单的语言解释

## Logistic Regression

首先先回顾一下什么是 Logistic Regression

其本质是一个 Distriminative Model ，通过学习原始数据并计算新输入数据属于各个类别的概率

Logistic Regression 是怎么运行的呢？

文中符号的意义：

* $x$：为数据
* $C_i$：为第 $i$ 类

### Step 1 - Function Set

首先我们先从直觉出发，如果给定一个数据集，通过学习，我们想要知道新的数据 $x$ 是属于哪一个类别 $C_i$，我们是可以用概率来表示这个问题的

考虑一个二分类问题，也就是我们只有两个类别 $C_1$ 和 $C_2$

此时我们假设 $P(C_1|x)$ 为新数据 $x$ 是 类别 $C_1$ 的概率

然后假设分类的阈值为 0.5

那么我们就有以下的条件选择

IF $P(C_1|x) \geq 0.5$ , OUTPUT $C_1$  
ELSE OUTPUT $C_2$

此处

$P(C_1|x) = \sigma(z)$

$z = \sum_{i=1}^{n} w_ix_i + b$ （n 为样本数量）

$\sigma(z) = \frac{1}{1+exp(-z)}$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1920px-Logistic-curve.svg.png)

> $\sigma$(z) （sigmoid) 函数图像 [来源参考](https://en.wikipedia.org/wiki/Sigmoid_function)

最终我们可以得到计算新数据属于特定分类的函数

$f_{w,b} = P_{w,b}(C_1|x)$ （此处用分类为 $C_1$ 举例）

用图像来解释即为

![](https://github.com/Brickea/machine_learning/blob/master/ML%20Theory/res/logistic_regression_function_set.jpg?raw=true)

> [来源参考](https://www.youtube.com/watch?v=hSXFuypLukA&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=10)

### Step 2 - Goodness of our function

通过上一步我们已经有了计算分类概率的函数了，但是我们并不知道这个函数的好坏程度，即他是否分类够准？

接下来我们需要可以衡量他好坏的函数 - Loss Function

对于分类问题，我们常用 Cross Entropy （交叉熵）来衡量分类函数的好坏

对于 Step 1 中提到的二分类问题

Loss Function 如下

$L(w,b) = \sum_{i=1}^{n} -[\hat{y_n}ln(f_{w,b}(x_n)+(1-\hat{y_n})ln(1-f_{w,b}(x_n)))]$

> 对于具体推导过程可以[参考](https://www.youtube.com/watch?v=hSXFuypLukA&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=10) 这个地方的 4：05 - 12：00 

### Step 3 - Find the best function

当我们有了计算分类概率的函数 $P$ 和 衡量该函数好坏的 Loss Function $L$

接下来我们就可以通过 Loss Function 来不断优化 $P$ 中的各个参数，进而获得最好的 $P$

在这里我们考虑梯度下降方法

最终更新 $P$ 中各个 $w$ 的方法如下

$w_j=w_j-\eta\sum_{i=1}^{n}-(\hat{y_i}-f_{w,b}(x_i))x_{i,j}$

* 此处 j 代表第几个 feature 的参数
* 此处 n 代表样本数量
* 此处 i 代表第几个样本

之后，模型只需要通过梯度下降的方法来更新每个feature的参数，最终可以得到能够预测较好结果的模型

## 如何从 Logistic Regression 的角度看 Neural Network 神经网络

接下来就是老师讲的一个非常有意思的例子，老师用这个例子进而引入到了神经网络

### Logistic Regression 的缺点

我们先从 Logistic Regression 的一个短板来出发

考虑以下情况

![](https://github.com/Brickea/machine_learning/blob/master/ML%20Theory/res/logistic_regression_weakness.png?raw=true)

> [图片参考](https://www.youtube.com/watch?v=hSXFuypLukA&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=10) 58:03

当我们的数据线性不可分的时候（如上图中的红点和蓝点）

这时候 Logistic Regression 就不好用了，我们很难找到一个合适 decision boundary

### Logistic Regression + Feature Transformation

我们任然考虑刚刚提到的红蓝例子

虽然原始数据线性不可分，但是我们可以通过某种数据变换，把原始数据转变成为可分的问题

在这里我们选择的特征变换是把原问题转变成为“原始数据距离 (0,0) 和 (1,1) 的距离”

变换后，两个蓝色的点在一侧，两个红色点因为转变后数值一样，所以重合在另一侧

![](https://github.com/Brickea/machine_learning/blob/master/ML%20Theory/res/logistic_regression_data_transfermation.png?raw=true)

> [图片参考](https://www.youtube.com/watch?v=hSXFuypLukA&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=10) 1:00:43

然后再使用 Logistic Regression 去拟合转变后的数据，此时就能得到正确的分类模型

但是这里有一个问题

我们不可能每一次都去人工找这个 Feature Transformation 的方法，而且有的时候在高维信息中，这个方法是很难找到的

### Cascading Logistic Regression

那我们就想到了，如果我们专门设置一些 Logistic Regression 去帮我们找到这个 Feature Transformation 可不可以？

可

![](https://github.com/Brickea/machine_learning/blob/master/ML%20Theory/res/cascading_logistic_regression.png?raw=true)

> [图片参考](https://www.youtube.com/watch?v=hSXFuypLukA&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=10) 01:02:38

如上图，我们添加了两个 Logistic Regression 去帮我们寻找经过 Feature Transformation 后的特征，最后再用一个 Logistic Regression 去得到最终结果

这个结构，是不是很眼熟。

换个高大上的名字，这种结构也可以叫做神经网络

## 结语

通过这个角度去看神经网络，似乎黑盒也没有那么黑了，我们可以简单的理解神经网络会帮我们去寻找一些我们看不出来的数据特征。

我在这里只是想记录分享这种看待神经网络的角度，但是我们不能直接说神经网络就是做特征转换的。因为依据结构的不同，神经网络的功能会很复杂。可以把这个看作理解神经网络黑盒的一个角度

---

20200322  
拒绝伸手，从我做起