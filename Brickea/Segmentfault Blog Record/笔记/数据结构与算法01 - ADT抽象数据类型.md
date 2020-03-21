# ADT - Abstract Data Type

ADT 抽象数据类型：我们在组合已有的数据结构，进而实现一种新的数据存储结构。其定义了所采用的数据和对应的操作

## ADT 例子 - Bag 背包

Bag 背包：背包是一种容器类型，我们可以向里面添加东西也可以移除东西，而且我们还可以知道背包里面放了多少东西

## Bag 背包 - python实现

Bag concept：

* Data: 容器
* Method：
  * 添加东西：add
  * 移除东西：remove
  * 获取物品数量：len
  * 迭代物品：iter

思考：

1. 如何选用恰当的数据结构作为存储？
2. 选取的数据结构能否满足 ADT 的功能需求
3. 实现效率如何？

## 相关问题:
* 你了解 python 的魔术方法吗？ 比如 ```__len__ ``，调用 ``len(l)`` 的时候发生了什么？
  * 当使用 ```len(obj)``` 时，解释器实际会调用 ```obj.__len__```
* 你了解单测吗？我们以后将使用 pytest 运行单元测试，保证我们实现的数据结构和算法是正确的。你可以网上搜索下它的简单用法
* python中断言 ```assert``` 是做什么的
  * python 中assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达式为假。可以理解assert断言语句为raise-if-not，用来测试表示式，其返回值为假，就会触发异常。  

    Python的assert是用来检查一个条件，如果它为真，就不做任何事。如果它为假，则会抛出AssertError并且包含错误信息。

    assert的异常参数，其实就是在断言表达式后添加字符串信息，用来解释断言并更好的知道是哪里出了问题。格式如下：
    ```
    assert expression [, arguments]
    即：assert 表达式 [, 参数]
    ```

    一般的用法是：

    ```assert condition```
    用来让程序测试这个condition，如果condition为false，那么raise一个AssertionError出来。逻辑上等同于：
    ```python
    if not condition:
        raise AssertionError()
    ```
    抛出异常是为了可以及时查找到错误信息，避免把错误信息在程序中继续运行。
## 延伸阅读

* [ADT 抽象数据类型](http://www.atjiang.com/data-structures-using-python-ADT/)