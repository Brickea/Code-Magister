# 数据结构与算法 - Python

记录学习数据结构与算法的过程

[课程参考](https://pegasuswang.github.io/python_data_structures_and_algorithms/)

## 需要 Pre intall 的环境

* pytest 测试框架和 when-changed 文件变动监控工具
  * 使用 ```pip``` 安装
    ```
    pip install pytest
    pip install when-changed
    ```
  * 在的 ~/.bashrc or ~/.zshrc 里边加上这个映射（别忘记加上之后source下）:
    ```
    # 监控当前文件夹文件变动自动执行命令
    alias watchtest='when-changed -v -r -1 -s ./ '
    ```
    然后在你的代码目录里头执行 ```watchtest pytest -s somefile.py ```
  * pytest 会自动发现以 test 开头的函数并执行测试代码。良好的工程需要我们用单测来保证，将来即使修改了内部实现逻辑也方便做回归验证。

## 测试用例设计 [参考来源](https://pegasuswang.github.io/python_data_structures_and_algorithms/)

在刚学习编程的时候总是忘记处理一些特例(尤其是动态语言可以传各种值)，为了养成良好的编程和测试习惯，在编写单元测试用例的时候， 我们注意考虑下如下测试用例(等价类划分)：

* 正常值功能测试
* 边界值（比如最大最小，最左最右值）
* 异常值（比如 None，空值，非法值）

```python
def binary_search(array, target):
    if not array:
        return -1
    beg, end = 0, len(array)
    while beg < end:
        mid = beg + (end - beg) // 2  # py3
        if array[mid] == target:
            return mid
        elif array[mid] > target:
            end = mid
        else:
            beg = mid + 1
    return -1


def test():
    """
    如何设计测试用例：
    - 正常值功能测试
    - 边界值（比如最大最小，最左最右值）
    - 异常值（比如 None，空值，非法值）
    """
    # 正常值，包含有和无两种结果
    assert binary_search([0, 1, 2, 3, 4, 5], 1) == 1
    assert binary_search([0, 1, 2, 3, 4, 5], 6) == -1
    assert binary_search([0, 1, 2, 3, 4, 5], -1) == -1
    # 边界值
    assert binary_search([0, 1, 2, 3, 4, 5], 0) == 0
    assert binary_search([0, 1, 2, 3, 4, 5], 5) == 5
    assert binary_search([0], 0) == 0

    # 异常值
    assert binary_search([], 1) == -1
```
当然我们也不用做的非常细致，要不然写测试是一件非常繁琐累人的事情，甚至有时候为了测试而测试，只是为了让单测覆盖率好看点。 当然如果是web应用用户输入，我们要假设所有的参数都是不可信的。 但是很多内部调用的函数我们基于约定来编程，如果你瞎传参数，那就是调用者的责任了。