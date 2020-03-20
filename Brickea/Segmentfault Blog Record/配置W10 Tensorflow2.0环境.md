# 配置Win10 Tensorflow2.0 GPU 环境

最近在写Reinforcement Learning 相关的作业，要用到Deep Q-Learning的方法

建立神经网络的时候打算用tensorflow，发现有CPU和GPU两个支持模式

鉴于台式机有个1060，我寻思着不能浪费

配置的时候发现这东西可真麻烦，特此记录流程，以备查询

## Step 1 安装 Tensorflow

参考官方[文档](https://www.tensorflow.org/install/gpu)

2.0 版本之后使用```pip```安装的时候只需要安装```tensorflow```即可，不需要指定tensorflow-gpu

```
pip install tensorflow  # stable
```

## Step 2 查看显卡是否支持CUDA

先确认自己的显卡版本

1. Win + S：输入```设备管理器```

2. 找到```Display adapters```，点击下拉，查看显卡版本

我的是 GeForce GTX 1060

之后参考 Nvidia 说明，查找自己的显卡版本是否支持CUDA [网站](https://developer.nvidia.com/cuda-gpus)

## Step 3 下载软件依赖

* CUDA Toolkit [网站](https://developer.nvidia.com/cuda-toolkit-archive)
  * 注意版本问题：TensorFlow supports CUDA 10.1 (TensorFlow >= 2.1.0)
  * 一定要版本对应！！！
  * 查看已安装 ```tensorflow``` 版本

    ```
    # 进入命令行python环境并输入一下代码
    import tensorflow as tf
    tf.__version__ 
    # 在这之下会输出对应版本 '2.1.0'
    ```

* Nvidia GPU 驱动 [网站](https://www.nvidia.com/download/index.aspx?lang=en-us)
  * 注意版本问题：CUDA 10.1 requires 418.x or higher.
* CUPIT 安装 [网站](https://docs.nvidia.com/cupti/Cupti/r_main.html#r_initialization)
  * 貌似不用主动安装，这个东西采用的是惰性安装，只要调用到就会自动装
* cuDNN SDK [网站](https://developer.nvidia.com/cudnn)
* vc_redist [网站](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
  * 64位系统一般都是选x64的那个就可以了
    ```
    Visual Studio 2015, 2017 and 2019
    Download the Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019. The following updates are the latest supported Visual C++ redistributable packages for Visual Studio 2015, 2017 and 2019. Included is a baseline version of the Universal C Runtime see MSDN for details.

    x86: vc_redist.x86.exe

    x64: vc_redist.x64.exe # 选这个

    ARM64: vc_redist.arm64.exe
    ```

## Step 4 配置环境变量

参考官方 'windows设置' 部分 [文档](https://www.tensorflow.org/install/gpu?hl=zh-cn#windows_setup)

将 CUDA、CUPTI 和 cuDNN 安装目录添加到 %PATH% 环境变量中。例如，如果 CUDA 工具包安装到 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1，同时 cuDNN 安装到 C:\tools\cuda，请更新 %PATH% 以匹配路径：

```
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\libx64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```

## Step 5 验证GPU是否使用

前四步做完了之后，系统会默认使用GPU，接下来验证一下

进入到python环境

输入

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) # 输出可用GPU数量
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU'))) # 输出可用CPU数量
tf.test.is_gpu_available() # 输出当前是否正在使用GPU，不出意外应该是True
```

---
20200319  
拒绝伸手，从我做起