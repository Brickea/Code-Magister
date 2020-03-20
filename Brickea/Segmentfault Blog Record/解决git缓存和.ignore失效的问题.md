# 清除git已有的记录缓存解决.ignore文件失效的问题

最近更新github项目的时候，发现多了很多```.DS_Store```文件

像是这种文件一般我们都不希望出现在项目中的

## Step 1 清除git 缓存

### 结论:

在对应目录下终端执行

```
git rm -r --cached 文件名
```

### Reference:

鉴于```rm```往往代表删除，直接查git的[文档](https://git-scm.com/docs/git-rm)

发现可以用```git rm```操作来删除缓存区的文件追踪

```
-r
Allow recursive removal when a leading directory name is given.

--cached
Use this option to unstage and remove paths only from the index. Working tree files, whether modified or not, will be left alone.
```

关于缓存区和文件追踪的概念可以参考这个[博客](https://viencoding.com/article/228)

## Step 2 添加 .gitignore 文件

### 结论

在对应的项目根目录创建 ```.gitignore``` 文件

文件中输入

```
*.DS_Store
```

### Reference

```.gitignore```文件屏蔽规则参考此[博客](https://www.jianshu.com/p/13612fb4b224)

## Step 3 重新提交文件记录即可

```
git add .
git commit -m ".gitignore update"
```

---
20200319  
拒绝伸手，从我做起