# GitHub 上传步骤

## 1. 本地检查

```bash
cd herb-security-terminal
git status
find . -name "*.key" -o -name "*.pem" -o -name "*.crt" -o -name "*.onnx"
```

正常情况下，上面的命令不应该找到真实密钥和 ONNX 模型文件。

## 2. 初始化仓库

```bash
git init
git add .
git commit -m "init: add herb security terminal source code"
```

## 3. GitHub 新建仓库

在 GitHub 网页端新建一个空仓库。不要勾选自动创建 README、.gitignore 或 LICENSE，因为本地已经有这些文件。

## 4. 绑定远程仓库并推送

```bash
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

## 5. 后续更新

```bash
git status
git add .
git commit -m "update: 修改内容说明"
git push
```

## 6. 如果一定要上传模型

推荐把模型作为 GitHub Release 附件或使用 Git LFS，不要直接塞进普通 Git 提交里。
