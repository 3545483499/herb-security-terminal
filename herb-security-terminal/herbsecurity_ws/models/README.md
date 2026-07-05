# 模型文件说明

本目录只保留分类文件和占位目录，不提交 ONNX 模型。

原始工程中模型文件较大，其中 `buffalo_s/1k3d68.onnx` 超过 GitHub 普通仓库单文件限制，建议：

1. 私有传输：把模型文件单独打包，在 K1 上解压到 `~/herbsecurity_ws/models/`。
2. GitHub Release：把模型压缩包作为 Release 附件上传。
3. Git LFS：确实需要版本管理模型时再使用。

部署时按 README 中的路径放回模型。
