# AWS IoT 证书目录

不要把真实证书和私钥提交到 GitHub。

部署时手动放入：

- `AmazonRootCA1.pem`
- `device.pem.crt`
- `private.pem.key`

如果私钥已经上传到公共仓库，应立即在 AWS IoT 后台禁用/删除原证书并重新生成。
