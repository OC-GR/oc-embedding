# oc-embedding

基于 Hugging Face Text Embeddings Inference（TEI）1.9.4 的独立文本向量推理项目，支持 ARM64 和 AMD64 CPU Docker 镜像构建，可加载 BGE-M3 等模型。

## 源码基线

- 上游：<https://github.com/huggingface/text-embeddings-inference>
- 固定提交：`e80ef225ed0e6cb1717ce632a6a84b6cf211bb67`
- 上游版本：`1.9.4`
- 来源及构建配置：[UPSTREAM.json](UPSTREAM.json)
- 上游文档（已调整介绍文字）：[README.upstream.md](README.upstream.md)

这是此前完成 ARM64、AMD64 镜像构建及官方镜像对比验证的同一份源码快照。初始化时应用源码、Cargo 包名、二进制名、依赖锁文件和根目录的上游 Dockerfile 保持原样；项目名称和本地镜像名称使用 `oc-embedding`。

上游 Git 历史由新的独立仓库替代。原有许可证及版权声明保留，见 [LICENSE](LICENSE)。

## CPU 镜像构建

| 架构 | 已验证构建配方 | 对应上游文件 | 本地镜像名称 |
| --- | --- | --- | --- |
| ARM64 | `docker/Dockerfile.cpu-arm64` | `Dockerfile-arm64` | `oc-embedding:1.9.4-arm64` |
| AMD64 | `docker/Dockerfile.cpu-amd64` | `Dockerfile` | `oc-embedding:1.9.4-amd64` |

`docker/` 中的两份构建文件与此前成功构建时使用的配方完全一致。相对上游仅增加 `ENV CARGO_NET_GIT_FETCH_WITH_CLI=true`，解决 Cargo 内置 Git 下载依赖时出现的 SSL 错误。

先由用户准备目标平台的两个基础镜像：`lukemathwalker/cargo-chef:latest-rust-1.92-bookworm`、`debian:bookworm-slim`。构建过程仍会下载系统软件包和 Rust 编译依赖。

在本仓库根目录执行；ARM64 使用 `ARCH=arm64`，Intel/AMD x86_64 使用 `ARCH=amd64`：

```bash
ARCH=arm64
OC_UPSTREAM_SHA=e80ef225ed0e6cb1717ce632a6a84b6cf211bb67

docker buildx build \
  --builder "$(docker context show)" \
  --pull=false \
  --platform "linux/$ARCH" \
  --file "docker/Dockerfile.cpu-$ARCH" \
  --target http \
  --build-arg "GIT_SHA=$OC_UPSTREAM_SHA" \
  --build-arg "DOCKER_LABEL=oc-embedding-1.9.4-$ARCH" \
  --build-arg SCCACHE_GHA_ENABLED=false \
  --label org.opencontainers.image.title=oc-embedding \
  --label "org.opencontainers.image.revision=$OC_UPSTREAM_SHA" \
  --tag "oc-embedding:1.9.4-$ARCH" \
  --load \
  .
```

使用当前 Docker context 对应的 `docker` driver builder，可复用用户已拉取的基础镜像。Apple Silicon 上构建或运行 AMD64 使用模拟环境，部署到 Intel 服务器后需原生复验。

上述 `GIT_SHA` 标记初始化时未修改的上游源码基线。后续修改应用源码后，应改用新仓库的实际提交号标记构建。

## 模型与验证范围

模型文件单独下载并在运行时挂载，镜像不包含模型权重。本仓库复制源码时未复制模型、旧镜像、构建缓存或验证结果。

此前验证使用 `BAAI/bge-m3`，模型提交为 `5617a9f61b028005a4858fdac845db406aefb181`，CPU float32、CLS pooling、2048-token 输入上限。两个架构与各自官方镜像的稠密向量样本及检索 Top3 比较均通过；AMD64 为 Mac 模拟运行。新仓库初始化后只做了文件一致性检查，尚未以 `oc-embedding` 名称重新构建镜像。

## 上游自动化

活动工作流位于 `.github/workflows/`，推送到任意分支、创建或更新 PR、手动运行均会触发：

| 工作流 | 检查内容 |
| --- | --- |
| `ci.yml` | Rust 1.92.0 格式、上游文件检查、默认 HTTP workspace 编译和 Clippy、library 测试、上游 MiniLM CPU/HTTP 快照测试、CPU gRPC 编译和 Clippy |
| `secrets.yml` | 恢复上游 TruffleHog 密钥泄漏扫描 |
| `docker.yml` | 原生 AMD64、ARM64 CPU 镜像构建，架构和源码 SHA 校验，实际启动服务和离线推理检查，最后删除临时容器及镜像 |

Docker 使用 `ubuntu-24.04` 和 `ubuntu-24.04-arm` 两个原生 GitHub runner，无需仓库密钥或镜像仓库登录。构建使用当前 Git 提交号，不再将新项目的镜像标记为最初的上游 SHA。

镜像检查会用 Python 标准库生成一个小型、固定权重的 XLM-RoBERTa 测试模型，检查 `/health`、`/info`、`/embed`、`/v1/embeddings`、向量维度/归一化、重复输入一致、不同输入有区别、两种 API 结果一致及超长输入拒绝。测试模型无需下载；它用于验证镜像的加载和推理链路，不衡量 BGE-M3 语义效果。Rust 上游测试仍会在 runner 下载公开 tokenizer 和 MiniLM 模型，无需私有 HF token。

镜像仅临时加载到本次 runner，不推送到 registry，不上传镜像 artifact，也不导出镜像构建缓存。无论检查成功或失败，清理步骤都会删除本次测试容器和镜像，并确认已删除；临时 Buildx builder 由 Action 的清理步骤移除。

原始工作流及构建矩阵保存在 `.github/upstream-workflows/`。上游全量测试包含大型模型、GPU、Gaudi 和受限模型；当前恢复的是适合普通 CPU runner 的上述检查。HF 内部镜像发布、Gaudi 定时测试和 HF 文档发布仍归档，需对应的基础设施才能启用。

检查脚本位于 `scripts/ci/`。已在本机现有的两种架构镜像上通过离线推理检查；GitHub 上的完整 CI 编译、构建和运行结果，以代码推送后 Actions 的实际结果为准。
