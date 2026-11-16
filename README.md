# oc-embedding

基于 **Infinity Python/v2 0.0.77** 的文本向量服务，可通过 HTTP 接口加载 BGE-M3，支持 AMD64 与 ARM64 CPU 镜像构建。

## 源码版本

- 上游：[michaelfeil/infinity](https://github.com/michaelfeil/infinity/tree/0.0.77)
- 固定 tag：`0.0.77`
- 固定提交：`4355feffbcbb037cab5c68f510855043d57e6bd7`
- 官方参考镜像：`michaelf34/infinity:0.0.77-cpu`
- 应用源码：`libs/infinity_emb/infinity_emb/`
- 上游依赖及锁文件：`libs/infinity_emb/pyproject.toml`、`poetry.lock`
- 许可：[MIT](LICENSE)；模型权重和第三方依赖遵循各自许可。

项目仓库和自建镜像使用 `oc-embedding` 名称。Python 包名 `infinity_emb`、CLI `infinity_emb v2` 和上游推理实现保持原样。完整来源、镜像 digest、依赖版本和构建差异记录在 [UPSTREAM.json](UPSTREAM.json)，原介绍保存在 [README.upstream.md](README.upstream.md)。

## CPU 镜像构建

官方 `0.0.77-cpu` 发布配置仅构建 AMD64。本项目新增 ARM64 构建配方，两个平台均使用 `docker/Dockerfile.cpu`。

先手动准备对应平台的基础镜像：

```bash
docker pull --platform linux/amd64 ubuntu:22.04
docker pull --platform linux/arm64 ubuntu:22.04
```

在仓库根目录执行；Apple Silicon 使用 `arm64`，Intel/AMD x86_64 使用 `amd64`：

```bash
ARCH=arm64
OC_BUILD_REF=$(git rev-parse HEAD)

docker buildx build \
  --builder "$(docker context show)" \
  --pull=false \
  --platform "linux/$ARCH" \
  --file docker/Dockerfile.cpu \
  --target production \
  --build-arg "VCS_REF=$OC_BUILD_REF" \
  --tag "oc-embedding:0.0.77-$ARCH" \
  --load \
  .
```

`docker` driver 可使用本机已准备的基础镜像。Mac 上运行或构建 AMD64 使用模拟环境；GitHub CI 使用两种架构的原生 runner。

构建会下载系统包和 Python 依赖，不下载 Hugging Face 模型。相对官方 CPU 配方，本项目固定 lock 中的 Torch/Torchvision 版本，并将 OpenVINO `1.22.0` 限定为 AMD64。ARM64 保留标准 ONNX Runtime。Transformers 保留官方配方指定的提交 `7547f55e5d93245c0a013b50df976924f2d9e8b0`，从该提交的源码归档安装并校验 SHA-256。复制应用源码和安装测试依赖时复用已有依赖，避免在同一次构建中重复下载全部运行时包。

## 使用本地 BGE-M3

模型由使用者提前下载，镜像不包含权重。下述部署显式使用 `torch` 后端，可加载 PyTorch 模型文件，无需 ONNX 导出文件。

SentenceTransformers 需要明确的 pooling 配置。只有权重、tokenizer 和 `config.json` 的目录可能回退到 mean pooling，因此首次部署前先准备带 CLS pooling 的模型目录：

```bash
python3 scripts/prepare-bge-m3.py \
  --source /absolute/path/to/bge-m3 \
  --output /absolute/path/to/bge-m3-infinity \
  --max-seq-length 2048
```

脚本检查 XLM-RoBERTa/1024 维配置，补齐 SentenceTransformers 模块及 CLS pooling 配置，并将输入上限明确设为 2048。原模型目录不变；大文件优先使用硬链接，不支持硬链接时才复制。输出目录必须新建或为空。

启动服务；`MODEL_DIR` 必须是上述输出目录的绝对路径：

```bash
ARCH=arm64
MODEL_DIR=/absolute/path/to/bge-m3-infinity

docker run --detach --pull=never \
  --name oc-embedding \
  --platform "linux/$ARCH" \
  --publish 127.0.0.1:18080:7997 \
  --mount "type=bind,source=$MODEL_DIR,target=/models/bge-m3,readonly" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env INFINITY_ANONYMOUS_USAGE_STATS=0 \
  --env OMP_NUM_THREADS=2 \
  --env MKL_NUM_THREADS=2 \
  "oc-embedding:0.0.77-$ARCH" \
  v2 --model-id /models/bge-m3 \
  --served-model-name bge-m3 \
  --engine torch --device cpu --dtype float32 \
  --batch-size 2 --no-bettertransformer --no-model-warmup \
  --host 0.0.0.0 --port 7997
```

验证服务和向量接口：

```bash
curl --fail http://127.0.0.1:18080/health
curl --fail http://127.0.0.1:18080/models
curl --fail http://127.0.0.1:18080/embeddings \
  --header 'Content-Type: application/json' \
  --data '{"model":"bge-m3","input":["如何重置账户密码？"],"encoding_format":"float"}'
```

响应中每条 BGE-M3 向量有 1024 维，Infinity 执行 L2 归一化。默认向量接口是 `/embeddings`；如果客户端需要 `/v1/embeddings`，启动参数增加 `--url-prefix /v1`，此时模型列表为 `/v1/models`，健康检查仍为 `/health`。

## CI 与镜像检查

`.github/workflows/` 在 push、PR 和手动触发时执行：

| 工作流 | 内容 |
| --- | --- |
| `ci.yml` | Python 静态检查、适合 CPU 的原有单元/API 测试，以及本地生成模型的真实推理比对 |
| `docker.yml` | 原生 AMD64、ARM64 镜像构建、OCI 源码标签和版本检查、服务启动及实际向量接口验证 |
| `secrets.yml` | 密钥泄漏扫描 |

CI 使用生成的微型 BERT/SentenceTransformers 模型，检查模型加载、真实 forward、归一化、不同输入、重复输入、float/base64 编码、鉴权和 HTTP 结果与直接模型调用的一致性。测试不下载额外 Hugging Face 模型；微型模型只验证执行链路，不衡量 BGE-M3 的语义效果。

Docker 镜像仅临时加载到 runner，不发布 registry，不上传镜像 artifact，不导出镜像缓存。成功或失败后均删除本次测试容器、镜像和 Buildx builder，并检查不存在。

上游原始 workflow 保存在 `.github/upstream-workflows/`，保留其 GPU、多模态、大模型及发布任务供参考；这些任务不直接在本项目运行。

## 验证范围

官方 AMD64 镜像中的 50 个应用 Python 文件已与固定上游提交逐一做 SHA-256 比对，无差异。官方镜像已离线加载 BGE-M3，并通过 6 条中英文输入的 1024 维、归一化检查。

两种自建镜像均已完成离线 BGE-M3 对比，使用同样的 6 条中英文输入、CLS pooling、CPU float32 和 2048 输入上限：

| 平台 | 本次运行方式 | 与官方输出的最大绝对误差 | 相似度 top-3 排序 |
| --- | --- | --- | --- |
| `linux/arm64` | Apple Silicon 原生 | `2.4586915969848633e-7` | 一致 |
| `linux/amd64` | Apple Silicon 模拟 | `0.0` | 一致 |

两种镜像均通过真实离线模型/API 验证，输出 1024 维归一化向量。此结果验证 BGE-M3 的文本 dense embedding；本次没有做性能比较。

完整结果、镜像 ID 和实际依赖版本记录在 [UPSTREAM.json](UPSTREAM.json)。本地测试容器已删除，保留已验证镜像。模型准备、构建日志和完整验证产物不纳入 Git 仓库；GitHub workflow 的实际运行在提交推送后触发。
