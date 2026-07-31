# Podcast RAG Agent — 项目规格 (spec.md)

> 本文档用于指导后续 vibe coding。凡标注 `[假设:...]` 处为在缺乏明确信息下的推断，实现前请确认或修正。

---

## 1. 项目定义

一个**本地运行的多播客知识库工具**：给定播客 RSS 链接，自动下载、转录、解析全部往期，把内容变成**可跨集语义检索、可对话问答、可定位到秒并跳转播放**的知识库。给需要深度消化播客内容的个人用户（作者本人 / 学习者），解决"播客内容无法检索、无法定位、听过就忘"的问题。

---

## 2. MVP 范围 (In-scope)

明确要做的，仅以下 7 项：

1. **RSS 解析 + 音频下载**：输入一个 RSS 链接 = 导入该播客的往期若干集；音频落本地磁盘。
2. **WhisperX 转录**：产出 **word-level 时间戳**（跳转功能的地基）。
3. **轻量文本清洗**：过滤口头禅（uh/ah/um 等），**必须在词级时间戳结构上操作**（删词不影响其余词的时间戳）；`[假设:清洗版仅用于 embedding，展示/跳转用原文以保文字↔音频一致]`。仅做口头禅过滤，重复检测与广告识别不做（见 Future Work）。
4. **切 chunk + 绑定 metadata**：每个 chunk 携带 `start/end` 时间戳、`podcast_id`、集元信息；存入 pgvector。**chunk 时长须落在上下限区间**（`[假设:~20–90 秒]`，太短→听感碎、太长→点进去要等很久），不足则并入相邻、超限则切开；**每个 chunk 必须是一段时间连续的音频**（见 §4.4 铁律）。
5. **语义检索 + LLM 对话召回**：回答附**来源引用**（播客名 + 时间戳），来源可点击跳转播放。
6. **按播客自动分组**：检索可限定单个播客或全库；靠 `podcast_id`，导入时天然确定。
7. **检索层评估脚本**：产出 Recall@k / MRR，及不同 chunk 参数的对比表（回归用）。

**明确不做（写入 Future Work，见 §7）**：时间点→内容查询（纯 SQL 范围查询、非 RAG，且依赖"跟随播放进度显示当前文本"的额外前端同步，价值薄；核心的时间点相关交互已被"来源跳转"覆盖）、**广告识别过滤**（价值高但是独立 NLP 难题：无固定标记、位置不定，需关键词/LLM判断/RSS章节标记等方案，各有漏判误删；适合作为后期亮点，可配准确率评估）、**整段重复检测**（判定不可靠，易误伤正常强调/回顾）、结构化学习笔记（作为问答引擎的衍生展示层，MVP 阶段砍掉以聚焦核心；复用同一 chunk+跳转逻辑，后续可低成本补回）、自定义标签系统、说话人真名映射（speaker diarization 只预留字段）、流式输出、缓存层、意图路由、句子级引用跳转、大规模（先小批量验证）、前端升级为 React/Vue（MVP 用 HTML+TS 已足够，后端纯 JSON 保证届时零返工）。

---

## 3. 技术栈

| 组件 | 选型 | 为什么选它 |
|---|---|---|
| RSS 解析 | feedparser | RSS feed 天然是"一个 feed 挂多集"的结构，feedparser 直接给出集列表和 enclosure 音频 URL，无需自己解析 XML。 |
| 音频转录 | WhisperX（本地 GPU） | 唯一在此栈中原生输出 **word-level 时间戳**的选项；时间戳跳转是本项目核心亮点，没有词级时间戳整个亮点不成立。**注意：embedding 改走 API 后，转录成为唯一的本地 GPU 依赖**——它与"易分发/Mac 友好"目标存在张力（别人仍需 GPU 才能转录）。是否也把转录改为托管 Whisper API 见 §7 未决问题。**本机依赖链已实测可解析**（`uv pip install --dry-run`，Python 3.12/Windows）：`whisperx==3.8.6` pin `torch~=2.8.0`，需从 `https://download.pytorch.org/whl/cu128` 装 GPU 版（PyPI 默认是 CPU 版）；`ctranslate2==4.8.1`/`faster-whisper==1.2.1`。本机 GPU 是 Blackwell 架构，兼容性细节与实测风险见 §7。 |
| 后端 | FastAPI | 吐纯 JSON API，使前端可替换（HTML+TS → 未来 React 零返工）；契合作者当前 Python/FastAPI 学习目标，每行可自解释。 |
| 存储 + 向量检索 | PostgreSQL + pgvector | 向量与 metadata 存同一张表，检索时时间戳/来源随 chunk 一并返回；比独立向量库少一个组件，且 `podcast_id`/时间范围过滤直接用 SQL，无需二次系统。 |
| LLM | DeepSeek / OpenAI API（用户自配 key，接口抽象成可切换） | 只负责在检索结果之上生成回答，不本地跑；两家均可，接口抽象以便切换。**"帮助学习理解英文播客"的诉求落在这一层**：通过 prompt 控制用英文回答、解释术语，这是用户实际会读的内容。 |
| Embedding | **BGE-M3 via 托管 API**（DeepInfra 或 SiliconFlow 二选一，均已确认可用，OpenAI 兼容格式，接口抽象成可切换） | 与 LLM 解耦独立选型（embedding 决定检索准不准，是命脉）。**选托管 API 而非本地，核心是易分发**：①仓库轻——无模型权重、无 torch/CUDA 重依赖，`requirements.txt` 干净；②别人秒部署——clone→配 key→`compose up`，无需下载 2GB+ 权重、无需配 GPU 环境；③Mac 友好——纯 API 调用绕开 Apple Silicon 跑模型的坑；④与 LLM 同为 OpenAI 兼容格式，调用范式统一，"可切换"极自然（配 base_url + key 即可）；⑤成本可忽略，embedding 是最便宜的 API 之一。**仍用 BGE-M3 本体**（非改用 OpenAI embedding），选型、rerank 配套、M4 评估逻辑不变。**注意：embedding 只做后台向量检索，用户不可见，与"学习体验偏英文"无关——偏英文是 LLM 生成层职责。** 权衡：转录文本会发第三方平台，与 LLM 走 API 同性质，自用/demo 场景可接受。 |
| 前端 | **HTML + TypeScript**（Vite 构建，不引入框架） | 一两天可出活，`<audio>` + 点击跳转逻辑仅一行；先验证核心引擎。**用 TS 而非纯 JS 的收益**：把后端返回的 JSON（`{answer, sources[...]}`）定义为 TS `interface`，前后端数据契约获得类型保障——写错/漏字段编译期即报错，对前后端同一人开发尤其省 bug；代价仅是多一道 Vite 编译。**不引入 React/Vue**：项目护城河在 RAG 不在前端，框架留作 MVP 后升级（后端为纯 JSON，届时换框架零返工，见 §7）。 |
| 转录任务编排 | `[假设:轻量任务队列或简单状态表 + 后台 worker]` | 批量转录耗时（数小时级），必须异步、按集入库、可断点续传；不引入重型编排以免过度工程。方案待定，见 §7。 |
| 运行环境 | Docker Compose | 一键拉起 app + pgvector；契合作者 Docker 经验；`clone → 配 .env → compose up` 便于复现（面试可复现性）。 |

---

## 4. 数据模型 / 核心数据流

### 4.1 实体关系

```
podcast (播客/RSS feed)
  └── 1:N episode (单集)
        └── 1:N chunk (转录切片，检索与跳转的最小单元)
```

### 4.2 核心表（概念 schema，字段名待实现细化）

**podcast**
- `id`
- `rss_url`
- `name`
- `author`
- `cover_url`

**episode**
- `id`
- `podcast_id` → podcast
- `title`
- `published_date`
- `audio_local_path`（本地音频文件路径，前端跳转播放的目标）
- `transcript_status` `[假设:枚举 pending/transcribing/done/failed，支撑断点续传]`
- `language`（WhisperX 自动检测的语种，转录完成时写入；nullable——转录前或转录失败为空）
- `index_status`（枚举 pending/processing/done/failed，语义与 `transcript_status` 相同但独立跟踪切片/embedding 阶段，支撑断点续传）

**chunk**（项目地基，所有功能收敛于此）
- `id`
- `episode_id` → episode
- `podcast_id` → podcast（冗余，方便检索直接按播客过滤）
- `text`（原始转录片段）
- `embedding`（向量）
- `start` / `end`（秒，来自 WhisperX word-level 时间戳合并）
- `speaker` `[假设:预留字段，MVP 不填自动值或仅存 "speaker_1/2"]`
- `search_vector`（`tsvector`，从 `text` 生成的 DB generated column，GIN 索引；为未来 BM25/混合检索预留存储，本阶段只产出该列，检索/排序逻辑不在此阶段实现）

> **设计铁律**：一切功能都是 chunk 表的推论——跳转靠 `start`，来源溯源靠 `podcast_id`+集信息，分组靠 `podcast_id`。此表设计错误会连累所有功能，实现前必须先定死。

### 4.3 主数据流

```
RSS链接
 → feedparser 解析出集列表 + 音频URL
 → 下载音频到本地 (episode.audio_local_path)
 → [异步任务] WhisperX 转录 → word-level 时间戳列表 [(word, start, end), ...]
 → 【清洗】在词级列表上过滤口头禅(删词不动其余词时间戳)；严禁拼成纯文本后再做字符串清洗
 → 按 [假设:固定窗口+时长上下限] 切 chunk（连续词序列，时长夹在 [下限,上限]），合并词级时间戳为 chunk.start/end，绑定 metadata
 → embedding（[假设:用清洗版文本]）→ 存入 pgvector（[假设:原文另存以供展示/跳转对齐]）
 ─────────────────────────────────────────
 用户提问
 → 检索相关 chunk（可选 WHERE podcast_id 过滤）
 → chunk(text + metadata + 时间戳) 拼进 prompt 喂 LLM
 → 回答 + 来源列表（每个来源 = 原文片段 + 播客名 + 时间戳 + 播放按钮）
 → 点击来源 → audio.currentTime = chunk.start → 播放
```

### 4.4 跳转机制（明确约束）

- 可点击跳转的锚点是**检索出的 chunk（来源）**，**不是** LLM 生成的回答正文。
- 生成的回答是 LLM 综合改写的产物，不映射到具体音频时刻；来源 chunk 自带 `start`，才是跳转依据。
- **铁律：一个 chunk = 一段时间连续的音频**（`start` 到 `end` 连续）。切分时按时间顺序切连续的词，**严禁为"把讲同一话题的分散句子凑一起"而把不连续片段拼进同一 chunk**——否则点击跳转会在音频里乱跳。
- **多来源分散是正常的，不是 bug**：一次问答 top-k 召回的多个 chunk 可能来自播客不同位置（12:24 / 35:10 / 58:03），因为话题本就多处出现。呈现方式是**把它们作为多个独立来源分别列出，各自一个播放按钮，各自连续播放**；严禁把分散片段强行拼成连续播放。把分散硬凑成连续才是 bug。

---

## 5. 模块划分

| 模块 | 职责 | 关键产出 |
|---|---|---|
| `ingestion` | RSS 解析、音频下载、写入 podcast/episode | episode 记录 + 本地音频文件 |
| `transcription` | WhisperX 转录（异步、按集、可续传）、更新 `transcript_status` | word-level 时间戳列表 |
| `cleaning` | 在词级时间戳列表上过滤口头禅（uh/ah/um）；不做重复/广告 | 清洗后的词级列表 |
| `indexing` | 切 chunk、合并时间戳、embedding、写入 pgvector | chunk 表填充 |
| `retrieval` | 语义检索（含 podcast 过滤） | 带 metadata 的 chunk 列表 |
| `qa` | 组装 prompt、调 LLM、返回"回答 + 结构化来源" | JSON: `{answer, sources[]}` |
| `eval` | LLM 合成测试集 + 人工黄金集，算 Recall@k / MRR，输出参数对比表 | 指标表（回归可重跑） |
| `api` (FastAPI) | 暴露以上能力为纯 JSON 接口 | REST endpoints |
| `frontend` | 封面+集信息展示、问答界面、来源列表、`<audio>`、点击来源跳转播放；后端 JSON 镜像为 TS interface | 静态页面（TS/Vite）消费 JSON |

### 关键接口（草案，`[假设:具体路由/字段待定]`）

- `POST /podcasts` — 传 rss_url，触发导入
- `GET /podcasts/{id}/episodes` — 列出集及转录状态
- `POST /query` — `{question, podcast_id?}` → `{answer, sources[{text, podcast_name, episode_title, start, audio_url}]}`

---

## 6. 里程碑

每阶段有独立可验证的"完成"定义，不满足不进下一阶段。

### M1 — 数据地基跑通（单集）
先定死 chunk 表结构，跑通单集全链路。
- **完成定义**：给 1 集播客，能下载→转录→（清洗）→切 chunk 带正确 `start/end`→存入 pgvector；抽查若干 chunk，其 `start/end` 与实际音频内容吻合（人工抽查 3 处，误差在可接受范围 `[假设:±2 秒内]`）；**若已接入清洗，需额外验证清洗后时间戳未错位**。

### M2 — 检索 + 问答 + 跳转（核心引擎，单集）
- **完成定义**：对该集提问，返回带来源的回答；前端点击来源能让 `<audio>` 跳到对应 `start` 并播放，播放内容与来源文本对应。这是项目正身，必须最先跑通。

### M3 — 批量 + 分组（多集/多播客）
- **完成定义**：导入 1 个播客的 `[假设:10–20]` 集，转录异步进行、按集入库、中断可续传；能做"仅在某播客内检索"和"全库检索"；前端按播客分组展示。

### M4 — 检索评估
- **完成定义**：产出 Recall@k / MRR 脚本可一键重跑；至少覆盖 chunk size、overlap、有无 rerank 三维度的对比表；另有 `[假设:20–30]` 条人工黄金集作质量背书。
- **本项目特有维度**：chunk 长度不仅影响 Recall@k/MRR（检索精度），还影响点击跳转后的**音频听感**（太短碎、太长啰嗦）。评估时一并主观记录不同 chunk 长度下的听感，形成"检索精度 vs 听感"的权衡记录——这是普通 RAG 项目没有的叙事点。

---

## 7. 已知风险与未决问题

### 已知风险
- **转录是最大隐性成本**：单集 40–90 分钟，批量 = 成本 × 集数。口音、专业术语、语速会降低转录质量，直接拖累召回。缓解：本地 GPU（**已用 `nvidia-smi -L` 实测确认：RTX 5070 Ti Laptop，GB205 die，12GB GDDR7，60–115W TGP**——不是早先假设的桌面版 16GB）+ 异步任务 + 先小批量验证。笔记本功耗上限只有桌面同代卡的三分之一到一半，长音频批量转录更容易触发降频，M1 实测时应顺带记录单位时长转录耗时，供 M3 批量导入做时间预期。
- **WhisperX 依赖链在 Blackwell 架构（本机 GPU 所属代际，compute capability sm_120）上的兼容性需 M1 实测验证，不能只信文档**：torch 对 sm_120 的官方稳定支持从 2.7.0 才开始（此前只有 nightly，且经常报 cuDNN DLL 缺失）；WhisperX 当前 pin `torch~=2.8.0`，需显式从 `download.pytorch.org/whl/cu128` 装 GPU 版，PyPI 默认解析到的是 CPU 版。更深一层的坑在 `ctranslate2`（WhisperX 实际做转录推理的引擎，是独立于 torch 的另一套 CUDA kernel，torch 支持 Blackwell 不代表它也支持）：RTX 50 系历史上在 int8 量化下会报 `CUBLAS_STATUS_NOT_SUPPORTED` 崩溃（[OpenNMT/CTranslate2#1865](https://github.com/OpenNMT/CTranslate2/issues/1865)），已在 4.6.2 修复（禁用 sm_120 上的 int8 路径），当前可解析到的 `ctranslate2==4.8.1` 高于修复版本；但 cuDNN 版本要求在官方文档里前后矛盾（一处说 ctranslate2≥4.5.0 要求 cuDNN 9 且不兼容 cuDNN 8，另一处又写"cuDNN 8 for CUDA 12.x"），不可信，必须实测。**M1 第一步动作建议**：装好 `transcription` extra 后，先跑一个几行的探针脚本，加载模型时显式传 `compute_type="float16"`（不要用默认 int8），验证真实 GPU 上能跑通，再动手搭后续管道。
- **Windows+WSL2 部署前提：WSL2 默认内存上限不够 FunASR 加载**：转录 worker 已容器化为 `docker-compose.yml` 的 `worker` 服务（`backend/Dockerfile.worker`，装 `transcription` extra + `ffmpeg`，通过 `deploy.resources.reservations.devices` 拿 GPU）。FunASR 的 `AutoModel` 在同一进程里一次性加载三个模型（paraformer-large + VAD + punc-transformer），若 `.wslconfig` 没写 `memory=`，WSL2 默认只给宿主机内存的 50%——在 15.2GB 宿主机上约 7.6GB，不够用，会被 OOM kill（`docker events --filter event=oom` 可确认）。**修复**：在 `C:\Users\<you>\.wslconfig` 的 `[wsl2]` 段加 `memory=12GB`（或更高），然后 `wsl --shutdown` + 重启 Docker Desktop 使其生效；`docker info` 里的 `Total Memory` 应涨到对应值。仅 Windows+WSL2 宿主需要这一步，原生 Linux 不受此限制。
- **chunk 与时间戳对齐是最易做砸处**：若按字符粗暴切分会丢失词级时间戳映射，跳转与来源全部失效。M1 必须先解决。
- **清洗铁律（与时间戳对齐同为地基）**：任何文本清洗必须在词级时间戳列表 `[(word, start, end), ...]` 上操作——删词不影响其余词的时间戳；**严禁先拼成纯文本再做字符串清洗**，那会摧毁词与时间戳的对应关系，跳转全错位。另需决策：清洗版与原文若分开存（embedding 用清洗版、展示/跳转用原文），要保证两者能对齐回同一时间轴。
- **评估无 ground truth**：靠 LLM 合成测试集（需强制改写措辞，避免字面匹配虚高）+ 少量人工标注。生成层 LLM-as-judge 结果需打折看待，检索层指标更可信。
- **embedding 模型不可中途更换**：同一向量库内不同模型的向量空间不通用。一旦更换 embedding 模型或托管平台（可能导致维度或向量分布变化），pgvector 中已存向量全部作废，须整库重新 embedding。可切换接口是为"下次建库时换"，不是同库混用；M4 若对比不同 embedding，需各建一套向量重跑（可作为评估维度之一）。

### 未决问题 (Open Questions)
1. **LLM 已定** DeepSeek / OpenAI（接口可切换）；**Embedding 已定 BGE-M3 via 托管 API**，两个候选提供商均已确认可用、二选一待定：**DeepInfra**（`https://api.deepinfra.com/v1/openai/embeddings`，model 名 `BAAI/bge-m3`）、**SiliconFlow**（model 名 `BAAI/bge-m3` 或 `Pro/BAAI/bge-m3`，最大 8192 token）；两家均 OpenAI 兼容格式、dense 输出 **1024 维**，与 pgvector 向量列维度对齐确认无误（pgvector HNSW 全精度支持最多 2000 维，1024 维余量充足）。最终 embedding 选型仍可由 M4 评估复核，但换模型意味着换 API 且整库重建（见风险）。
2. **转录是否也改托管 API？**（新张力）embedding 走 API 后，WhisperX 本地转录成为唯一 GPU 依赖，与"易分发/Mac 友好"目标冲突。选项：①保留本地 WhisperX（转录质量/成本可控，但别人需 GPU）；②改托管 Whisper API（彻底无本地依赖，但需确认该 API 是否提供 **word-level 时间戳**——这是不可退让的硬需求，多数托管 Whisper 只给段级时间戳，若无词级则不可用）。`[假设:MVP 先保留本地 WhisperX，把"完全无 GPU 部署"列为 enhancement]`
3. **chunk 切分策略**：语义切分 vs 固定窗口+overlap？初始 size/overlap 取值？以及**时长上下限**取多少（`[假设:~20–90 秒]`）？注意本项目切分有**双重目标**——既要检索精度（偏向短 chunk），又要跳转听感（偏向长 chunk），二者拉扯，无理论最优值，靠 M4 评估 + 听感调。`[假设:先固定窗口起步、带时长上下限约束，用 M4 调优]`
4. **异步转录方案**：轻量队列（如 RQ/Celery）还是"状态表 + 后台 worker"？倾向后者以避免过度工程，待定。
5. **每个 RSS 默认导入多少集**：全部往期还是最近 N 集？`[假设:MVP 先限最近 10–20 集]`
6. **时间戳对齐可接受误差**是多少（影响 M1 完成判定）？`[假设:±2 秒]`
7. **Late chunking 已评估、暂不采用**：late chunking 需要 embedding 后端在 pooling 之前暴露 token-level 输出（整篇先编码，再按 chunk 边界 mean-pool），而当前 DeepInfra 托管的 BGE-M3 走标准 OpenAI 兼容 `/embeddings` 接口，只返回单一 pooled 向量，没有 token-level 输出。要拿到 token-level 输出，只能改用 Jina 的 API（目前唯一原生支持 `late_chunking` 参数的托管方案）或自托管 BGE-M3——两者都会重新打开 §3 已经做过的决定（embedding 选托管 API 正是为了避免第二个本地 GPU 依赖）。若未来重新评估"是否自托管 embedding"，可一并重新评估 late chunking。