# Podcast RAG Agent — 项目骨架搭建计划

> 本文件是独立、可自解释的执行计划：即使在全新对话里打开这一份文件，也能知道要做什么、为什么、怎么做。所有决策均来自与用户的多轮讨论 + 本机实测验证（详见 `spec.md`），这里不再重复论证，只落地执行细节。

---

## Context（为什么做这件事）

`spec.md` 里的架构决策和依赖兼容性已经全部讨论/实测完成（uv、npm、SQLAlchemy async + pgvector、docker-compose 只含 db+api 两服务、WhisperX 依赖链在本机 RTX 5070 Ti Laptop 上的兼容性等）。现在要把这些决策**第一次落地成实际的目录结构和可运行环境**——目标不是实现任何业务逻辑（RSS 解析、转录、切 chunk 等留给 M1 里程碑），而是**搭出骨架并证明"环境本身是通的"**：`docker compose up` 能起来、FastAPI 能连上 pgvector 数据库、前端能跑起来并成功调到后端。这是所有后续开发的地基，一旦骨架里环境配置有问题（依赖版本冲突、容器连不上库、CORS 不通），会持续拖累后面每个模块的开发。

**骨架深度**（已与用户确认）：目录结构 + 真实依赖清单（pyproject.toml/package.json 里是会用到的真实包，不是占位符）+ 能跑通的最小环境（FastAPI 健康检查接口连通 DB + Vite 页面成功 fetch 到后端）。各业务模块（`ingestion`/`transcription`/`cleaning`/`indexing`/`retrieval`/`qa`/`eval`）只建目录和空 `__init__.py`，不写业务逻辑。

---

## 已确认的环境基线（本机已实测，直接采用）

| 项 | 版本/事实 | 来源 |
|---|---|---|
| Python | 3.12.7 | 已装 |
| uv | 0.11.7 | 已装 |
| Node | v24.18.0 | 已装 |
| npm | 11.16.0 | 已装 |
| Docker | 29.4.2 / Compose v5.1.3 | 已装 |
| GPU | RTX 5070 Ti Laptop（GB205, Blackwell, sm_120），12GB GDDR7，60–115W TGP | `nvidia-smi -L` 实测 |
| 驱动 | 596.13，支持 CUDA 13.2 | `nvidia-smi` 实测 |
| 核心后端依赖可解析 | `fastapi==0.139.0` `sqlalchemy==2.0.51`(+asyncio) `asyncpg==0.31.0` `pgvector==0.5.0` `feedparser==6.0.12` `alembic==1.18.5` `pydantic-settings==2.14.2` `httpx==0.28.1` | `uv add` 实测于 scratchpad 一次性项目，Python 3.12/Win 无冲突 |
| 转录依赖可解析 | `whisperx==3.8.6` pin `torch~=2.8.0`；GPU 版需 `--index-url https://download.pytorch.org/whl/cu128`（PyPI 默认是 CPU 版）；`ctranslate2==4.8.1` `faster-whisper==1.2.1`（均高于 Blackwell int8 修复版本 4.6.2） | `uv pip install --dry-run` 实测 |
| pgvector 镜像 | `pgvector/pgvector:pg16`（或 `0.8.4-pg16`），HNSW 全精度最多 2000 维，1024 维（BGE-M3）余量充足 | Docker Hub 确认 |
| Embedding API | DeepInfra `BAAI/bge-m3` @ `https://api.deepinfra.com/v1/openai/embeddings`；或 SiliconFlow `BAAI/bge-m3`/`Pro/BAAI/bge-m3`；均 1024 维、OpenAI 兼容格式 | 官方文档确认，二选一待用户实际配置时决定 |
| Vite | 需要 Node ≥20.19/22.12，本机 24.18 满足 | 官方文档确认 |

**未在此阶段验证、留到 M1 实测**：ctranslate2 在本机 GPU 上是否真的能以 `compute_type="float16"` 跑通（依赖解析成功 ≠ 运行时 kernel 正确，见 spec.md §7 风险）。骨架阶段的 backend Dockerfile **不包含** WhisperX/torch（转录跑在 host，见下）。

---

## 根目录结构

```
rag-podcast/
├── backend/
├── frontend/
├── data/                    # 音频落盘根目录，.gitignore 忽略内容
│   └── .gitkeep
├── docker-compose.yml
├── .env.example
├── .gitignore
└── spec.md                  # 已存在，不动
```

---

## backend/ 详细结构

**文件分两类，动手前先分清楚**：
1. **业务模块骨架**（`ingestion`/`transcription`/`cleaning`/`indexing`/`retrieval`/`qa`/`eval` 这几个目录）——只建目录 + 空 `__init__.py`（0 字节），不写任何逻辑。这些留给 M1/M2/M4 里程碑实现，现在写了也是要推倒重来的占位代码。
2. **环境骨架本身**（`config.py`/`db.py`/`models/*`/`api/health.py`/`main.py`）——**不是空文件**，是真实能跑的最小代码，专门用来证明"环境本身通不通"（FastAPI 能起来、能连上 pgvector 容器）。具体代码见本节末尾"健康检查连通性验证"。

**`__init__.py` 是什么**：Python 用它标记一个文件夹是"可以被 import 的包"。比如 `rag_podcast/ingestion/__init__.py` 存在，才能在别处写 `from rag_podcast.ingestion import xxx`。没有它 Python 3 也能勉强当"隐式命名空间包"处理，但容易在测试发现、打包、IDE 补全上出怪问题，所以显式建一个是约定俗成的稳妥做法。现在每个业务模块目录下的 `__init__.py` 都是空文件，纯粹"占坑声明这是个包"；等 M1 往 `ingestion/` 里加 `rss.py` 等真实代码时，才会在 `__init__.py` 里写 `from .rss import parse_feed` 这类导出语句。

**`pyproject.toml` 从哪来**：`uv init` 会自动生成一份最小骨架（只有 `[project]` 里的 name/version/requires-python，`dependencies = []` 是空的）。下面写的内容不是从零替代它，而是**在 uv 生成的文件基础上编辑追加**——把真正要用的依赖列表、`transcription` optional group、torch 走 cu128 索引这几段配置填进去，顺序是：`uv init` 生成骨架 → 编辑同一个文件补内容 → `uv sync` 按内容装依赖。

```
backend/
├── pyproject.toml
├── .python-version          # 内容: 3.12
├── alembic.ini
├── alembic/
│   ├── env.py                # 由 `alembic init` 生成后接入 SQLAlchemy 异步引擎
│   └── versions/
│       └── 0001_init.py      # 手写：CREATE EXTENSION vector; + podcast/episode/chunk 建表
├── src/
│   └── rag_podcast/
│       ├── __init__.py
│       ├── main.py           # FastAPI app + CORS + /health 路由
│       ├── config.py         # pydantic-settings：读 .env（DATA_DIR/DATABASE_URL/EMBEDDING_*/LLM_*）
│       ├── db.py              # async engine + get_session()
│       ├── models/
│       │   ├── __init__.py
│       │   ├── podcast.py     # id/rss_url/name/author/cover_url
│       │   ├── episode.py     # id/podcast_id/title/published_date/audio_local_path/transcript_status(Enum)
│       │   └── chunk.py       # id/episode_id/podcast_id/text/embedding(Vector(1024))/start/end/speaker
│       ├── ingestion/__init__.py       # 空，M1 实现
│       ├── transcription/__init__.py   # 空，M1 实现
│       ├── cleaning/__init__.py        # 空，M1 实现
│       ├── indexing/__init__.py        # 空，M1/M2 实现
│       ├── retrieval/__init__.py       # 空，M2 实现
│       ├── qa/__init__.py              # 空，M2 实现
│       ├── eval/__init__.py            # 空，M4 实现
│       └── api/
│           ├── __init__.py
│           └── health.py      # GET /health -> 查一次 SELECT 1 证明 DB 连通
├── scripts/
│   └── run_transcription_worker.py   # 空壳 + TODO 注释，M1 实现；host 上用 `uv run --extra transcription` 跑
├── tests/
│   └── __init__.py
└── Dockerfile                 # 只装核心依赖（无 torch/whisperx），跑 uvicorn
```

**pyproject.toml 关键内容**：
```toml
[project]
name = "rag-podcast-backend"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi", "uvicorn[standard]",
  "sqlalchemy[asyncio]", "asyncpg", "pgvector",
  "feedparser", "alembic", "pydantic-settings", "httpx",
]

[project.optional-dependencies]
transcription = ["whisperx", "torch", "torchaudio", "ctranslate2", "faster-whisper"]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```
这样 `uv sync`（api 容器用）只装核心依赖；host 上跑 `uv sync --extra transcription` 时 torch/torchaudio 自动从 cu128 索引解析，不需要手动二次 reinstall。

**models/chunk.py 要点**：`embedding` 用 `pgvector.sqlalchemy.Vector(1024)`；`text` 只存原文一份（清洗版是索引期间的临时产物，不落库，见此前讨论结论）。

**0001_init.py 迁移要点**：第一条语句 `op.execute("CREATE EXTENSION IF NOT EXISTS vector")`，然后建三张表，`chunk.embedding` 用 `Vector(1024)` 列类型。

**健康检查连通性验证：完整代码与请求链路**

四个文件各自的职责、以及它们如何串成一条"能证明环境真的通了"的链路：

```python
# config.py —— 读 .env
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    class Config:
        env_file = ".env"

settings = Settings()
```

```python
# db.py —— 建立到 pgvector 容器的异步连接
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from .config import settings

engine = create_async_engine(settings.database_url)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_session():
    async with async_session() as session:
        yield session
```

```python
# api/health.py 
from fastapi import APIRouter, Depends
from sqlalchemy import text
from ..db import get_session

router = APIRouter()

@router.get("/health")
async def health(session=Depends(get_session)):
    await session.execute(text("SELECT 1"))   # 真的向数据库发一条查询
    return {"status": "ok", "db": "connected"}
```

```python
# main.py —— 挂路由 + 允许前端跨域调用
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.health import router as health_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # 两个都要列，浏览器视为不同 origin
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
```

**请求链路**：浏览器（Vite `:5173`）执行 `fetch('http://localhost:8000/health')` → FastAPI 收到请求 → 用 `get_session()` 建一个到 `db` 容器的连接 → 真的执行一句 `SELECT 1` → 成功就返回 `{"status":"ok","db":"connected"}`，失败（比如 `db` 容器没起来、密码不对）就直接抛异常、FastAPI 返回 500。**失败路径很重要**——证明这不是个假的"永远绿灯"健康检查，它确实在验证数据库连得上；前端页面拿到 500 会显示报错文字而不是"connected"，这样一眼能看出骨架哪一环没通。

**Dockerfile 要点**：基于 `python:3.12-slim`，装 uv，`COPY pyproject.toml uv.lock README.md`（README.md 必须一起复制，因为 `pyproject.toml` 里 `readme = "README.md"`，hatchling 校验这个字段时如果文件不存在会直接构建失败），然后 **先 `COPY src ./src` 再 `RUN uv sync --frozen --no-dev`**（顺序不能反——`uv sync` 会把 `rag_podcast` 这个包装进虚拟环境，如果这时候 `src/` 还没复制进去，装出来的包是空的，容器启动时会报 `ModuleNotFoundError: No module named 'rag_podcast'`），最后 `CMD uvicorn rag_podcast.main:app --host 0.0.0.0`。

---

## frontend/ 详细结构

```
frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts      # server.host 固定成 127.0.0.1，见下方说明
├── index.html
├── src/
│   ├── main.ts        # fetch('http://localhost:8000/health') 并把结果渲染到页面上
│   └── types.ts        # 先放一个占位 HealthResponse interface，M2 再扩充 QueryResponse/Source
└── public/
```

`npm create vite@latest frontend -- --template vanilla-ts` 起步，删掉模板自带的 counter/logo 示例代码（`counter.ts`、`style.css`、`src/assets/*`、`public/icons.svg`），`main.ts` 内容只做一件事：调后端 `/health`，成功就在页面显示"Backend + DB connected ✅"，失败显示错误信息——前端骨架阶段唯一要证明的事就是"能跨域调通后端"。

**`vite.config.ts` 为什么要显式写 `server: { host: '127.0.0.1' }`**：本机 `localhost` 优先解析到 IPv6（`::1`），Vite 默认绑定行为在这台机器上只监听了 IPv6，导致 `curl http://localhost:5173` 或某些工具连不上（`netstat` 能看到只有 `[::1]:5173 LISTENING`，没有 IPv4 那一行）。显式指定 `127.0.0.1` 强制走 IPv4。对应地，后端 `main.py` 的 CORS `allow_origins` 需要同时列出 `http://localhost:5173` 和 `http://127.0.0.1:5173` 两个来源，因为浏览器把这两个当成不同的 origin。

---

## 根目录文件

**docker-compose.yml**（实际落地版本，和最初草稿有两处不同，见后面说明）：
```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes: ["pgdata:/var/lib/postgresql/data"]
    ports: ["5433:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5
  api:
    build: ./backend
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
      DATA_DIR: /data
    volumes: ["./data:/data"]
    ports: ["8000:8000"]
    depends_on:
      db:
        condition: service_healthy
volumes:
  pgdata:
```

**和最初草稿的两处差异，为什么改**：
1. **端口从 `5432:5432` 改成 `5433:5432`**：本机已经装了本地 PostgreSQL，占用了宿主机 5432 端口，Docker 起不来会直接报端口冲突（`docker ps` 会显示 db 容器完全没有端口映射）。改成 `5433:5432` 只是宿主机对外端口换了，容器内部还是 5432，`api` 服务走内部网络用 `db:5432` 访问，不受影响。**只有从宿主机直接连 DB（比如用 psql 客户端调试）才需要记得用 5433。**
2. **加了 `healthcheck` + `depends_on.condition: service_healthy`**：如果不写，`depends_on: [db]` 默认只保证"db 容器的进程启动了"，不保证"PostgreSQL 已经能接受连接"——PostgreSQL 启动后还有一段初始化时间，这段时间内 `api` 容器如果已经在尝试连接就会失败。`pg_isready` 是一个轻量级的"能不能连上"检测，Compose 每 5 秒探测一次，探测成功才把 db 标记为 healthy，`api` 容器等到这个状态才会启动。
3. **`api` 服务多了一段 `environment` 覆盖**：`.env` 里 `DATABASE_URL` 写的是 `localhost`（host 上跑 alembic 等命令时用），但容器网络里 `db` 服务不叫 `localhost`，而是用 compose 的服务名 `db` 访问。所以 `docker-compose.yml` 单独给 `api` 服务覆盖了 `DATABASE_URL`（指向 `db:5432`）和 `DATA_DIR`（容器内挂载路径是 `/data`，不是宿主机的 `./data`）。

**.env.example**：
```
DATA_DIR=./data
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpass
POSTGRES_DB=ragpodcast
DATABASE_URL=postgresql+asyncpg://raguser:ragpass@db:5432/ragpodcast
EMBEDDING_BASE_URL=https://api.deepinfra.com/v1/openai
EMBEDDING_API_KEY=
EMBEDDING_MODEL=BAAI/bge-m3
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=
LLM_MODEL=deepseek-chat
```
注意 `DATABASE_URL` 里的 host 在容器网络里是 `db`（compose service 名），host 上跑 alembic/转录脚本时需要用 `localhost` 覆盖——`.env` 实际值里按 host 场景写 `localhost`，compose 的 `api` 服务通过 `environment` 覆盖成 `db`（或者更简单：`.env` 统一写 `localhost`，`docker-compose.yml` 里给 `api` 单独加一条 `environment: DATABASE_URL=postgresql+asyncpg://...@db:5432/...` 覆盖 env_file 的值）。

**.gitignore**：`data/`、`.venv/`、`node_modules/`、`.env`、`__pycache__/`、`*.pyc`、`dist/`。

---

## 执行步骤顺序

- [x] 1. 根目录建 `data/.gitkeep`、`.gitignore`、`.env.example`
2. `backend/`：
  - [x] `uv init`
  - [x] 写 `pyproject.toml`（含 optional group + uv sources）
  - [x] `uv sync`
  - [x] 建 `src/rag_podcast/` 全部子目录和空 `__init__.py`
  - [x] 写 `config.py`
  - [x] 写 `db.py`
  - [x] 写 `models/*`
  - [x] 写 `api/health.py`
  - [x] 写 `main.py`
  - [x] `alembic init alembic`
  - [x] 改 `alembic/env.py` 接入异步引擎
  - [x] 手写 `0001_init.py`
  - [x] 写 `Dockerfile`
- [x] 3. `frontend/`：`npm create vite@latest frontend -- --template vanilla-ts` → 精简掉模板自带的示例代码 → 写 `main.ts` 探针逻辑 + `vite.config.ts`
- [x] 4. 根目录写 `docker-compose.yml`
- [x] 5. `docker compose up --build` → 确认 `db` 健康、`api` 能起来
- [x] 6. `curl http://localhost:8000/health` → 返回 `{"status":"ok","db":"connected"}`
- [x] 6.5. `docker compose exec api alembic upgrade head` → 建表（`podcast`/`episode`/`chunk`/`alembic_version`）。**注意**：迁移是在 `api` 容器内跑的，不是在 host 上跑——host 上跑会因为宿主机 5432 端口被本地 PostgreSQL 占用、compose 只能把 db 映射到 5433，如果 host 上 `.env` 的 `DATABASE_URL` 还写着 `localhost:5432` 就会连错端口。容器内部走的是 `db:5432`，不受宿主机端口冲突影响，最简单可靠。
- [x] 7. `cd frontend && npm install && npm run dev` → 打开 `http://127.0.0.1:5173` 确认能 fetch 到后端 `/health` 并显示 "Backend + DB connected ✅"

---

## 验证清单（骨架完成的定义）

- [x] `docker compose up` 一键拉起 `db` + `api`，无报错
- [x] `GET /health` 返回 200，包含 DB 连通确认（证明 SQLAlchemy async + asyncpg + pgvector 扩展这条链是通的）
- [x] `alembic upgrade head` 能成功建表（含 `vector` 扩展和 `chunk.embedding` 列）——`docker compose exec api alembic upgrade head`
- [x] 前端 `npm run dev` 起来后，页面能跨域 fetch 到后端 `/health` 并展示结果（证明 CORS 配置无误）
- [ ] host 上 `uv sync --extra transcription` 能成功解析安装（不要求此时就跑通 WhisperX 真实转录，那是 M1 的事）——**尚未执行**

**骨架搭建阶段结束**（步骤 1-7 全部完成，2026-07-12）。日常开机/关机流程见下方"日常启停"。

## 明确不做（留给里程碑）

- 不写任何 ingestion/transcription/cleaning/indexing/retrieval/qa/eval 的业务逻辑
- 不做 WhisperX 真实转录探针（等 M1）
- 不做前端问答界面/来源列表/audio 跳转（等 M2）
- 不引入 LangChain（此前已讨论确认，全项目都不需要）

---

## 日常启停

**启动**（开发时）：
```
docker compose up --build   # 首次或改了 backend 代码后加 --build，否则可以省略
cd frontend && npm run dev  # 另开一个终端
```
浏览器打开 `http://127.0.0.1:5173`，看到 "Backend + DB connected ✅" 说明骨架全通。

**关闭**（收工时）：
```
# 关前端：在跑 npm run dev 的终端按 Ctrl+C，或者杀掉 node 进程
# 关后端 + 数据库：
docker compose down          # 停容器，pgdata volume 保留，数据不丢
docker compose down -v       # 连 volume 一起删（下次要重新 alembic upgrade head）
```
默认用 `docker compose down`（不带 `-v`）——数据库里的表结构和数据下次开机还在，不用重新建表。
