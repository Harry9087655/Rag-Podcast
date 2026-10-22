# QA 评测数据集

人工标注的 QA 评测集，用来评 `qa/core.py::answer()`。评分、门禁、CLI 不在这里，由 `scoring` / `runner` 子任务实现。

| 文件 | 作用 |
|---|---|
| `dataset_v1.json` | 标注的 case |
| `schema/dataset_v1.schema.json` | case 的 JSON Schema |
| `check_dataset.py` | schema 校验 + 一致性检查 + 打印 fingerprint |
| `_inventory/` | 标注用的 DB 快照，不提交（见下） |

## 工作流

所有命令在 `backend/` 下运行。

```bash
# 1. 导出 DB 快照到 evals/qa/_inventory/（chunks / episodes / segments）
uv run python scripts/dump_eval_inventory.py

# 2. 对照快照编辑 dataset_v1.json

# 3. schema + 一致性检查，打印 fingerprint
uv run --with jsonschema python ../evals/qa/check_dataset.py

# 4. 去 DB 核对每个 gold（需要 DB 在线）
uv run python scripts/verify_eval_gold.py
```

`_inventory/` 里的三份快照：

- `chunks.json`：每个 chunk 的 `id` 和原文，标 `gold_chunks` 用。
- `segments.json`：每集清洗后的逐句 `start` / `end` / `text`，标 `gold_segments` 用。文本与 full-context 臂喂给模型的文本同源。
- `episodes.json`：episode 和 podcast 的 id、语言、标题，写 `scope` 用。

## case 字段

| 字段 | 说明 |
|---|---|
| `id` | 稳定 slug，`<类别>_<主题>`，如 `parameter_toast_grow_price`。baseline 按 id 对齐，改 id 等于换一条 case |
| `tags` | 受控分类词，schema 里枚举锁定（见下） |
| `critical` | 高危 case，约占 20%（见下） |
| `question` | 用户提问，与语料同语言 |
| `expected_arm` | `full-context` 或 `search` |
| `scope` | 传给 `answer()` 的范围参数 |
| `gold_chunks` | search 臂的 gold：chunk id 列表；其他臂为 `null` |
| `gold_segments` | full-context 臂的 gold：`[{start, end}]`；其他臂为 `null` |
| `reference_answer` | 完整正确的参考答案，供 judge 对照 |
| `required_claims` | 正确答案必须包含的原子事实，供 judge 判完整性 |
| `conflicts` | 可选，已知冲突 |

可用 tags：`single_fact` `definition` `parameter` `cross_document` `cross_podcast` `comparison` `multi_hop` `temporal` `table` `code` `ambiguity` `no_knowledge` `near_entity` `typo` `long_question`。

两个「跨文档」类别的区别：

- `cross_document`：同一播客内、跨多个 episode 的证据组合，scope 用 `{"podcast_id": N}`。当前语料每个播客只有一集，暂未覆盖，等多集语料上线后再出题。
- `cross_podcast`：跨不同播客的证据组合，scope 用 `{}`（全库检索）。

## scope 决定走哪条臂

`answer()` 按 `scope` 路由，`expected_arm` 必须和 `scope` 对得上：

| scope | 实际走的臂 |
|---|---|
| `{"episode_id": N}` | full-context（transcript 放得进 input budget 时） |
| `{"podcast_id": N}` | search，限定在该播客 |
| `{}` | search，全库（跨播客） |

当前语料每集都放得进 budget，所以**写 `{episode_id}` 就一定走 full-context**。要测 search 臂，用 `{podcast_id}` 或 `{}`。每个播客目前只有一集，`{podcast_id}` 等于在这一集里检索。

## 标注规范

1. **gold_chunks**：在 `chunks.json` 里找原文含答案证据的 chunk，记它的 `id`。答案跨多个 chunk 就都记上。
2. **gold_segments**：在 `segments.json` 里找含答案证据的句子，**原样复制** `start` 和 `end`，不要四舍五入，也不要自己合并时间窗。答案跨多句就逐句列出。
3. **reference_answer**：与问题同语言，1–3 句，忠于原文。
4. **required_claims**：拆成原子事实，一个数字、日期、名字或结论一条。
5. **no_knowledge（应弃答）**：问语料里没有的内容。`gold_chunks` 和 `gold_segments` 都为 `null`，`required_claims` 为 `[]`，`reference_answer` 写一句弃答（如「资料中没有相关信息」），tag 含 `no_knowledge`。出题前先在快照里搜一遍，确认关键词确实不存在。
6. **critical**：答案是精确数字或名称、答错会误导用户时才标 `true`，整体控制在 20% 左右。

`check_dataset.py` 会强制检查这些规则：

- full-context：`gold_chunks` 为 `null`，`gold_segments` 非空，`scope` 必须有 `episode_id`。
- search：`gold_chunks` 非空，`gold_segments` 为 `null`。
- no_knowledge：两个 gold 都为 `null`，`required_claims` 为空。
- `id` 不能重复。

## fingerprint

`dataset_fingerprint` 是数据集内容的哈希。改动数据集后，旧的 observation 和 baseline 会因指纹不符被 scoring 拒绝，避免拿新数据集和旧基线比较。

```python
def dataset_fingerprint(dataset: dict) -> str:
    canonical = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

指纹对解析后的 JSON 计算，所以改缩进、换行或键顺序不影响结果，只有内容变化才会改变指纹。`scoring` 模块必须使用完全相同的算法，三个参数一个都不能变。

## 已知缺口

- **`table`、`code` 没有覆盖**：当前语料是财经和访谈播客，没有表格或代码，留到受控语料阶段补。
- **`cross_document`（同播客跨集）没有覆盖**：当前语料每个播客只有一集，跨 episode 的题目要等多集播客上线后再补。现有跨播客题目用 `cross_podcast` 标记。
- **ASR 转写错误**：英文人名拼法不一致（如 Kalachi/Calacci、Thelot/Tullo、Mamdani/Mondani），中文有同音错字（如「九七」应为「久期」）。出题时选原文表述清楚的事实。
