# EgoFound3R 2D 可视化交接

更新时间：2026-09-16。工作目录：
`/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/EgoFound3R-baselines`。

## 1. 交接状态

当前有两个不同状态的结果，必须分开解释：

1. `visualization/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/`
   是已经完成完备性核验的冻结 114 段结果。它包含 114 张中心帧 PNG、
   114 段按实际长度截断的 MP4、12,559 个解码展示帧，全部为 30 FPS；
   `completeness_audit.json` 为 `verified`，无错误。它仍是当前窗口身份和
   非重叠选择的权威来源。
2. `visualization/overlay_2d_contact_face_3x4_example_20260917/` 是新版 3×4
   面级渲染的单帧 pilot。审核目标是 H2O rank 17、中心帧 `000208`。
   该 pilot 已通过身份、数组形状、12 面板非空和 RGB 裁剪检查，但尚未启动
   114 段批量渲染。

旧版 114 gallery 的 2×4 点渲染效果不再作为未来批量的布局参考；其冻结窗口、
逐帧身份、插值审计和完备性结论仍然有效。

### 1.1 Git 快照与本次提交状态

- 项目路径：`/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/EgoFound3R-baselines`；
- 当前分支：`formal-hand-fix`；
- 本 2D 里程碑的父提交：
  `2fe84f1a940ca241b25e238b3483c60dcdc781ec`（3D 可视化提交）；
- 2D commit 是包含本文件的下一条限定路径提交。可运行
  `git log -1 --format='%H %D' -- visualization/2d_overlay/HANDOFF.md` 查询其精确 ID。

提交时共享工作树仍存在其他正式评测修改；2D commit 采用限定路径提交，不包含也不重置
这些修改。大型 PNG/MP4、逐帧 sidecar、逐段 report 和已淘汰 pilot 由
`visualization/.gitignore` 排除。

## 2. 项目路径、脚本路径与执行逻辑

### 2.1 脚本索引

| 阶段 | 脚本路径 | 作用 |
|---|---|---|
| 2D 单帧评分 | `formal_evaluation/select_2d_overlap_worker.py` | 使用 Ego/GT 投影生成 silhouette IoU、Boundary-F、中心误差和面积误差，并输出候选测量 |
| 评分入口审计 | `formal_evaluation/audit_2d_overlap.py` | 构造并核验六数据集评分任务输入与输出身份 |
| 帧轴核验 | `formal_evaluation/audit_2d_frame_axes.py` | 核对 sequence、source frame ID、RGB/GT/Ego 可读取性和帧轴关系 |
| 初版覆盖审计 | `formal_evaluation/audit_2d_video_coverage.py` | 保存早期基于登记帧号的覆盖核验入口，供历史结果追溯 |
| 实际窗口审计 | `formal_evaluation/audit_2d_video_coverage_real.py` | 从中心帧向两侧构窗，遇未解释源 ID 缺口截断并记录实际长度、逐帧 refs 和双手有效性 |
| 114 段选择 | `visualization/2d_overlay/build_visibility60_selected114.py` | 对 600 条完备性候选应用首尾双手有效、内部缺失不超过 60 帧和最大互不重叠动态规划 |
| 冻结 2×4 渲染 | `visualization/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/provenance/overlay114_remote.py` | 按冻结 manifest 渲染 114 张 PNG 和 114 段 MP4 |
| 冻结任务入口 | `visualization/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/provenance/overlay114_launch.py` | 登记六数据集 prepared/GT/Ego 输入并调度冻结渲染 |
| 冻结完整性检查 | `visualization/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/provenance/audit_overlay114.py` | 核对输出数量、视频可解码帧数、FPS、首尾身份和错误 |
| 新版 3×4 pilot | `visualization/overlay_2d_contact_face_3x4_example_20260917/render_example.py` | 渲染面级 GT/Ego/S²Contact/ContactOpt/InteractVLM 单帧示例 |

上述四项 audit、`select_2d_overlap_worker.py`、对应单元测试和当前 3×4 pilot renderer
与本交接目录共同构成独立 2D 提交，不能只提交 Markdown 而遗漏实际实现。

### 2.2 执行逻辑

1. 六个数据集分别运行 2D 单帧评分，并按组合分数保留各自前 100 个中心帧。
2. 对 600 个中心逐条核对 frame axis、RGB、GT、Ego、内参和手部有效性；以中心构造
   最长 300 展示帧的窗口，遇未解释源 ID 缺口立即截断。
3. 以实际长度窗口做完备性筛选：首尾双手有效，内部连续双手缺失不超过 60 帧。
4. 在通过完备性的窗口内按 dataset + sequence 求最大互不重叠集合，得到冻结 114 条。
5. 旧版 2×4 结果已经完整生成并核验；新版 3×4 当前只完成单帧 pilot。批量前先做四方法
   覆盖审计，再做一个完整实际长度 MP4 smoke，用户确认后才启动 114 段批量。

## 3. 第一部分：筛选前六数据集路径

筛选没有直接重写六个原始数据集。原始数据根只用于数据桥接和身份解析；实际 2D
单帧评分、窗口拼接和渲染读取的是已经物化的 60 帧 `window_input`，其中登记了
逐帧 RGB、geometry、frame ID、sequence/window/cache ID 和 calibrated intrinsics。

| 数据集 | 原始数据根 | 筛选前 prepared-window 根 |
|---|---|---|
| H2O | `/mnt/workspace/sjc/DATA/H2O/h2o_data` | `/mnt/workspace/sjc/eval_artifacts/prep_h2o_60f_strict_20260822T030835Z_5001_g1/window_inputs/h2o` |
| HOT3D | `/mnt/workspace/sjc/DATA/HOT3D/hot3d/hot3d/dataset` | `/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_20260818T0410Z_8b1a806/window_inputs/hot3d` |
| ARCTIC | `/mnt/workspace/sjc/DATA/EgoForce/ARCTIC` | `/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_arctic_20260820T000131Z/window_inputs/arctic` |
| OakInk-v2 | `/mnt/workspace/sjc/DATA/OakInk-v2` | `/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_oakink_v2_20260819T144050Z/window_inputs/oakink_v2` |
| TACO | `/mnt/cpfs/sjc/DATA/TACO_resized` | `/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_taco_20260822T101500Z_5001_g4_bin0/window_inputs/taco` |
| HOI4D | `/mnt/workspace/sjc/DATA/mnt-1/HOI4D` | `/mnt/workspace/sjc/eval_artifacts/prep_6dataset_60f_strict_hoi4d_20260822T015700Z_5001_g0/window_inputs/hoi4d` |

原始根来自 `formal_evaluation/taskctl.py` 的 six-dataset data-root map；prepared-window
根来自冻结 renderer 的 `overlay114_launch.py`，HOI4D 根来自同一批 strict-60f prep
任务登记。HOI4D 最终没有窗口通过非重叠选择，但它仍属于筛选前的六数据集输入，
不能从源清单中删除。

配套 GT 与 Ego 预测入口如下。`{dataset}` 使用小写数据集键；TACO GT 是单独产物：

- H2O/HOT3D/ARCTIC/OakInk-v2/HOI4D GT：
  `/mnt/oss/pre-train/ego/eval_artifacts/formal_standard_reports_10methods_20260824/{dataset}/gt_cache/{dataset}`；
- TACO GT：
  `/mnt/workspace/sjc/DATA/eval_artifacts/formal_contact_taco_metrics_20260831T234159Z/gt_cache/taco`；
- 冻结 renderer 当时读取的 Ego 根：
  `/mnt/cpfs/sjc/eval_artifacts/base3e_traincrop_512_stride5_1078f15_20260908/{dataset}/{dataset}/egofound3r/formal`；
- 当前已迁移的 Ego OSS 镜像：
  `oss://quic-pre-train/ego/eval_artifacts/p95_table_freeze_20260915/recompute_dependency/base3e_traincrop_512_stride5_1078f15_20260908/{dataset}/{dataset}/egofound3r/formal`。

执行时必须逐条读取 `window_input.json` 登记的 RGB/geometry 路径，不得用 raw root
自行猜测帧名，也不得用 3D gallery 的三张抽样 RGB 替代完整逐帧输入。

## 4. 第二部分：筛选逻辑

### 4.1 600 条候选输入

`visualization/2d_overlay/bracketable600_20260914.json.gz` 保存了选择器实际使用的
600 条完备性候选。解压后的 JSON 是 600 条记录，SHA-256 为：

`7e36ee5f1a51760304097c8095e6f1d1b235ed3241740c7f02fe071445a89280`

每条记录包含 dataset、dataset rank、sequence/window/cache/frame 身份、请求和实际
窗口边界、源 ID 缺口、逐帧 frame IDs/refs、双手有效性和
`bracketable_visibility`。这是 `selection_summary.json` 原先记录为
`/private/tmp/bracketable600.json` 的精确归档副本。

当前 checkout 中没有 `visualization/overlay_2d_candidates_20260913/`，也没有独立的
202 段冻结清单。因此后续不得从对话数字重建这两个清单；如需恢复 202 方案，必须
先找到原始 manifest 并重新核对身份和校验值。

### 4.2 从 2D 单帧到实际长度窗口

最初的单帧排序在六个数据集内分别进行，每个数据集保留前 100 帧，共 600 个中心。
单帧组合分数只由 2D 投影重叠组成：silhouette IoU、Boundary-F、中心误差和面积误差；
使用 778 MANO vertices 与 MANO faces 栅格化。Ego 使用其预测内参，GT 使用 calibrated
RGB 内参。不使用 3D W-MPJPE、既有 3D 片段选择或 3D 展示坐标。

每个中心的窗口构造以该帧为中心向两侧展开，目标展示长度为 300 帧；遇到未解释的
源 frame-ID 缺口就在缺口处截断。因此 `actual_length` 可以小于 300，且后续完整性
检查以该实际长度为准。300 帧/30 FPS 只是显示口径，不证明原始数据真实连续经过
10 秒。

### 4.3 完备性后再做非重叠

冻结选择遵循以下顺序：

1. 对六个数据集各自的前 100 个 2D 单帧中心构造实际长度窗口；遇到未解释的源 ID
   缺口即截断，不强制 300 帧。
2. 窗口首帧和尾帧双手都有效；内部“双手合并缺失”的连续区间最多 60 帧。
3. 在通过完备性的窗口中，按 dataset + sequence 求最大互不重叠集合；数量优先，
   再最大化原始 2D 分数，最后按 dataset rank 做确定性排序。

最终 `selected_manifest.jsonl` 共 114 条，SHA-256 为：

`6a95a1c01288fbe91589345e013bd7c96bcb038ebef2414bea140240824f4d2f`

| 数据集 | 窗口数 | 展示帧数 |
|---|---:|---:|
| ARCTIC | 9 | 2,700 |
| H2O | 26 | 3,116 |
| HOI4D | 0 | 0 |
| HOT3D | 1 | 300 |
| OakInk-v2 | 18 | 1,966 |
| TACO | 60 | 4,477 |
| 合计 | 114 | 12,559 |

冻结后的 ID 权威路径为：

`visualization/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/selected_manifest.jsonl`

其中每行保留 dataset、dataset rank、sequence/window/cache ID、center frame ID、实际窗口
边界、逐帧 frame IDs/refs 和缺失区间。选择汇总位于同目录 `selection_summary.json`；
跨产物索引位于 `visualization/2d_overlay/manifest_registry.json`。

## 5. 第三部分：可视化作图逻辑

### 5.1 冻结 2×4 gallery 的语义

旧版布局为两行四列：GT/Ego × geometry/contact distance/contact/visibility。

- GT 使用 calibrated RGB intrinsics、GT camera-space 778 MANO vertices。
- Ego 使用 camera-space 21 joints、195 markers 和固定 195→778 mesh 重建；
  195 个源 marker 锚点保持精确。该 mesh 不是原生 778 预测。
- Ego contact 是模型输出的 marker contact probability 经固定 195→778 标量映射；
  contact distance 是独立计算的 `vertex_contact_distance`。
- Ego visibility 是预测的 `marker_visibility` 经 195→778 映射，不等于 RGB 中人眼
  判断的整只手可见性。旧版 GT visibility 面板为空。
- 当当前帧 Ego K 无效时，显示使用最近的有效 stride-5 intrinsics anchor；该操作只
  影响显示，不改变正式评测有效性。

8 段视频共有 310 个 hand-side frame 使用了仅用于显示的内部缺失插值。插值对应
commit `8fc061a615895bd3b5a556f7387bae306e32d9db`：camera-space smooth fill，随后做
root-relative `[1,2,1]/4` 局部形状平滑。没有重跑推理、没有 root-UV 重求解，也没有
修改正式 hand validity。

旧版完整性证据：

- `selected_manifest.jsonl`：窗口和逐帧身份；
- `selection_summary.json`：选择规则和数据集统计；
- `run_contract.json`：渲染合同；
- `render_manifest.jsonl`：114 个输出身份；
- `completeness_audit.json`：最终计数和解码核验；
- `provenance/`：选择、渲染和审计代码快照。

大型媒体、逐帧 sidecar 和逐段 report 保留在同一 artifact 目录，但不进入 Git。

### 5.2 当前 3×4 面级方案

新版布局固定为：

| 行 | 第 1 列 | 第 2 列 | 第 3 列 | 第 4 列 |
|---|---|---|---|---|
| 1 | GT geometry | GT contact distance | GT contact | GT visibility |
| 2 | Ego geometry | Ego contact distance | Ego contact | Ego visibility |
| 3 | Ego contact | S²Contact contact | ContactOpt contact | InteractVLM contact |

渲染合同：

- 只在 RGB 图像矩形内绘制，letterbox 区域不画 overlay；
- contact 和 visibility 以 MANO 三角面绘制；每个面的数值为三个顶点值的均值；
- contact 使用 cyan=0、yellow=0.5、red=1；visibility 使用
  red=hidden、yellow=partial、green=visible；透明度约 0.48；
- 物体和双手共用逐像素 nearest-surface z-buffer；只显示当前最近的手部三角面；
- GT visibility 由 GT hand + object 几何的 z-buffer 推导；Ego visibility 保持模型
  预测含义，再经 195→778 映射和面均值显示；
- 第三行只比较右手 contact。

当前 pilot 身份：

- dataset/rank：H2O / 17；
- sequence：`subject4_ego/h1/5`；
- 60 帧窗口：`subject4_ego/h1/5:000156-000215`；
- cache：`abc07caace9823faa3ffe7ce`；
- 中心帧：`000208`，窗口索引 52；
- 输出：`h2o_r017_000208_3x4_face_contact_example.png`。

几何和 contact 来源：

- Ego：195 markers 重建 778 mesh；contact/visibility 从 195 映射至 778；
- S²Contact：正式产物同时提供 `hand_vertices_camera [60,2,778,3]` 和
  `vertex_contact_probability [60,2,778]`，使用其自身预测 mesh；
- ContactOpt：同样使用自身预测的 778 camera-space mesh 和 778 contact；
- InteractVLM：使用 topology-verified 6890→MANO-778 contact 字段；现有正式窗口
  没有经核验的预测手 mesh，因此只用 GT MANO 作为投影支撑，并在图中明确标注。

所有精确远端/OSS 输入路径、数组形状、阈值统计和 overlay pixel 计数记录在 pilot
JSON 中。该帧右手 `p >= 0.5` 顶点数为：Ego 0、S²Contact 170、ContactOpt 164、
InteractVLM 386。这个统计只用于核对当前帧，不是方法性能结论。

## 6. 批量前的未完成项

在启动 114 段新版批量前必须完成：

1. 对 114 个中心和全部展示帧审计 RGB、GT hand/object geometry、GT contact/distance、
   Ego prediction/K、S²Contact 778、ContactOpt 778、InteractVLM 778 的精确覆盖。
2. InteractVLM 当前只与 114 个中心中的 4 个中心精确重叠，只有 1 段覆盖完整实际长度。
   在补推理之前，其余面板必须保持 unavailable，禁止用 GT、其他方法或相邻窗口补值。
3. 决定是否接受“方法缺失时保留 RGB 并显示 unavailable”作为批量合同；如果要求所有
   114 段四方法全覆盖，则必须先补齐 InteractVLM，并重新做覆盖审计。
4. 确认 S²Contact/ContactOpt 新 778 产物对 114 个冻结窗口的覆盖；当前只核验了 pilot
   所在 H2O cache，不得由单窗通过推断全量通过。
5. 新批量使用新的唯一输出根，禁止覆盖已验证的 2×4 gallery。视频继续按每段
   `actual_length` 和 30 FPS 输出；不得把 300 展示帧或 30 FPS 解释为已验证的真实连续
   10 秒。

## 7. 推荐执行顺序

1. 读取 `selected_manifest.jsonl`，按其中 frame refs 做四方法覆盖审计，生成一个新的
   machine-readable preflight；任何缺失保持显式。
2. 用 H2O rank 17 生成一张中心 PNG 和一段完整实际长度 MP4 smoke，核对首尾帧、
   z-buffer、颜色、方法几何来源和无图外 overlay。
3. 用户确认 smoke 后再分片渲染；不要改变冻结窗口或重新按 3D 指标选帧。
4. 汇总后验证 PNG/MP4/sidecar/report 数量、所有 MP4 的解码帧数与 30 FPS、首尾帧
   identity、总展示帧数、非重叠和缺失方法原因，最后写 `COMPLETE` 或等价审计结果。

## 8. 本地核验命令

```bash
gzip -dc visualization/2d_overlay/bracketable600_20260914.json.gz | shasum -a 256
shasum -a 256 visualization/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/selected_manifest.jsonl
cmp visualization/2d_overlay/build_visibility60_selected114.py /private/tmp/build_visibility60_selected114.py
python3 visualization/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914/provenance/audit_overlay114.py \
  visualization/overlay_2d_top100_visibility60_maxnonoverlap114_fill8_20260914
```

复现单帧 pilot 时，先按 JSON 中的 `sources` 将输入暂存为
`window_input.json`、`rgb.png`、`geometry.npz`、`mapping.npz`、`gt.npz`、
`gt_contact.npz`、`ego_prediction.npz`、`ego_contact.npz`、`s2contact.npz`、
`contactopt.npz` 和 `interactvlm.npz`，再运行：

```bash
python visualization/overlay_2d_contact_face_3x4_example_20260917/render_example.py \
  --input-root <staged-input-dir> \
  --output-dir <new-output-dir>
```

DSW 分配是动态的。连接后应先按项目 `AGENTS.md` 用当前 instance 配置和 taskctl 核验
节点及路径。OSSFS 失效时不要自行重挂；使用经过授权的 macOS `ossutil` relay 或等待
平台恢复。
