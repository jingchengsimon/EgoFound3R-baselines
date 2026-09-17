# 3D 10s 可视化说明

## 当前结果

- 本地项目：`/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/EgoFound3R-baselines`
- 冻结代码与清单：`visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914`
- 筛选后完整 ID：`visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest_hydrated.jsonl`
- 正式 run ID：`visualization-four-10s-gallery-endpoint104-5001-auxmethods-hawor-native-handzoom-panels-full104_auxmethods_handzoom_panels_v5_5001_r1-20260916`
- 远端正式结果：`/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_endpoint104_hawor_native_unmasked_20260915_full104_auxmethods_handzoom_panels_v5_5001_r1`
- 本地轻量副本：`visualization/gallery_10s_endpoint104_hawor_native_unmasked_20260915_full104_auxmethods_handzoom_panels_v5_5001_r1`

5001 当前已断开，本次文档整理没有重新访问节点。以下状态来自断联前已经冻结并校验的结果：`COMPLETE` 存在，`summary.json` 为 `status=complete`，104/104 段成功、失败数为 0。总览包含 104 张 PNG 和 104 个 H.264 MP4；每个视频为 300 帧、30 FPS、10 秒。远端另有 5,136 张逐方法逐视角 2048×2048 panel PNG。本地轻量副本包含 104 张总览 PNG、104 个 MP4、完成元数据及 SHA-256 传输清单，不包含 panel gallery、runtime、logs 和 control。

## 一、筛选前六个原始数据集的路径

项目根目录为：

```text
/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/EgoFound3R-baselines
```

六数据集正式 60 帧评测清单为：

```text
formal_evaluation/datasets/manifests/evaluation_test_windows_60f_strict_20260819T023430Z_d9d2963f.jsonl
```

筛选前读取的原始数据根目录如下。它们来自本地冻结的注册记录；5001 当前断联，因此本次没有重新做远端可读性确认。

| 数据集 | 原始数据根目录 |
| --- | --- |
| H2O | `/mnt/workspace/sjc/DATA/H2O/h2o_data` |
| HOT3D | `/mnt/workspace/sjc/DATA/HOT3D/hot3d/hot3d/dataset` |
| ARCTIC | `/mnt/workspace/sjc/DATA/EgoForce/ARCTIC` |
| OakInk-v2 | `/mnt/workspace/sjc/DATA/OakInk-v2` |
| TACO | `/mnt/cpfs/sjc/DATA/TACO_resized` |
| HOI4D | `/mnt/workspace/sjc/DATA/mnt-1/HOI4D` |

筛选实际依赖的不只是 RGB 原始数据，还包括 Ego stride5 完整预测和 GT cache。最终 104 段的每个子窗口都在下列 hydrated manifest 中固化了 `window_id`、`frame_ids`、Ego prediction directory、GT NPZ 和 P95 provenance：

```text
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest_hydrated.jsonl
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest_hydrated.jsonl.gz
```

六数据集后续方法结果、GT、RGB input index 和相机来源的精确注册路径集中记录在：

```text
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/source_alignment_auxmethods_full104_5001.json
```

该 alignment 文件还记录了 CPFS 释放后使用的 OSS alias target、HaWoR 六数据集 `predictions.jsonl` 路径、PAD-Hand 修复结果、EgoForce audit run、Dyn-HaMR 稀疏索引以及相机外参标注。新节点上不得按相似目录名猜路径，必须先以注册任务和该文件做身份及可读性核验。

## 二、数据筛选脚本与逻辑

### 脚本和产物

```text
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/build_selection.py
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/endpoint_audit.jsonl
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/eligible_manifest.jsonl
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest.jsonl
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest_hydrated.jsonl
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/summary.json
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/audit_provenance.json
```

`endpoint_audit.jsonl` 是 602 个 10 秒候选段的冻结审计输入。每段由连续的 5 个 60 帧窗口组成，共 300 帧；候选之间可以重叠，例如同一序列可以同时出现 0–10 秒和 2–12 秒。

对每个候选段，在第一帧和最后一帧分别检查左手、右手，四项条件必须全部成立：

1. Ego `hand_valid=true`；
2. GT `hand_valid=true`；
3. Ego 的 195 markers 全部为有限值；
4. GT 的 195 markers 全部为有限值。

中间帧不作为淘汰条件，因为中间缺失可以在前后有效锚点之间插值。首帧或末帧缺手时直接淘汰，禁止单边外推。筛选读取完整 Ego/GT 数组，不使用 Joint8 P95 的逐帧 mask；P95 只保留为指标及候选来源的 provenance。

602 段涉及 978 个唯一窗口，metadata 身份、`frame_ids`、60 帧形状、`hand_valid [60,2]` 和 markers `[60,2,195,3]` 都通过审计。319 段通过端点门槛。之后对每个数据集、每条 sequence 独立执行结束时间优先的贪心区间调度：按结束帧升序遍历，只接受 `start > last_end` 的候选。该算法给出最大数量的互不重叠区间集合，最终得到 104 段。

| 数据集 | 端点门槛后候选 | 最终互不重叠段 |
| --- | ---: | ---: |
| ARCTIC | 166 | 48 |
| H2O | 6 | 5 |
| HOT3D | 137 | 44 |
| OakInk-v2 | 10 | 7 |
| TACO | 0 | 0 |
| HOI4D | 0 | 0 |
| 合计 | 319 | 104 |

104 段中 62 段来自旧 177 清单，42 段来自原先因重叠未被选中的候选。

筛选后的 ID 路径为：

```text
# 104 段及 segment_id/gallery_stem
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest.jsonl

# 104 段、520 个子窗口的完整 window_id 和输入引用
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest_hydrated.jsonl

# DSW 部署使用的压缩冻结版本
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest_hydrated.jsonl.gz
```

未压缩 hydrated manifest 的 SHA-256 必须为：

```text
e88ac84845a5a61e6e472b12d160816ef0d8a48eaeb0e2085985335ff33c92a5
```

## 三、可视化脚本与作图逻辑

### 脚本路径

```text
# 注册、启动、inspect、verify、export
formal_evaluation/taskctl.py

# 远端预检、逐段输入 staging、16 路 CPU 调度、完成门槛
formal_evaluation/render_10s_endpoint_gallery_worker.py

# v5 作图实现；文件名保留 pilot 只是历史原因，完整 104 段也使用它
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/render_aux_pilot.py

# Ego 中间缺帧插值
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/ego_bracketed_fill.py

# MP4 编码与视频 I/O
visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/video_io.py

# 本地轻量同步及 SHA-256 校验
formal_evaluation/export_10s_gallery.py
```

### 执行逻辑

1. `taskctl.py` 读取冻结 manifest 和 source alignment，校验 104 段、manifest digest、Ego 插值来源 commit，并以唯一 run/output root 注册任务。
2. worker 先核对 520 个窗口的 Ego、GT、WiLoR、HaWoR、ReViV4D、PAD-Hand、EgoForce 和可选 Dyn-HaMR 输入身份与数组形状；PAD-Hand 必须来自修复后的双手结果，HaWoR 必须没有 identity fallback。
3. 每个 10 秒段先拼接 5 个 60 帧窗口。Ego 只对拼接后的内部缺帧做相机坐标 smoothstep 插值，保持观测帧不变，不做首尾外推。
4. 方法进入共同世界坐标时按标签区分：Ego 原始行、HaWoR、Dyn-HaMR 使用各自预测/native camera，再以每个 60 帧窗口第一帧 GT 相机做刚性锚定；Ego + GT ext、WiLoR、ReViV4D、PAD-Hand、EgoForce 和 GT 使用逐帧 GT 外参。
5. PNG 总览显示 5 个等间隔 3D 时刻和 3 张 RGB（0、149、299）。MP4 显示完整 300 帧，30 FPS、10 秒。六视角为 Front、Oblique left、Top、Oblique right、Side、Bottom。
6. 每个方法/视角单独保存 2048×2048 panel。总览使用较低分辨率拼接。视频视角采用按方法跟踪的平滑中心：15 帧跟踪、31 帧 median 后接 15 帧 mean，half-span 限制为 0.13–0.24 m。
7. Dyn-HaMR 只在段内至少一个 2 秒窗口有注册预测时加入该行。当前为 24/104 段、27/520 窗口；因此 panel 总数为 `80×48 + 24×54 = 5,136`。
8. ReViV4D 的注册结果只有 21 joints，因此画骨架；PAD-Hand 与 EgoForce 使用 778 顶点 mesh。

### 当前正式结果和后续操作

正式完成 run：

```text
visualization-four-10s-gallery-endpoint104-5001-auxmethods-hawor-native-handzoom-panels-full104_auxmethods_handzoom_panels_v5_5001_r1-20260916
```

误继承 pilot `--segment` 的旧 run 以 `full104_auxmethods_handzoom_panels_v5_5001-20260916` 结尾，已弃用并暂停，禁止恢复。后续修改必须使用新的 suffix、run ID 和 output root，不能覆盖当前正式结果。

5001 目前断联，不要重复探测。本轮未访问远端。连接恢复后先更新 current instance profile，做 `taskctl resources` 握手，再用精确 run ID 执行 `inspect`；不能仅凭 PID、部分文件或目录时间判断完成。
