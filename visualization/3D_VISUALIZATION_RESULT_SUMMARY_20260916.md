# 3D 10s 可视化结果总结

## 项目与结果路径

- 本地项目：`/Users/jingchengshi/Desktop/MIMO-Rutgers/1-Codes/EgoFound3R-baselines`
- 冻结代码与清单：`visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914`
- 筛选后完整 ID：`visualization/batch_10s_104_endpoint_renderable_p95_wmpjpe_20260914/selected_manifest_hydrated.jsonl`
- 正式 run ID：`visualization-four-10s-gallery-endpoint104-5001-auxmethods-hawor-native-handzoom-panels-full104_auxmethods_handzoom_panels_v5_5001_r1-20260916`
- 远端正式结果：`/mnt/workspace/sjc/DATA/eval_artifacts/gallery_10s_endpoint104_hawor_native_unmasked_20260915_full104_auxmethods_handzoom_panels_v5_5001_r1`
- 本地轻量副本：`visualization/gallery_10s_endpoint104_hawor_native_unmasked_20260915_full104_auxmethods_handzoom_panels_v5_5001_r1`

5001 当前已断开，本次没有重新访问节点。下面的完成状态来自断联前已经冻结并校验的结果：`COMPLETE` 存在，`summary.json` 为 `status=complete`，104/104 段成功、失败数为 0。总览包含 104 张 PNG 和 104 个 H.264 MP4；每个视频为 300 帧、30 FPS、10 秒。远端另有 5,136 张逐方法逐视角 2048×2048 panel PNG。本地轻量同步只保留 104 张总览 PNG、104 个 MP4、完成元数据及 SHA-256 传输清单，不包含 panel gallery、runtime、logs 和 control。

## 筛选结果

筛选从 602 个允许重叠的 10 秒候选段开始，涉及 978 个唯一 2 秒窗口。端点可渲染门槛通过 319 段，再按序列执行最大数量互不重叠选择，得到 104 段、共 520 个 60 帧子窗口。

| 数据集 | 端点门槛后候选 | 最终互不重叠段 |
| --- | ---: | ---: |
| ARCTIC | 166 | 48 |
| H2O | 6 | 5 |
| HOT3D | 137 | 44 |
| OakInk-v2 | 10 | 7 |
| TACO | 0 | 0 |
| HOI4D | 0 | 0 |
| 合计 | 319 | 104 |

最终 hydrated manifest 的未压缩 SHA-256 为 `e88ac84845a5a61e6e472b12d160816ef0d8a48eaeb0e2085985335ff33c92a5`。104 段中 62 段来自旧 177 清单，42 段来自原先因重叠未被选中的候选。

## 可视化内容

总览 PNG 等间隔展示 5 个 3D 时刻，并在左侧显示索引 0、149、299 的 3 张 RGB。MP4 使用完整 300 帧。六个视角是 Front、Oblique left、Top、Oblique right、Side、Bottom。

方法行为 Ego stride5、Ego stride5 + GT 外参、WiLoR、HaWoR native camera、ReViV4D joints、PAD-Hand 778、EgoForce 778、条件 Dyn-HaMR、GT。Dyn-HaMR 只在有注册预测的段显示：104 段中有 24 段、520 个子窗口中有 27 个窗口重合；另外 80 段不画 Dyn-HaMR 行。

HaWoR 直接使用 `hand_*_world` 和 `camera_c2w`，再以每个 60 帧窗口的第一帧相机位姿与 GT 对齐到共同 10 秒世界坐标，不用逐帧 GT 相机替换，也没有 identity fallback。图中必须保留标签 `HaWoR native camera (unmasked SLAM)`。

Ego 只对 10 秒拼接后、两侧均有有效锚点的中间缺帧做相机坐标系 smoothstep 插值；保留所有观测帧，禁止首尾单边外推。对应来源 commit 为 `8fc061a615895bd3b5a556f7387bae306e32d9db`，分支 `codex/hand-depth-scale-fusion-20260908`。
