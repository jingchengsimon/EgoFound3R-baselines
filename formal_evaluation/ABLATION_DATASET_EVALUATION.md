# 消融单数据集评测入口

复用 `run_egofound3r_stride_evaluation.py`。每个已登记任务的 spec 只保留一个数据集，`TASKCTL_GPU` 由 taskctl 分配一张物理卡；每个消融最多六个同时占卡的组合，smoke 也计入占卡数。

消融 spec 需要指定：

- `methods_config`：独立 JSON 文件，保留适配器内部的 `methods.egofound3r` 键，但填入该消融实际 `source_commit`、`inference_commit`、`checkpoint_sha256` 等来源字段。不修改固定 14-method 配置。任务 ID、method-set 和输出目录均独立区分消融与数据集。
- `required_smoke_strides: [5]`：只评测 stride5 的显式 smoke 门槛。未指定时保留原正式评测 `[1, 5]` 门槛。指定时必须包含请求评测的 stride。
- `smoke_roots`：对应消融的已验证 smoke，不能借用正式模型回执。来源、stride/phase、完成状态必须一致。
- `datasets`：单个数据集，沿用正式模型固定的完整窗口清单及 GT；窗口数分别为 283/400/434/400/400/461。

入口在推理前检查 methods 配置和 spec 的模型来源一致性；现有逐窗口校验、指标计算及 COMPLETE 流程保持不变。模型级六数据集汇总另做 CPU 步骤。

本改动仅完成入口配置支持，并不证明消融运行时已就绪。四个训练分支仍需各自核验 BF16 加载兼容性、最终 step4999 checkpoint、训练完成证据和真实 GPU smoke，通过后才能登记并启动正式评测。禁止为了兼容而替换消融结构、跳过 checkpoint contract 或借用正式模型身份。

本地入口测试（不加载真实模型、不占 GPU）：

```sh
python -m unittest formal_evaluation.tests.test_egofound3r_single_dataset
```
