# 消融单数据集评测入口

复用 `run_egofound3r_stride_evaluation.py`。每个已登记任务的 spec 只保留一个数据集，`TASKCTL_GPU` 由 taskctl 分配一张物理卡；每个消融最多六个同时占卡的组合，smoke 也计入占卡数。

消融 spec 需要指定：

- `methods_config`：独立 JSON 文件，保留适配器内部的 `methods.egofound3r` 键，但填入该消融实际 `source_commit`、`inference_commit`、`checkpoint_sha256` 等来源字段。不修改固定 14-method 配置。任务 ID、method-set 和输出目录均独立区分消融与数据集。
- 在该独立 method 配置设置 `runtime_mode: "ablation_bf16"`：按 ZeRO-2 训练路径将模型转为 BF16，再调用各消融自身的严格 checkpoint 加载器，仍检查结构、shape 和 dtype。MANO 自身的 `_apply` 会保留冻结几何层的 FP32。其他评测默认仍用 `checkpoint_native`，不自动回退。
- `required_smoke_strides: [5]`：只评测 stride5 的显式 smoke 门槛。未指定时保留原正式评测 `[1, 5]` 门槛。指定时必须包含请求评测的 stride。
- `smoke_roots`：对应消融的已验证 smoke，不能借用正式模型回执。来源、stride/phase、完成状态必须一致。
- `require_metric_smoke: true`：使用 `run_ablation_dataset_smoke.py --spec <单数据集spec> --output-root <独立smoke目录>`。它先复用完整输入核验与单窗推理，再复用正式指标入口验证同一 GT 窗口；覆盖完整后才写顶层完成标记。正式入口同时核对该数据集的单窗指标报告。未定义指标仍保留 NaN。
- `datasets`：单个数据集，沿用正式模型固定的完整窗口清单及 GT；窗口数分别为 283/400/434/400/400/461。

入口在推理前检查 methods 配置和 spec 的模型来源一致性；现有逐窗口校验、指标计算及 COMPLETE 流程保持不变。模型级六数据集汇总另做 CPU 步骤。

0.1B 必须使用其训练快照 `ablation_hand_0p1b_dynamic_20step_speedtest_4gpu_20260904.toml`（宽度 516），0.05B 使用 `ablation_hand_0p05b_dynamic_5k_3gpu_20260904.toml`（宽度 360）。这些历史文件名不代表最终训练步数或实际 GPU 数；以该运行的 checkpoint/日志为准。两者都不能用未覆盖宽度的公共 `seven_dataset_dynamic_multirate_adaptation_1600_8gpu_zero2_20260903.toml` 替代，否则会构造默认宽度 768。

本改动并不证明消融运行时已就绪。四个训练分支仍需各自核验最终 step4999 checkpoint、训练完成证据和真实 GPU smoke，通过后才能启动正式评测。禁止为了兼容而替换消融结构、跳过 checkpoint contract 或借用正式模型身份。

本地入口测试（不加载真实模型、不占 GPU）：

```sh
python -m unittest formal_evaluation.tests.test_egofound3r_single_dataset
```

`config/ablation_step4999_hand_0p1b_dsw.json` 和 `config/ablation_step4999_hand_0p05b_dsw.json` 各自同时提供独立 methods 配置和 strict runtime 路径配置；不改变固定 14-method registry。checkpoint 摘要来自已登记的 CPU 核验，运行时路径仍须在实际提交节点逐项通过 strict 检查。

- MANO续训最终checkpoint混存FP32/BF16，使用 `ablation_checkpoint_dtypes_bf16`：只通过mmap读取保存dtype，逐张量对齐后调用原有严格加载器，再转BF16推理；MANO `_apply` 保留冻结几何FP32。旧BF16 checkpoint仍用 `ablation_bf16`，不自动回退或跳过contract/shape校验。
