# RQ1 55-case E2E：最新更新

按冻结的 55 条规则（Robot 25、IoT 30）完成一对一受控 E2E 汇总。远程模型 Qwen3-VL-8B，temperature=0.2。

- Contract 校验通过：**54/55**；未输出动作：**5/55**（含 fail-closed 与预测空域）。
- 已输出动作：**50**；其中独立 source-rule oracle 判安全 **47/50**。
- Exact candidate mask：**47/54**；false-admission cases **3**，共纳入 **4** 个不安全候选。
- 初次补跑发现字符串比较表达式被旧 validator 错误拒绝，已修复受限 parser 并重跑；剩余不正确 Contract/空域保留为模型语义错误，没有手工改写结果。

完整失败分析、实验口径及英文 Evaluation 草稿见 [完整报告](RQ1_TRUE_E2E_8B_55CASE_FULL_REPORT.md)；逐案数据见 [CSV](rq1_true_e2e_8b_55case_rows.csv) 和合并 [JSON](rq1_true_e2e_8b_55case_merged.json)。本实验不是仿真器 replay，也不测 task-goal success。
