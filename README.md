# 并行策略对比实验

## 一、作业内容
- 4 种并行策略对比（DP / PP / TP / 3D）+ Mermaid 图
- ZeRO 三阶段内存优化原理详解
- PyTorch 数据并行最小示例

## 二、环境
- Python 3.13
- PyTorch（CPU 也能跑）

## 三、运行
```bash
python parallelism_demo.py
```

## 四、文件说明
| 文件 | 说明 |
|------|------|
| parallel_strategy_comparison.md | 4 种并行对比 + Mermaid 图 |
| zero_explained.md | ZeRO 三阶段详解 |
| parallelism_demo.py | 数据并行最小示例 |
| .env.example | 环境变量模板 |

## 五、AI 协作记录
- 工具：Claude
- AI 生成：对比表格结构、ZeRO 公式推导、demo 代码骨架
- 我修改：表格内容核对、代码注释、README 组织
- prompt 示例：
  - "对比数据/流水线/张量/3D 并行，用 mermaid 画图 + 表格"
  - "详细解释 ZeRO-1/2/3 的显存优化原理和公式"
  - "写一个 PyTorch 数据并行最小示例，CPU 也能跑"

## 六、反思
（自己写 3-5 句）