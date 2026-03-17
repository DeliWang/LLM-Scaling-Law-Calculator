# LLM Scaling Law Calculator

<p align="center">
  <strong>基于论文 arXiv:2508.06617 的大模型训练缩放律计算器</strong>
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#技术栈">技术栈</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#核心算法">核心算法</a> •
  <a href="#api-文档">API 文档</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/React-19-blue?style=flat-square&logo=react" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-5-blue?style=flat-square&logo=typescript" alt="TypeScript">
  <img src="https://img.shields.io/badge/Tailwind-4-38B2AC?style=flat-square&logo=tailwind-css" alt="Tailwind">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## 简介

LLM Scaling Law Calculator 是一个交互式的 Web 应用，帮助研究人员和工程师快速估算大语言模型训练所需的计算资源、数据量和训练时间。

在线体验：[https://aitool.coze.site/](https://aitool.coze.site/)

基于论文 [arXiv:2508.06617](https://arxiv.org/abs/2508.06617) 提出的缩放律公式，本工具支持：

- **稠密模型 (Dense Model)** 计算
- **稀疏模型 (Sparse/MoE Model)** 计算
- **Chinchilla 最优训练** 数据量推荐
- **657+ GPU 规格** 对比与选择

## 功能特性

### 🧮 智能计算

- **训练损失预测**：根据模型参数量、数据量预测训练损失
- **算力需求估算**：计算所需 FLOPs 和 GPU 数量
- **训练时间预测**：基于 GPU 算力估算训练时长
- **损失下界计算**：预测模型最优损失

### 📊 可视化展示

- **损失曲线图**：直观展示损失随数据量变化趋势
- **GPU 对比表格**：支持排序、筛选、分页
- **实时结果更新**：参数调整即时响应

### 🎛️ 灵活配置

- **模型类型**：支持 Dense、Frantar、MoE 三种模型
- **稀疏度调节**：0-99% 稀疏度滑块
- **数据量选择**：自定义或 Chinchilla 最优推荐
- **GPU 选择**：支持多卡配置和利用率调整

### 🌓 用户体验

- **深色/浅色主题**：根据时间自动切换
- **响应式设计**：适配桌面和移动设备
- **高性能渲染**：基于 Recharts 的流畅图表

## 技术栈

| 类别 | 技术 |
|------|------|
| **框架** | Next.js 16 (App Router) |
| **前端** | React 19, TypeScript 5 |
| **样式** | Tailwind CSS 4, shadcn/ui |
| **图表** | Recharts |
| **数据库** | SQLite (better-sqlite3) |
| **图标** | Lucide React |

## 快速开始

### 环境要求

- Node.js 18+
- pnpm 9+

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/llm-scaling-calculator.git
cd llm-scaling-calculator

# 安装依赖
pnpm install
```

### 开发

```bash
# 启动开发服务器
pnpm dev
```

访问 [http://localhost:5000](http://localhost:5000) 查看应用。

### 构建

```bash
# 构建生产版本
pnpm build

# 启动生产服务器
pnpm start
```

## 项目结构

```
src/
├── app/                          # Next.js App Router
│   ├── api/                      # API 路由
│   │   ├── gpus/route.ts         # GPU 数据接口
│   │   └── stats/route.ts        # 统计数据接口
│   ├── layout.tsx                # 根布局
│   ├── page.tsx                  # 首页计算器
│   └── globals.css               # 全局样式
├── components/                   # React 组件
│   ├── ui/                       # shadcn/ui 组件
│   └── theme-toggle.tsx          # 主题切换
├── lib/                          # 核心库
│   ├── scaling-law.ts            # 缩放律计算公式
│   ├── gpu-data.ts               # GPU 类型定义
│   └── db/                       # 数据库操作
│       ├── sqlite.ts             # SQLite 连接
│       ├── sqlite-gpus.ts        # GPU 数据操作
│       └── sqlite-stats.ts       # 统计数据操作
├── hooks/                        # 自定义 Hooks
│   └── use-gpus.ts               # GPU 数据加载
└── public/                       # 静态资源
    └── coze-logo.svg             # Logo
```

## 核心算法

### 缩放律公式

训练损失 L(N, D) 由以下公式计算：

```
L(N, D) = L_min(N) + C / D^β
```

其中：
- `N`：模型参数量
- `D`：训练数据量（tokens）
- `L_min(N)`：损失下界，随模型规模递减
- `C, β`：拟合系数，因模型类型而异

### 稀疏模型

对于稀疏模型（如 MoE），引入稀疏度 S：

```
L_sparse(N, S, D) = L_min_sparse(N, S) + C_sparse(S) / D^β_sparse(S)
```

系数通过论文提供的拟合参数计算。

### Chinchilla 最优训练

推荐数据量计算：

```
D_opt ≈ 20 × N
```

### 算力与时间

```
FLOPs = 6 × N × D                    # 训练算力
Time = FLOPs / (TFLOPS × Util × GPU) # 训练时间
```

## API 文档

### GET /api/gpus

获取 GPU 列表。

**参数：**
- `isActive` (boolean)：仅返回活跃 GPU

**响应：**
```json
{
  "success": true,
  "data": [
    {
      "id": "nvidia_h100_sxm",
      "name": "NVIDIA H100 SXM",
      "tfops": 1979,
      "utilization": 0.5,
      "memory_gb": 80,
      "price_usd": 30000
    }
  ]
}
```

### GET /api/stats

获取使用统计。

**参数：**
- `period` (string)：统计周期 (day/week/month/all)

**响应：**
```json
{
  "total_calculations": 100,
  "total_page_views": 500,
  "avg_model_params": 7000000000,
  "total_training_hours": 1000
}
```

### POST /api/stats

记录使用统计。

**请求体：**
```json
{
  "action": "page_view | calculation | gpu_selection",
  "params_n": 7000000000,
  "params_model_type": "dense",
  "params_sparsity": 0
}
```

## 🚀 Roadmap

| 状态 | 功能 | 描述 |
|:----:|------|------|
| 🔜 | **数据集推荐** | 根据模型规模和任务类型推荐合适的数据集 |
| 🔜 | **模型部署硬件成本估计** | 估算推理阶段的硬件需求和成本 |
| 🔜 | **自动更新硬件数据及价格** | 定期同步最新 GPU 规格和市场价格 |
| 🔜 | **Token 吞吐量和成本估计** | 估算推理吞吐量和每 token 成本 |

> 🔜 计划中 | 🚧 开发中 | ✅ 已完成

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

### 开发规范

- 使用 TypeScript 进行类型安全开发
- 遵循 ESLint 和 Prettier 配置
- 组件优先使用 shadcn/ui
- 提交信息遵循 Conventional Commits

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- 论文作者提供的缩放律公式和拟合参数
- [shadcn/ui](https://ui.shadcn.com) 提供的优秀 UI 组件
- [Recharts](https://recharts.org) 提供的图表库
- [GPUs-Specs](https://github.com/RonnyMuthomi/GPUs-Specs) 提供 GPU 规格数据

---

<p align="center">
  <a href="https://www.coze.cn" target="_blank">
    <img src="public/coze-logo.svg" alt="Coze" width="80">
  </a>
  <br>
  <sub>Powered by <a href="https://www.coze.cn">Coze</a></sub>
</p>
