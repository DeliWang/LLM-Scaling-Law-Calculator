/**
 * LLM 缩放律计算核心算法（论文 arXiv:2508.06617）
 * 
 * 实现：
 * - 稠密：Hoffmann Eq.(2) + Table II
 * - 稀疏：Equation 7 + Table VII / VIII / IX
 * - 算力 C=6ND、训练时间 T 与所需 TFLOPS 换算
 */

// ---------------------------------------------------------------------------
// 系数（论文 Table II, VII, VIII, IX）
// ---------------------------------------------------------------------------

export interface Coefficients {
  e: number;
  a: number;
  b: number;
  c: number;  // 仅 Eq.7 使用，稠密时为 0
  alpha: number;
  beta: number;
  gamma: number;  // 仅 Eq.7 使用，稠密时可为 0.01
}

// 稠密 — Hoffmann，论文 Table II（与 Eq.2 一致）
export const COEFFS_DENSE: Coefficients = {
  e: 1.69, a: 406.4, b: 410.7, c: 0.0, alpha: 0.34, beta: 0.28, gamma: 0.01
};

// 稀疏 — 论文 Table VII（S=0 时与 Hoffmann 一致，c=0）
export const COEFFS_DENSE_EQ7: Coefficients = {
  e: 1.69, a: 406.4, b: 410.7, c: 0.0, alpha: 0.34, beta: 0.28, gamma: 0.01
};

// 稀疏 — 剪枝类 Frantar，论文 Table VIII
export const COEFFS_FRANTAR: Coefficients = {
  e: 0.15, a: 86.03, b: 7.90, c: 29.26, alpha: 0.28, beta: 0.08, gamma: 2.00
};

// 稀疏 — MoE，论文 Table IX
export const COEFFS_MOE: Coefficients = {
  e: 0.57, a: 8.26, b: 6324.82, c: 3.57, alpha: 0.08, beta: 0.40, gamma: 1.19
};

// 模型类型
export type ModelType = 'dense' | 'sparse_pruned' | 'sparse_moe';

// 获取系数
export function getCoefficients(modelType: ModelType, S: number = 0): Coefficients {
  if (modelType === 'dense' || (modelType.startsWith('sparse') && S === 0)) {
    return COEFFS_DENSE;
  }
  if (modelType === 'sparse_pruned') {
    return COEFFS_FRANTAR;
  }
  if (modelType === 'sparse_moe') {
    return COEFFS_MOE;
  }
  throw new Error(`未知 model_type: ${modelType}`);
}

// ---------------------------------------------------------------------------
// 损失与下界
// ---------------------------------------------------------------------------

/** 稠密损失 L(N,D) = e + a/N^α + b/D^β */
export function lossDense(N: number, D: number, coeffs: Coefficients = COEFFS_DENSE): number {
  return coeffs.e + coeffs.a / Math.pow(N, coeffs.alpha) + coeffs.b / Math.pow(D, coeffs.beta);
}

/** 稀疏损失 Eq.7: L = e(1-S)^γ + (a(1-S)^α + c·S)/N^α + b/D^β */
export function lossSparse(N: number, D: number, S: number, coeffs: Coefficients): number {
  const term1 = coeffs.e * Math.pow(1 - S, coeffs.gamma);
  const term2 = (coeffs.a * Math.pow(1 - S, coeffs.alpha) + coeffs.c * S) / Math.pow(N, coeffs.alpha);
  const term3 = coeffs.b / Math.pow(D, coeffs.beta);
  return term1 + term2 + term3;
}

/** 稠密时损失下界（D→∞）：e + a/N^α */
export function LMinDense(N: number, coeffs: Coefficients = COEFFS_DENSE): number {
  return coeffs.e + coeffs.a / Math.pow(N, coeffs.alpha);
}

/** 稀疏时损失下界（D→∞）：e(1-S)^γ + (a(1-S)^α + c·S)/N^α */
export function LMinSparse(N: number, S: number, coeffs: Coefficients): number {
  const term1 = coeffs.e * Math.pow(1 - S, coeffs.gamma);
  const term2 = (coeffs.a * Math.pow(1 - S, coeffs.alpha) + coeffs.c * S) / Math.pow(N, coeffs.alpha);
  return term1 + term2;
}

// ---------------------------------------------------------------------------
// 反解：由目标损失 L 求所需 token 数 D
// ---------------------------------------------------------------------------

/** 稠密：已知 N、L，求 D。若 L <= L_min 则抛出错误 */
export function tokensFromLossDense(N: number, LTarget: number, coeffs: Coefficients = COEFFS_DENSE): number {
  const LMin = LMinDense(N, coeffs);
  if (LTarget <= LMin) {
    throw new Error(`目标损失 L=${LTarget} 过低，至少需 L > L_min ≈ ${LMin.toFixed(4)}（N=${N.toExponential(2)}）`);
  }
  const dataTerm = LTarget - coeffs.e - coeffs.a / Math.pow(N, coeffs.alpha);
  if (dataTerm <= 0) {
    throw new Error(`目标损失 L=${LTarget} 过低，至少需 L > ${LMin.toFixed(4)}`);
  }
  return Math.pow(coeffs.b / dataTerm, 1.0 / coeffs.beta);
}

/** 稀疏：已知 N、S、L，求 D */
export function tokensFromLossSparse(N: number, S: number, LTarget: number, coeffs: Coefficients): number {
  const LMin = LMinSparse(N, S, coeffs);
  if (LTarget <= LMin) {
    throw new Error(`目标损失 L=${LTarget} 过低，至少需 L > L_min ≈ ${LMin.toFixed(4)}（N=${N.toExponential(2)}, S=${S}）`);
  }
  const term1 = coeffs.e * Math.pow(1 - S, coeffs.gamma);
  const term2 = (coeffs.a * Math.pow(1 - S, coeffs.alpha) + coeffs.c * S) / Math.pow(N, coeffs.alpha);
  const dataTerm = LTarget - term1 - term2;
  if (dataTerm <= 0) {
    throw new Error(`目标损失 L=${LTarget} 过低，至少需 L > ${LMin.toFixed(4)}`);
  }
  return Math.pow(coeffs.b / dataTerm, 1.0 / coeffs.beta);
}

// ---------------------------------------------------------------------------
// 算力与训练时间
// ---------------------------------------------------------------------------

const SECONDS_PER_DAY = 86400;
const FLOP_PER_TFLOP = 1e12;

/** 训练总算力 FLOPs：C = 6 * N * D。N 为活跃参数量 */
export function computeFlops(N: number, D: number): number {
  return 6.0 * N * D;
}

/** 训练天数 T = C / (TFLOPS × 1e12 × 86400 × utilization) */
export function trainingDays(CFlops: number, tflops: number, utilization: number = 0.35): number {
  const flopsPerDay = tflops * FLOP_PER_TFLOP * SECONDS_PER_DAY * utilization;
  return CFlops / flopsPerDay;
}

/** 给定期望训练天数 T（天），反推所需 TFLOPS */
export function requiredTflops(CFlops: number, TDays: number, utilization: number = 0.35): number {
  if (TDays <= 0) {
    throw new Error('训练天数 T 须为正数');
  }
  return CFlops / (TDays * SECONDS_PER_DAY * FLOP_PER_TFLOP * utilization);
}

// ---------------------------------------------------------------------------
// 统一入口：按规划文档的输入组合计算
// ---------------------------------------------------------------------------

export interface ComputeInput {
  N: number;                    // 活跃参数量
  modelType: ModelType;         // 模型类型
  S?: number;                   // 稀疏度（稀疏时必填）
  LTarget?: number;             // 目标损失（与 D 二选一）
  DTokens?: number;             // 训练 token 数（与 L 二选一）
  gpuTflops?: number;           // GPU 算力（与 T 二选一）
  gpuUtilization?: number;      // GPU 利用率
  TDays?: number;               // 期望训练天数（与 GPU 二选一）
  defaultUtilization?: number;  // 默认利用率
}

export interface ComputeResult {
  N: number;                    // 活跃参数量
  modelType: ModelType;         // 模型类型
  S: number;                    // 稀疏度
  L: number;                    // 损失
  D: number;                    // 训练 token 数
  C: number;                    // 总算力 FLOPs
  TDays?: number;               // 训练天数
  requiredTflops?: number;      // 所需 TFLOPS
  totalParams?: number;         // 总参数量（稀疏时）
  LMin: number;                 // 损失下界
}

export function compute(input: ComputeInput): ComputeResult {
  const {
    N,
    modelType,
    S = 0,
    LTarget,
    DTokens,
    gpuTflops,
    gpuUtilization,
    TDays,
    defaultUtilization = 0.35
  } = input;

  // 验证输入
  if (N <= 0) throw new Error('N（活跃参数量）须为正');
  if (!['dense', 'sparse_pruned', 'sparse_moe'].includes(modelType)) {
    throw new Error('model_type 应为 dense | sparse_pruned | sparse_moe');
  }
  if (S < 0 || S >= 1) throw new Error('稀疏度 S 应满足 0 <= S < 1');
  if ((LTarget === undefined) === (DTokens === undefined)) {
    throw new Error('须且仅须指定 L_target 或 D_tokens 之一');
  }
  if (gpuTflops !== undefined && TDays !== undefined) {
    throw new Error('gpu_tflops 与 T_days 二选一');
  }

  const coeffs = getCoefficients(modelType, S);
  const isDense = modelType === 'dense' || S === 0;
  const utilization = gpuUtilization ?? defaultUtilization;

  // 计算 L 和 D
  let L: number;
  let D: number;

  if (LTarget !== undefined) {
    // 由 L 求 D
    L = LTarget;
    D = isDense
      ? tokensFromLossDense(N, LTarget, coeffs)
      : tokensFromLossSparse(N, S, LTarget, coeffs);
  } else {
    // 由 D 求 L
    D = DTokens!;
    L = isDense
      ? lossDense(N, D, coeffs)
      : lossSparse(N, D, S, coeffs);
  }

  // 计算算力
  const C = computeFlops(N, D);

  // 计算损失下界
  const LMin = isDense ? LMinDense(N, coeffs) : LMinSparse(N, S, coeffs);

  // 结果
  const result: ComputeResult = {
    N,
    modelType,
    S,
    L,
    D,
    C,
    LMin
  };

  // 总参数量（稀疏时）
  if (S > 0) {
    result.totalParams = N / (1 - S);
  }

  // 训练时间或所需 TFLOPS
  if (gpuTflops !== undefined) {
    result.TDays = trainingDays(C, gpuTflops, utilization);
  } else if (TDays !== undefined) {
    result.requiredTflops = requiredTflops(C, TDays, utilization);
  }

  return result;
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

/** 格式化参数量（如 7B, 70B, 175B） */
export function formatParams(params: number): string {
  if (params >= 1e12) return `${(params / 1e12).toFixed(1)}T`;
  if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`;
  if (params >= 1e3) return `${(params / 1e3).toFixed(1)}K`;
  return params.toFixed(0);
}

/** 格式化 FLOPs */
export function formatFlops(flops: number): string {
  if (flops >= 1e24) return `${(flops / 1e24).toFixed(2)} YFLOPs`;
  if (flops >= 1e21) return `${(flops / 1e21).toFixed(2)} ZFLOPs`;
  if (flops >= 1e18) return `${(flops / 1e18).toFixed(2)} EFLOPs`;
  if (flops >= 1e15) return `${(flops / 1e15).toFixed(2)} PFLOPs`;
  if (flops >= 1e12) return `${(flops / 1e12).toFixed(2)} TFLOPs`;
  return `${(flops / 1e9).toFixed(2)} GFLOPs`;
}

/** 格式化天数 */
export function formatDays(days: number): string {
  if (days >= 365) return `${(days / 365).toFixed(1)} 年`;
  if (days >= 30) return `${(days / 30).toFixed(1)} 月`;
  if (days >= 1) return `${days.toFixed(1)} 天`;
  if (days >= 1/24) return `${(days * 24).toFixed(1)} 小时`;
  return `${(days * 24 * 60).toFixed(0)} 分钟`;
}

/** 格式化 token 数 */
export function formatTokens(tokens: number): string {
  if (tokens >= 1e15) return `${(tokens / 1e15).toFixed(2)}P`;
  if (tokens >= 1e12) return `${(tokens / 1e12).toFixed(2)}T`;
  if (tokens >= 1e9) return `${(tokens / 1e9).toFixed(2)}B`;
  if (tokens >= 1e6) return `${(tokens / 1e6).toFixed(2)}M`;
  return tokens.toFixed(0);
}
