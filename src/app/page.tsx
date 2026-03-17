'use client';

import { useState, useMemo, useEffect, useRef } from 'react';
import {
  compute,
  formatFlops,
  formatTokens,
  LMinDense,
  LMinSparse,
  COEFFS_DENSE,
  COEFFS_FRANTAR,
  COEFFS_MOE,
  getCoefficients,
  type ModelType,
} from '@/lib/scaling-law';
import { checkMemoryConstraint, getRequiredMemoryPerGpu, type GPU } from '@/lib/gpu-data';
import { useGPUs, useStats, recordStat } from '@/hooks/use-gpus';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/theme-toggle';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Calculator, Database, Cpu, TrendingDown, Plus, X, Search, Check, Zap, Clock, FileText, Github, AlertCircle } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

// 时间刻度值（以天为单位）
const TIME_TICKS = [
  1 / (24 * 60),    // 1分钟
  10 / (24 * 60),   // 10分钟
  1 / 24,           // 1小时
  10 / 24,          // 10小时
  1,                // 1天
  10,               // 10天
  30,               // 1月
  365,              // 1年
];

// 图表配置常量
const CHART_CONFIG = {
  LOSS_OFFSET_MAX: 2,      // 损失上界偏移量（相对于损失下界）
  LOSS_OFFSET_MIN: 0.01,   // 损失下界偏移量（相对于损失下界，确保略高于下界）
  DATA_POINTS: 20,         // 图表数据点数量
  MAX_TRAINING_DAYS: 365,  // 最大训练天数（1年）
} as const;

// Chinchilla 最优训练公式：数据量与参数量的最优比例
const CHINCHILLA_OPTIMAL_RATIO = 20; // D_opt ≈ 20 × N

// 格式化时间刻度
function formatTimeTick(days: number): string {
  if (days >= 365) return '1年';
  if (days >= 30) return '1月';
  if (days >= 10) return '10天';
  if (days >= 1) return '1天';
  if (days >= 10 / 24) return '10时';
  if (days >= 1 / 24) return '1时';
  if (days >= 10 / (24 * 60)) return '10分';
  return '1分';
}

// 格式化时间
function formatTime(days: number): string {
  if (days >= 365) return `${(days / 365).toFixed(1)}年`;
  if (days >= 30) return `${(days / 30).toFixed(1)}月`;
  if (days >= 1) return `${days.toFixed(1)}天`;
  if (days >= 1 / 24) return `${(days * 24).toFixed(1)}小时`;
  return `${(days * 24 * 60).toFixed(0)}分钟`;
}

// 生成图表数据点
function generateChartData(
  N: number,
  modelType: ModelType,
  S: number,
  gpuTflops: number,
  utilization: number,
  gpuCount: number
) {
  const data = [];
  const totalTflops = gpuTflops * gpuCount;

  // 计算损失下界
  let LMin: number;
  if (modelType === 'dense' || S === 0) {
    LMin = LMinDense(N, COEFFS_DENSE);
  } else {
    // 使用已定义的系数常量，避免重复
    const coeffs = modelType === 'sparse_pruned' ? COEFFS_FRANTAR : COEFFS_MOE;
    const term1 = coeffs.e * Math.pow(1 - S, coeffs.gamma);
    const term2 =
      (coeffs.a * Math.pow(1 - S, coeffs.alpha) + coeffs.c * S) / Math.pow(N, coeffs.alpha);
    LMin = term1 + term2;
  }

  const maxL = LMin + CHART_CONFIG.LOSS_OFFSET_MAX;
  const minL = LMin + CHART_CONFIG.LOSS_OFFSET_MIN;

  for (let i = 0; i <= CHART_CONFIG.DATA_POINTS; i++) {
    const L = minL + (maxL - minL) * (i / CHART_CONFIG.DATA_POINTS);
    try {
      const result = compute({
        N,
        modelType,
        S,
        LTarget: L,
        gpuTflops: totalTflops,
        gpuUtilization: utilization,
      });

      // 只保留训练时间不超过一年的数据点
      const tDays = result.TDays || 0;
      if (tDays <= CHART_CONFIG.MAX_TRAINING_DAYS) {
        data.push({
          D: result.D,
          DFormatted: formatTokens(result.D),
          L: result.L,
          TDays: tDays,
          C: result.C,
        });
      }
    } catch {
      // 跳过无效点
    }
  }

  // 反转数据，使数据量由小到大排列
  data.reverse();

  // 计算推荐数据量的精确点（Chinchilla最优：D_opt ≈ 20 × N）
  const recommendedD = N * CHINCHILLA_OPTIMAL_RATIO;
  let recommendedPoint = null;

  if (data.length > 0) {
    const maxD = data[data.length - 1].D;
    const minD = data[0].D;

    if (recommendedD >= maxD) {
      // 推荐值超出范围，使用最右侧的点
      recommendedPoint = data[data.length - 1];
    } else if (recommendedD <= minD) {
      // 推荐值低于范围，使用最左侧的点
      recommendedPoint = data[0];
    } else {
      // 推荐值在范围内，计算精确点并替换最近的数据点
      try {
        const result = compute({
          N,
          modelType,
          S,
          DTokens: recommendedD,
          gpuTflops: totalTflops,
          gpuUtilization: utilization,
        });
        recommendedPoint = {
          D: result.D,
          DFormatted: formatTokens(result.D),
          L: result.L,
          TDays: result.TDays || 0,
          C: result.C,
        };
        // 找到精确点应该插入的位置
        let insertIndex = data.findIndex(p => p.D > recommendedD);
        if (insertIndex === -1) insertIndex = data.length;
        
        // 找到最近的数据点并移除，然后插入精确点
        // 比较 insertIndex-1 和 insertIndex 位置的点哪个更近
        const prevIndex = insertIndex - 1;
        if (prevIndex >= 0 && insertIndex < data.length) {
          const prevDist = Math.abs(data[prevIndex].D - recommendedD);
          const nextDist = Math.abs(data[insertIndex].D - recommendedD);
          // 移除距离更近的点
          if (prevDist <= nextDist) {
            data.splice(prevIndex, 1, recommendedPoint);
          } else {
            data.splice(insertIndex, 1, recommendedPoint);
          }
        } else if (prevIndex >= 0) {
          // 只能和前一个点比较
          data.splice(prevIndex, 1, recommendedPoint);
        } else {
          // 在数组开头，直接替换第一个点
          data.splice(0, 1, recommendedPoint);
        }
      } catch {
        // 计算失败，使用最近的点
        recommendedPoint = data.reduce((closest, point) =>
          Math.abs(point.D - recommendedD) < Math.abs(closest.D - recommendedD) ? point : closest
        , data[0]);
      }
    }
  }

  return { data, LMin, recommendedPoint };
}

// 具体数据结果卡片（有数据量时）
function GPUSpecificResultCard({
  gpu,
  N,
  D,
  modelType,
  S,
  gpuCount,
  utilization,
}: {
  gpu: GPU;
  N: number;
  D: number;
  modelType: ModelType;
  S: number;
  gpuCount: number;
  utilization: number;
}) {
  const result = useMemo(() => {
    try {
      return compute({
        N,
        modelType,
        S,
        DTokens: D,
        gpuTflops: gpu.tfops * gpuCount,
        gpuUtilization: utilization,
      });
    } catch {
      return null;
    }
  }, [N, D, modelType, S, gpu.tfops, gpuCount, utilization]);

  if (!result) {
    return null;
  }

  // 格式化价格
  const formatPrice = (price: number | null) => {
    if (price === null) return '未知';
    return `$${price.toLocaleString()}`;
  };

  return (
    <div className="bg-muted/60 border border-border rounded-xl p-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* GPU信息 */}
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-chart-2/10 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-chart-2" />
          </div>
          <div>
            <h3 className="text-foreground font-medium">{gpu.name}</h3>
            <p className="text-xs text-muted-foreground">
              {gpu.tfops} TFLOPS · {gpu.memory_gb}GB · {gpuCount}张 · {formatPrice(gpu.price_usd)}/卡
            </p>
          </div>
        </div>

        {/* 计算结果 */}
        <div className="flex flex-wrap items-center gap-6">
          {/* 方案总价 */}
          <div className="text-center">
            <p className="text-xs text-muted-foreground mb-1">方案总价</p>
            <p className="text-lg font-bold text-foreground">
              {gpu.price_usd ? formatPrice(gpu.price_usd * gpuCount) : '无数据'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground mb-1">目标损失</p>
            <p className="text-xl font-bold text-chart-1">{result.L.toFixed(4)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground mb-1">总算力</p>
            <p className="text-xl font-bold text-chart-4">{formatFlops(result.C)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground mb-1">训练时间</p>
            <p className="text-xl font-bold text-chart-2">{formatTime(result.TDays || 0)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground mb-1">总算力</p>
            <p className="text-lg font-medium text-foreground">{(gpu.tfops * gpuCount).toFixed(0)} TFLOPS</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// 图表结果卡片（无数据量时）
function GPUChartCard({
  gpu,
  N,
  modelType,
  S,
  gpuCount,
  utilization,
}: {
  gpu: GPU;
  N: number;
  modelType: ModelType;
  S: number;
  gpuCount: number;
  utilization: number;
}) {
  const { data, LMin, recommendedPoint } = useMemo(() => {
    return generateChartData(N, modelType, S, gpu.tfops, utilization, gpuCount);
  }, [N, modelType, S, gpu.tfops, gpuCount, utilization]);

  // 格式化价格
  const formatPrice = (price: number | null) => {
    if (price === null) return '未知';
    return `$${price.toLocaleString()}`;
  };

  if (data.length === 0 || !recommendedPoint) {
    return null;
  }

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: Array<{ payload: { D: number; L: number; TDays: number; C: number } }>;
  }) => {
    if (active && payload && payload.length) {
      const p = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 text-xs">
          <p className="text-muted-foreground">
            数据量: <span className="text-foreground">{formatTokens(p.D)}</span>
          </p>
          <p className="text-muted-foreground">
            损失: <span className="text-chart-1">{p.L.toFixed(4)}</span>
          </p>
          <p className="text-muted-foreground">
            时间: <span className="text-chart-2">{formatTime(p.TDays)}</span>
          </p>
          <p className="text-muted-foreground">
            算力: <span className="text-muted-foreground">{formatFlops(p.C)}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-muted/60 border border-border rounded-xl p-4 md:p-6">
      <div className="flex flex-col lg:flex-row gap-6">
        {/* 左侧：GPU信息 */}
        <div className="lg:w-64 flex-shrink-0">
          <h3 className="text-foreground font-medium mb-1">{gpu.name}</h3>
          <p className="text-xs text-muted-foreground mb-3">
            {formatPrice(gpu.price_usd)}/卡
          </p>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">算力</span>
              <span className="text-muted-foreground">{gpu.tfops} TFLOPS</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">显存</span>
              <span className="text-muted-foreground">{gpu.memory_gb} GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">总算力</span>
              <span className="text-foreground font-medium">
                {(gpu.tfops * gpuCount).toFixed(0)} TFLOPS
              </span>
            </div>
            <div className="flex justify-between pt-2 border-t border-border">
              <span className="text-muted-foreground font-medium">方案总价</span>
              <span className="text-foreground font-bold">
                {gpu.price_usd ? `$${(gpu.price_usd * gpuCount).toLocaleString()}` : '无数据'}
              </span>
            </div>
          </div>
        </div>

        {/* 右侧：图表 */}
        <div className="flex-1 min-h-[280px]">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border/50" />
              <XAxis
                dataKey="DFormatted"
                interval="preserveStartEnd"
                tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={{ stroke: 'var(--border)' }}
                angle={-45}
                textAnchor="end"
                height={60}
                minTickGap={50}
              />
              <YAxis
                yAxisId="left"
                scale="log"
                domain={[1 / (24 * 60), 365]}
                ticks={TIME_TICKS}
                tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={{ stroke: 'var(--border)' }}
                tickFormatter={(v) => formatTimeTick(v)}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fill: 'var(--muted-foreground)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={{ stroke: 'var(--border)' }}
                domain={['dataMin', 'dataMax']}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine yAxisId="right" y={LMin} stroke="var(--muted-foreground)" strokeDasharray="5 5" />
              <ReferenceLine yAxisId="left" x={recommendedPoint.DFormatted} stroke="var(--chart-3)" strokeDasharray="5 5" label={{ value: '推荐', fill: 'var(--chart-3)', fontSize: 10, position: 'top' }} />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="TDays"
                stroke="var(--chart-2)"
                strokeWidth={2}
                dot={false}
                name="时间"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="L"
                stroke="var(--chart-1)"
                strokeWidth={2}
                dot={false}
                name="损失"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default function CalculatorPage() {
  // 从数据库加载GPU数据
  const { gpus: dbGpus, loading: gpuLoading } = useGPUs();
  // 从数据库加载使用统计
  const { stats } = useStats();
  
  // GPU 列表直接从数据库加载
  const gpuList = dbGpus;
  
  // 数据加载状态
  const hasGpuData = !gpuLoading && gpuList.length > 0;
  
  // 输入状态
  const [params, setParams] = useState<number>(1e9);
  const [paramsInput, setParamsInput] = useState<string>('1');
  const [paramsUnit, setParamsUnit] = useState<string>('B');
  const [modelType, setModelType] = useState<ModelType>('dense');
  const [sparsity, setSparsity] = useState<number>(0);

  // 数据量（可选）
  const [trainTokens, setTrainTokens] = useState<number>(0);
  const [tokensInput, setTokensInput] = useState<string>('');
  const [tokensUnit, setTokensUnit] = useState<string>('T');

  // GPU选择
  const [selectedGpuIds, setSelectedGpuIds] = useState<string[]>([]);
  const [gpuCount, setGpuCount] = useState<number>(1);
  const [gpuCountInput, setGpuCountInput] = useState<string>('1');
  const [utilization, setUtilization] = useState<number>(0.5);
  const [utilizationInput, setUtilizationInput] = useState<string>('50');

  // 弹窗状态
  const [dialogOpen, setDialogOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  // 弹窗分页状态
  const [dialogCurrentPage, setDialogCurrentPage] = useState(1);
  const dialogPageSize = 20;
  
  // 结果分页状态
  const [resultCurrentPage, setResultCurrentPage] = useState(1);
  const resultPageSize = 10;

  // 处理参数量输入
  const handleParamsChange = (value: string, unit: string) => {
    const num = parseFloat(value) || 0;
    const multiplier = unit === 'T' ? 1e12 : unit === 'B' ? 1e9 : unit === 'M' ? 1e6 : 1;
    setParams(num * multiplier);
  };

  // 处理数据量输入
  const handleTokensChange = (value: string, unit: string) => {
    setTokensInput(value);
    const num = parseFloat(value) || 0;
    const multiplier = unit === 'P' ? 1e15 : unit === 'T' ? 1e12 : unit === 'B' ? 1e9 : unit === 'M' ? 1e6 : 1;
    setTrainTokens(num * multiplier);
  };

  // 切换GPU选择
  const toggleGpuSelection = (gpuId: string) => {
    setSelectedGpuIds((prev) =>
      prev.includes(gpuId) ? prev.filter((id) => id !== gpuId) : [...prev, gpuId]
    );
  };

  // 清除所有选择
  const clearSelection = () => {
    setSelectedGpuIds([]);
  };

  // 显示的GPU列表（用于计算结果，过滤显存不足的GPU）
  const allDisplayGpus = useMemo(() => {
    if (gpuList.length === 0) return [];
    const baseList = selectedGpuIds.length === 0
      ? gpuList
      : gpuList.filter((gpu) => selectedGpuIds.includes(gpu.id));
    
    // 过滤显存不足的GPU
    return baseList.filter((gpu) => 
      checkMemoryConstraint(params, gpu.memory_gb, gpuCount)
    );
  }, [selectedGpuIds, params, gpuCount, gpuList.length]);

  // 结果分页后的GPU列表
  const displayGpus = useMemo(() => {
    const start = (resultCurrentPage - 1) * resultPageSize;
    const end = start + resultPageSize;
    return allDisplayGpus.slice(start, end);
  }, [allDisplayGpus, resultCurrentPage, resultPageSize]);

  // 结果总页数
  const resultTotalPages = Math.ceil(allDisplayGpus.length / resultPageSize);

  // 已选择的GPU对象列表
  const selectedGpus = useMemo(() => {
    if (gpuList.length === 0) return [];
    return gpuList.filter((gpu) => selectedGpuIds.includes(gpu.id));
  }, [selectedGpuIds, gpuList.length]);

  // 弹窗中筛选后的GPU列表（也过滤显存不足的GPU）
  const filteredGpus = useMemo(() => {
    if (gpuList.length === 0) return [];
    const baseList = gpuList.filter((gpu) => 
      checkMemoryConstraint(params, gpu.memory_gb, gpuCount)
    );
    
    if (!searchQuery.trim()) return baseList;
    const query = searchQuery.toLowerCase();
    return baseList.filter(
      (gpu) =>
        gpu.name.toLowerCase().includes(query) ||
        gpu.id.toLowerCase().includes(query)
    );
  }, [searchQuery, params, gpuCount, gpuList.length]);

  // 弹窗分页后的GPU列表
  const paginatedGpus = useMemo(() => {
    const start = (dialogCurrentPage - 1) * dialogPageSize;
    const end = start + dialogPageSize;
    return filteredGpus.slice(start, end);
  }, [filteredGpus, dialogCurrentPage, dialogPageSize]);

  // 弹窗总页数
  const dialogTotalPages = Math.ceil(filteredGpus.length / dialogPageSize);

  // 搜索变化时重置弹窗页码
  useEffect(() => {
    setDialogCurrentPage(1);
  }, [searchQuery]);
  
  // 选择变化时重置结果页码
  useEffect(() => {
    setResultCurrentPage(1);
  }, [selectedGpuIds]);

  // 记录页面访问统计 - 只在首次挂载时执行
  useEffect(() => {
    recordStat({ action: 'page_view' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 记录计算统计 - 使用 ref 跟踪上一次参数，避免重复记录
  const lastCalcParamsRef = useRef<string>('');
  useEffect(() => {
    // 生成参数签名
    const paramSignature = `${params}-${modelType}-${sparsity}-${trainTokens}`;
    
    // 只有参数真正变化时才记录
    if (lastCalcParamsRef.current !== paramSignature) {
      lastCalcParamsRef.current = paramSignature;
      
      // 延迟记录，等待用户完成调整
      const timer = setTimeout(() => {
        recordStat({ action: 'calculation' });
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [params, modelType, sparsity, trainTokens]);

  // 记录显卡选择统计 - 使用 ref 跟踪上一次数量
  const lastGpuCountRef = useRef(0);
  useEffect(() => {
    // 只有显卡数量变化且大于0时才记录
    if (selectedGpuIds.length > 0 && selectedGpuIds.length !== lastGpuCountRef.current) {
      lastGpuCountRef.current = selectedGpuIds.length;
      recordStat({ action: 'gpu_selection' });
    }
  }, [selectedGpuIds.length]);

  // 获取当前系数
  const currentCoeffs = useMemo(() => {
    const S = sparsity / 100;
    return getCoefficients(modelType, S);
  }, [modelType, sparsity]);

  // 计算损失下界
  const lossMin = useMemo(() => {
    const S = sparsity / 100;
    if (modelType === 'dense' || S === 0) {
      return LMinDense(params, COEFFS_DENSE);
    } else {
      return LMinSparse(params, S, currentCoeffs);
    }
  }, [params, modelType, sparsity, currentCoeffs]);

  // 计算推荐数据量 (Chinchilla最优: D_opt ≈ 20 × N)
  const recommendedTokens = useMemo(() => {
    return params * CHINCHILLA_OPTIMAL_RATIO;
  }, [params]);

  // 是否有数据量输入
  const hasTrainTokens = trainTokens > 0;

  // 加载中状态
  if (gpuLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">加载中...</p>
        </div>
      </div>
    );
  }

  // 无数据错误状态
  if (!hasGpuData) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center max-w-md px-4">
          <div className="w-16 h-16 rounded-xl bg-destructive/10 flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="w-8 h-8 text-destructive" />
          </div>
          <h2 className="text-xl font-semibold text-foreground mb-2">GPU 数据加载失败</h2>
          <p className="text-muted-foreground mb-4">
            无法加载 GPU 数据，请检查数据库配置或联系管理员。
          </p>
          <Button onClick={() => window.location.reload()} variant="outline">
            重新加载
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-xl bg-muted/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center shadow-sm">
                <Calculator className="w-5 h-5 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-foreground tracking-tight">
                  LLM训练缩放律计算器
                </h1>
                <p className="text-xs text-muted-foreground">基于论文 arXiv:2508.06617 · 训练阶段预测</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {/* 使用统计 */}
              <div className="hidden sm:flex items-center gap-1 px-3 py-2 text-xs text-muted-foreground bg-muted/60 border border-border rounded-lg">
                <span>计算次数:</span>
                <span className="font-medium text-foreground">{stats.total_calculations}</span>
                <span className="mx-1">·</span>
                <span>访问:</span>
                <span className="font-medium text-foreground">{stats.total_page_views}</span>
              </div>
              <a
                href="https://arxiv.org/abs/2508.06617"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground bg-muted/60 hover:bg-accent border border-border rounded-lg transition-colors"
              >
                <FileText className="w-4 h-4" />
                <span className="hidden sm:inline">论文</span>
              </a>
              <a
                href="https://github.com/DeliWang/LLM-Scaling-Law-Calculator"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground bg-muted/60 hover:bg-accent border border-border rounded-lg transition-colors"
              >
                <Github className="w-4 h-4" />
                <span className="hidden sm:inline">GitHub</span>
              </a>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* 上半部分：输入区域 */}
        <div className="bg-muted/50 border border-border/50 rounded-2xl p-4 mb-6 backdrop-blur-sm">
          {/* 第一行：参数量 + 模型类型 + 数据量 + 损失下界 */}
          <div className="flex flex-col md:flex-row gap-4 mb-4">
            {/* 参数量 N */}
            <div className="w-full md:w-44 space-y-2 shrink-0">
              <Label className="text-muted-foreground flex items-center gap-2">
                <Database className="w-4 h-4 text-chart-1" />
                活跃参数量
                <span className="text-muted-foreground text-xs ml-auto">N</span>
              </Label>
              <div className="flex gap-2">
                <Input
                  type="text"
                  inputMode="decimal"
                  value={paramsInput}
                  onChange={(e) => {
                    setParamsInput(e.target.value);
                    handleParamsChange(e.target.value, paramsUnit);
                  }}
                  className="bg-muted/60 border-border text-foreground flex-1"
                  placeholder="参数量"
                />
                <Select
                  value={paramsUnit}
                  onValueChange={(v) => {
                    setParamsUnit(v);
                    handleParamsChange(paramsInput, v);
                  }}
                >
                  <SelectTrigger className="w-16 bg-muted/60 border-border text-foreground">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    <SelectItem value="M">M</SelectItem>
                    <SelectItem value="B">B</SelectItem>
                    <SelectItem value="T">T</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* 模型类型 */}
            <div className="w-full md:w-44 space-y-2 shrink-0">
              <Label className="text-muted-foreground flex items-center gap-2">
                <Cpu className="w-4 h-4 text-chart-4" />
                模型类型
              </Label>
              <Select value={modelType} onValueChange={(v: ModelType) => setModelType(v)}>
                <SelectTrigger className="bg-muted/60 border-border text-foreground">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-card border-border">
                  <SelectItem value="dense">稠密模型 (Dense)</SelectItem>
                  <SelectItem value="sparse_pruned">剪枝模型 (Pruned)</SelectItem>
                  <SelectItem value="sparse_moe">MoE 模型 (MoE)</SelectItem>
                </SelectContent>
              </Select>
              {modelType !== 'dense' && (
                <div className="flex items-center gap-2">
                  <Slider
                    value={[sparsity]}
                    onValueChange={([v]) => setSparsity(v)}
                    max={90}
                    step={5}
                    className="flex-1"
                  />
                  <span className="text-xs text-muted-foreground w-14 shrink-0">S={sparsity}%</span>
                </div>
              )}
            </div>

            {/* 数据量 D */}
            <div className="w-full md:w-72 space-y-2 shrink-0">
              <Label className="text-muted-foreground flex items-center gap-2">
                <Zap className="w-4 h-4 text-chart-4" />
                数据量
                <span className="text-muted-foreground text-xs">(可选)</span>
                <span className="text-muted-foreground text-xs ml-auto">D</span>
              </Label>
              <div className="flex gap-2">
                <Input
                  type="text"
                  inputMode="decimal"
                  value={tokensInput}
                  onChange={(e) => handleTokensChange(e.target.value, tokensUnit)}
                  className="bg-muted/60 border-border text-foreground flex-1"
                  placeholder="可选"
                />
                <Select
                  value={tokensUnit}
                  onValueChange={(v) => {
                    setTokensUnit(v);
                    handleTokensChange(tokensInput, v);
                  }}
                >
                  <SelectTrigger className="w-16 bg-muted/60 border-border text-foreground">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-card border-border">
                    <SelectItem value="M">M</SelectItem>
                    <SelectItem value="B">B</SelectItem>
                    <SelectItem value="T">T</SelectItem>
                    <SelectItem value="P">P</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  type="button"
                  size="sm"
                  onClick={() => {
                    const rec = recommendedTokens;
                    if (rec >= 1e15) {
                      setTokensInput((rec / 1e15).toFixed(2));
                      setTokensUnit('P');
                    } else if (rec >= 1e12) {
                      setTokensInput((rec / 1e12).toFixed(1));
                      setTokensUnit('T');
                    } else if (rec >= 1e9) {
                      setTokensInput((rec / 1e9).toFixed(1));
                      setTokensUnit('B');
                    } else if (rec >= 1e6) {
                      setTokensInput((rec / 1e6).toFixed(1));
                      setTokensUnit('M');
                    } else {
                      setTokensInput(rec.toFixed(0));
                      setTokensUnit('M');
                    }
                    setTrainTokens(rec);
                  }}
                  className="bg-primary hover:bg-primary/90 text-primary-foreground shrink-0 h-9"
                >
                  推荐 {formatTokens(recommendedTokens)}
                </Button>
              </div>
            </div>

            {/* 损失下界 L_min - 占据剩余空间 */}
            <div className="flex-1 min-w-[200px] space-y-2">
              <Label className="text-muted-foreground flex items-center gap-2">
                <TrendingDown className="w-4 h-4 text-chart-2" />
                损失下界
                <span className="text-muted-foreground text-xs ml-auto">L<tspan baselineShift="sub" fontSize="10">min</tspan></span>
              </Label>
              <div className="h-10 bg-muted/60 border border-border rounded-md px-3 flex items-center gap-3">
                <span className="text-sm text-foreground font-mono shrink-0">{lossMin.toFixed(4)}</span>
                {modelType === 'dense' ? (
                  <span className="text-xs text-muted-foreground">= e + a/N<sup>α</sup></span>
                ) : (
                  <span className="text-xs text-muted-foreground">
                    = e(1-S)<sup>γ</sup> + (a(1-S)<sup>α</sup> + cS)/N<sup>α</sup>
                  </span>
                )}
                <div className="flex items-center gap-2 text-xs text-muted-foreground ml-auto shrink-0">
                  <span>e={currentCoeffs.e}</span>
                  <span>a={currentCoeffs.a}</span>
                  <span>α={currentCoeffs.alpha}</span>
                  {modelType !== 'dense' && (
                    <>
                      <span>c={currentCoeffs.c}</span>
                      <span>γ={currentCoeffs.gamma}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* 第二行：显卡选择 + GPU数量 + 利用率 */}
          <div className="border-t border-border pt-4">
            <div className="flex flex-col lg:flex-row gap-6">
              {/* 显卡选择 - 占50% */}
              <div className="flex-[2] space-y-2">
                <Label className="text-muted-foreground flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-chart-2" />
                  选择显卡
                </Label>
                <div className="flex gap-2">
                  <Input
                    value={
                      selectedGpus.length > 0
                        ? selectedGpus.map((g) => g.name).join('、')
                        : '未选择（显示全部显卡）'
                    }
                    readOnly
                    className="bg-muted/60 border-border text-foreground cursor-pointer"
                    onClick={() => setDialogOpen(true)}
                  />
                  <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                    <DialogTrigger asChild>
                      <Button className="bg-primary hover:bg-primary/90 text-primary-foreground shrink-0">
                        <Plus className="w-4 h-4 mr-1" />
                        添加
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="bg-card border-border max-w-4xl sm:max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
                      <DialogHeader>
                        <DialogTitle className="text-foreground">选择显卡</DialogTitle>
                        <p className="text-xs text-muted-foreground mt-1">
                          当前模型每卡需显存 <span className="text-chart-4">{getRequiredMemoryPerGpu(params, gpuCount).toFixed(1)} GB</span>，已过滤显存不足的显卡
                        </p>
                      </DialogHeader>

                      <div className="flex-1 overflow-hidden flex flex-col gap-4">
                        {/* 已选择的显卡 */}
                        <div>
                          <Label className="text-muted-foreground text-sm mb-2 block">
                            已选择 ({selectedGpuIds.length})
                          </Label>
                          <div className="min-h-[60px] bg-muted/60 rounded-lg p-3 flex flex-wrap gap-2">
                            {selectedGpus.length === 0 ? (
                              <span className="text-muted-foreground text-sm">点击下方显卡添加</span>
                            ) : (
                              selectedGpus.map((gpu) => (
                                <Badge
                                  key={gpu.id}
                                  className="bg-primary text-primary-foreground cursor-pointer hover:bg-primary/90"
                                  onClick={() => toggleGpuSelection(gpu.id)}
                                >
                                  {gpu.name}
                                  <X className="w-3 h-3 ml-1" />
                                </Badge>
                              ))
                            )}
                          </div>
                        </div>

                        {/* 搜索框 */}
                        <div className="relative">
                          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                          <Input
                            placeholder="搜索显卡..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="bg-muted/60 border-border text-foreground pl-10"
                          />
                        </div>

                        {/* GPU列表 */}
                        <div className="flex-1 overflow-auto">
                          <Label className="text-muted-foreground text-sm mb-2 block">
                            显卡列表 ({filteredGpus.length})
                          </Label>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            {paginatedGpus.map((gpu) => {
                              const isSelected = selectedGpuIds.includes(gpu.id);
                              return (
                                <div
                                  key={gpu.id}
                                  onClick={() => toggleGpuSelection(gpu.id)}
                                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                                    isSelected
                                      ? 'bg-primary/20 border-primary/50'
                                      : 'bg-muted/50 border-border/50 hover:border-border'
                                  }`}
                                >
                                  <div className="flex items-start justify-between">
                                    <div>
                                      <p className="text-foreground text-sm font-medium">{gpu.name}</p>
                                      <p className="text-muted-foreground text-xs mt-1">
                                        {gpu.tfops} TFLOPS · {gpu.memory_gb}GB
                                      </p>
                                    </div>
                                    {isSelected ? (
                                      <Check className="w-5 h-5 text-chart-1" />
                                    ) : (
                                      <Plus className="w-5 h-5 text-muted-foreground" />
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                          
                          {/* 分页控件 */}
                          {dialogTotalPages > 1 && (
                            <div className="flex items-center justify-center gap-2 mt-4 pt-4 border-t border-border">
                              <Button
                                variant="outline"
                                size="sm"
                                disabled={dialogCurrentPage === 1}
                                onClick={() => setDialogCurrentPage((p) => p - 1)}
                                className="text-muted-foreground border-border hover:text-foreground"
                              >
                                上一页
                              </Button>
                              <div className="flex items-center gap-1">
                                <span className="text-muted-foreground text-sm">
                                  {dialogCurrentPage} / {dialogTotalPages}
                                </span>
                              </div>
                              <Button
                                variant="outline"
                                size="sm"
                                disabled={dialogCurrentPage === dialogTotalPages}
                                onClick={() => setDialogCurrentPage((p) => p + 1)}
                                className="text-muted-foreground border-border hover:text-foreground"
                              >
                                下一页
                              </Button>
                            </div>
                          )}
                        </div>

                        {/* 底部操作按钮 */}
                        <div className="flex justify-between pt-4 border-t border-border">
                          <Button
                            onClick={clearSelection}
                            className="bg-red-600 hover:bg-red-700 text-white flex items-center justify-center"
                          >
                            清空选择
                          </Button>
                          <Button
                            onClick={() => setDialogOpen(false)}
                            className="bg-primary hover:bg-primary/90 text-primary-foreground"
                          >
                            确认
                          </Button>
                        </div>
                      </div>
                    </DialogContent>
                  </Dialog>
                  {selectedGpuIds.length > 0 && (
                    <Button
                      onClick={clearSelection}
                      className="bg-red-600 hover:bg-red-700 text-white shrink-0 flex items-center justify-center"
                    >
                      清空
                    </Button>
                  )}
                </div>
              </div>

              {/* GPU数量 - 占25% */}
              <div className="flex-1 space-y-2">
                <Label className="text-muted-foreground text-sm">GPU 数量</Label>
                <div className="flex items-center gap-2">
                  <Slider
                    value={[gpuCount]}
                    onValueChange={([v]) => {
                      setGpuCount(v);
                      setGpuCountInput(String(v));
                    }}
                    min={1}
                    max={1024}
                    step={1}
                    className="flex-1"
                  />
                  <Input
                    type="text"
                    inputMode="numeric"
                    value={gpuCountInput}
                    onChange={(e) => {
                      const val = e.target.value;
                      setGpuCountInput(val);
                      const v = parseInt(val) || 1;
                      setGpuCount(Math.max(1, Math.min(1024, v)));
                    }}
                    className="w-20 bg-muted/60 border-border text-foreground text-center text-sm"
                  />
                </div>
              </div>

              {/* GPU利用率 - 占25% */}
              <div className="flex-1 space-y-2">
                <Label className="text-muted-foreground text-sm">利用率</Label>
                <div className="flex items-center gap-2">
                  <Slider
                    value={[utilization]}
                    onValueChange={([v]) => {
                      setUtilization(v);
                      setUtilizationInput(String(Math.round(v * 100)));
                    }}
                    min={0.1}
                    max={0.8}
                    step={0.05}
                    className="flex-1"
                  />
                  <Input
                    type="text"
                    inputMode="numeric"
                    value={utilizationInput}
                    onChange={(e) => {
                      const val = e.target.value;
                      setUtilizationInput(val);
                      const v = parseInt(val) || 50;
                      setUtilization(Math.max(0.1, Math.min(0.8, v / 100)));
                    }}
                    className="w-16 bg-muted/60 border-border text-foreground text-center text-sm"
                  />
                  <span className="text-muted-foreground text-sm">%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 下半部分：结果列表 */}
        <div className="space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-medium text-foreground">
                计算结果
                {selectedGpuIds.length > 0 && (
                  <span className="text-muted-foreground text-sm font-normal ml-2">
                    （{selectedGpuIds.length} 个显卡）
                  </span>
                )}
              </h2>
              <div className="text-xs text-muted-foreground bg-muted/60 px-2 py-1 rounded">
                每卡需显存: <span className="text-chart-4 font-medium">{getRequiredMemoryPerGpu(params, gpuCount).toFixed(1)} GB</span>
              </div>
            </div>
            {hasTrainTokens ? (
              <p className="text-xs text-muted-foreground">
                数据量: {formatTokens(trainTokens)} · 显示具体计算结果
              </p>
            ) : (
              <div className="flex items-center gap-4 text-xs">
                <span className="text-muted-foreground">横轴：训练数据量</span>
                <span className="text-border">|</span>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-0.5 bg-chart-2 rounded" />
                  <span className="text-muted-foreground">左轴：训练时间</span>
                </div>
                <span className="text-border">|</span>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-0.5 bg-chart-1 rounded" />
                  <span className="text-muted-foreground">右轴：目标损失</span>
                </div>
              </div>
            )}
          </div>
          
          {allDisplayGpus.length === 0 && (
            <div className="bg-muted/60 border border-amber-800/50 rounded-xl p-6 text-center">
              <p className="text-chart-4 text-sm">
                当前配置下，所有 GPU 显存均不足
              </p>
              <p className="text-muted-foreground text-xs mt-1">
                每卡需要 {getRequiredMemoryPerGpu(params, gpuCount).toFixed(1)} GB 显存，请增加 GPU 数量或减少模型参数
              </p>
            </div>
          )}

          <div className="grid gap-3">
            {displayGpus.map((gpu) =>
              hasTrainTokens ? (
                <GPUSpecificResultCard
                  key={gpu.id}
                  gpu={gpu}
                  N={params}
                  D={trainTokens}
                  modelType={modelType}
                  S={sparsity / 100}
                  gpuCount={gpuCount}
                  utilization={utilization}
                />
              ) : (
                <GPUChartCard
                  key={gpu.id}
                  gpu={gpu}
                  N={params}
                  modelType={modelType}
                  S={sparsity / 100}
                  gpuCount={gpuCount}
                  utilization={utilization}
                />
              )
            )}
          </div>
          
          {/* 结果分页控件 */}
          {resultTotalPages > 1 && (
            <div className="flex items-center justify-center gap-4 mt-6">
              <Button
                variant="outline"
                size="sm"
                disabled={resultCurrentPage === 1}
                onClick={() => setResultCurrentPage((p) => p - 1)}
                className="text-muted-foreground border-border hover:text-foreground"
              >
                上一页
              </Button>
              <div className="flex items-center gap-2">
                {Array.from({ length: Math.min(5, resultTotalPages) }, (_, i) => {
                  let pageNum: number;
                  if (resultTotalPages <= 5) {
                    pageNum = i + 1;
                  } else if (resultCurrentPage <= 3) {
                    pageNum = i + 1;
                  } else if (resultCurrentPage >= resultTotalPages - 2) {
                    pageNum = resultTotalPages - 4 + i;
                  } else {
                    pageNum = resultCurrentPage - 2 + i;
                  }
                  return (
                    <Button
                      key={pageNum}
                      variant={resultCurrentPage === pageNum ? "default" : "outline"}
                      size="sm"
                      onClick={() => setResultCurrentPage(pageNum)}
                      className={resultCurrentPage === pageNum 
                        ? "bg-primary text-primary-foreground" 
                        : "text-muted-foreground border-border hover:text-foreground"}
                    >
                      {pageNum}
                    </Button>
                  );
                })}
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={resultCurrentPage === resultTotalPages}
                onClick={() => setResultCurrentPage((p) => p + 1)}
                className="text-muted-foreground border-border hover:text-foreground"
              >
                下一页
              </Button>
            </div>
          )}
        </div>

        {/* 底部说明 */}
        <footer className="mt-12 pb-8 text-center">
          <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground flex-wrap">
            <a 
              href="https://www.coze.cn" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-muted/50 hover:bg-muted transition-colors shrink-0"
            >
              <img src="/coze-logo.svg" alt="Coze" className="w-5 h-5" />
              <span className="font-medium">Powered by Coze</span>
            </a>
            <span>●</span>
            <span>基于论文 arXiv:2508.06617</span>
            <span>●</span>
            <span>结果为经验公式估算，仅供参考</span>
          </div>
        </footer>
      </main>
    </div>
  );
}
