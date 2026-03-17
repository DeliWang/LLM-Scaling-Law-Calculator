/**
 * GPU 配置类型和工具函数
 * GPU 数据从 SQLite 数据库加载
 */

export interface GPU {
  id: string;
  name: string;
  tfops: number;          // FP16/BF16 算力 (TFLOPS)
  utilization: number;    // 默认利用率
  memory_gb: number;      // 显存 (GB)
  price_usd: number | null; // 发布价格 (美元)
}

/**
 * 检查 GPU 显存是否足够容纳模型训练
 * 训练时每参数约需要 20 字节显存（FP16模型参数 + 梯度 + Adam优化器状态）
 * @param params 参数数量
 * @param memoryGB GPU显存（GB）
 * @param gpuCount GPU数量（用于多卡并行）
 * @returns 是否有足够显存
 */
export function checkMemoryConstraint(params: number, memoryGB: number, gpuCount: number = 1): boolean {
  // 每参数约 20 字节（FP16 训练：模型2B + 梯度2B + 优化器状态16B）
  const bytesPerParam = 20;
  const totalRequiredGB = (params * bytesPerParam) / 1e9;
  
  // 多卡并行时，显存需求分摊到各卡
  const requiredPerGpuGB = totalRequiredGB / gpuCount;
  
  return requiredPerGpuGB <= memoryGB;
}

/**
 * 计算训练所需显存
 * @param params 参数数量
 * @param gpuCount GPU数量
 * @returns 每卡所需显存（GB）
 */
export function getRequiredMemoryPerGpu(params: number, gpuCount: number = 1): number {
  const bytesPerParam = 20;
  const totalRequiredGB = (params * bytesPerParam) / 1e9;
  return totalRequiredGB / gpuCount;
}
