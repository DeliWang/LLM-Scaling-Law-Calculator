import { NextRequest, NextResponse } from 'next/server';
import { initDatabase } from '@/lib/db/sqlite';
import { importGPUDataFromCSV } from '@/lib/db/import-gpus';
import { getGPUs, getGPUCount, type GPURecord } from '@/lib/db/sqlite-gpus';

// 初始化数据库并导入数据（只执行一次）
let initialized = false;
function ensureInit() {
  if (!initialized) {
    initDatabase();
    importGPUDataFromCSV(); // 从 CSV 导入数据
    initialized = true;
  }
}

// 前端兼容的GPU格式
interface FrontendGPU {
  id: string;
  name: string;
  tfops: number;
  utilization: number;
  memory_gb: number;
  price_usd: number | null;
}

// 将数据库GPU转换为前端兼容格式
function toFrontendFormat(gpu: GPURecord): FrontendGPU {
  return {
    id: gpu.gpu_id,
    name: gpu.name,
    tfops: gpu.tflops || 0,
    utilization: 0.5, // 默认利用率50%
    memory_gb: gpu.memory_gb || 0,
    price_usd: gpu.price_usd || gpu.launch_price_usd || null,
  };
}

export async function GET(request: NextRequest) {
  try {
    ensureInit();
    
    const searchParams = request.nextUrl.searchParams;
    
    // 解析查询参数
    const minMemory = searchParams.get('minMemory') 
      ? parseFloat(searchParams.get('minMemory')!) 
      : undefined;
    const minTflops = searchParams.get('minTflops') 
      ? parseFloat(searchParams.get('minTflops')!) 
      : undefined;
    const brand = searchParams.get('brand') || undefined;
    const isActive = searchParams.get('isActive') === 'false' 
      ? false 
      : searchParams.get('isActive') === 'true' 
        ? true 
        : undefined;
    const limit = searchParams.get('limit') 
      ? parseInt(searchParams.get('limit')!) 
      : undefined;
    const offset = searchParams.get('offset') 
      ? parseInt(searchParams.get('offset')!) 
      : undefined;

    // 获取显卡列表
    const dbGpus = getGPUs({
      minMemory,
      minTflops,
      brand,
      isActive,
      limit,
      offset,
    });

    // 转换为前端兼容格式
    const gpus = dbGpus.map(toFrontendFormat);

    // 获取总数
    const total = getGPUCount();

    return NextResponse.json({
      success: true,
      data: gpus,
      meta: {
        total,
        count: gpus.length,
        limit,
        offset,
      },
    });
  } catch (error) {
    console.error('Error in GET /api/gpus:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to fetch GPUs',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
