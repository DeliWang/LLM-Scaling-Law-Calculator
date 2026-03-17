import { NextRequest, NextResponse } from 'next/server';
import { initDatabase } from '@/lib/db/sqlite';
import { getGPUById, getGPUByGpuId, type GPURecord } from '@/lib/db/sqlite-gpus';

let initialized = false;
function ensureInit() {
  if (!initialized) {
    initDatabase();
    initialized = true;
  }
}

interface FrontendGPU {
  id: string;
  name: string;
  tfops: number;
  utilization: number;
  memory_gb: number;
  price_usd: number | null;
}

function toFrontendFormat(gpu: GPURecord): FrontendGPU {
  return {
    id: gpu.gpu_id,
    name: gpu.name,
    tfops: gpu.tflops || 0,
    utilization: 0.5,
    memory_gb: gpu.memory_gb || 0,
    price_usd: gpu.price_usd || gpu.launch_price_usd || null,
  };
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    ensureInit();
    
    const { id } = await params;
    
    // 先尝试按数字ID查询，再按gpu_id查询
    let gpu: GPURecord | null = null;
    
    const numericId = parseInt(id, 10);
    if (!isNaN(numericId)) {
      gpu = getGPUById(numericId);
    }
    
    if (!gpu) {
      gpu = getGPUByGpuId(id);
    }

    if (!gpu) {
      return NextResponse.json(
        { success: false, error: 'GPU not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: toFrontendFormat(gpu),
    });
  } catch (error) {
    console.error('Error in GET /api/gpus/[id]:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to fetch GPU',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
