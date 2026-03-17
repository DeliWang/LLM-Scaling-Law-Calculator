import { NextRequest, NextResponse } from 'next/server';
import { initDatabase } from '@/lib/db/sqlite';
import { getGPUPriceHistory, getGPUByGpuId, addGPUPrice } from '@/lib/db/sqlite-gpus';

let initialized = false;
function ensureInit() {
  if (!initialized) {
    initDatabase();
    initialized = true;
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    ensureInit();
    
    const { id } = await params;
    
    // 验证GPU存在
    const gpu = getGPUByGpuId(id);
    if (!gpu) {
      return NextResponse.json(
        { success: false, error: 'GPU not found' },
        { status: 404 }
      );
    }

    const prices = getGPUPriceHistory(id);

    return NextResponse.json({
      success: true,
      data: prices,
      meta: {
        gpuId: id,
        count: prices.length,
      },
    });
  } catch (error) {
    console.error('Error in GET /api/gpus/[id]/prices:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to fetch GPU price history',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

// 添加价格记录
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    ensureInit();
    
    const { id } = await params;
    const body = await request.json();
    
    // 验证GPU存在
    const gpu = getGPUByGpuId(id);
    if (!gpu) {
      return NextResponse.json(
        { success: false, error: 'GPU not found' },
        { status: 404 }
      );
    }
    
    if (!body.price_usd || typeof body.price_usd !== 'number') {
      return NextResponse.json(
        { success: false, error: 'price_usd is required' },
        { status: 400 }
      );
    }

    const price = addGPUPrice(id, body.price_usd, body.source);

    return NextResponse.json({
      success: true,
      data: price,
    });
  } catch (error) {
    console.error('Error in POST /api/gpus/[id]/prices:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to add GPU price',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
