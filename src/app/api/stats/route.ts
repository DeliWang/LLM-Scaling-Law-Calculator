import { NextRequest, NextResponse } from 'next/server';
import { initDatabase } from '@/lib/db/sqlite';
import { recordStat, getUsageSummary } from '@/lib/db/sqlite-stats';

let initialized = false;
function ensureInit() {
  if (!initialized) {
    initDatabase();
    initialized = true;
  }
}

/**
 * POST /api/stats - 记录使用统计
 */
export async function POST(request: NextRequest) {
  try {
    ensureInit();
    
    const body = await request.json();
    
    // 验证action
    const validActions = ['page_view', 'calculation', 'gpu_selection'];
    if (!body.action || !validActions.includes(body.action)) {
      return NextResponse.json(
        { error: 'Invalid action. Must be one of: page_view, calculation, gpu_selection' },
        { status: 400 }
      );
    }
    
    // 记录统计
    const stat = recordStat({
      action: body.action,
      session_id: body.session_id,
      params_n: body.params_n || body.model_params,
      params_model_type: body.params_model_type || body.model_type,
      params_sparsity: body.params_sparsity || body.sparsity_ratio,
      params_tokens: body.params_tokens || body.data_tokens,
      params_gpu_count: body.params_gpu_count || body.gpu_count,
      params_utilization: body.params_utilization,
      result_loss: body.result_loss,
      result_time_days: body.result_time_days || (body.training_time_hours ? body.training_time_hours / 24 : undefined),
      result_flops: body.result_flops,
      selected_gpu_ids: body.selected_gpu_ids || (body.gpu_id ? [body.gpu_id] : undefined),
    });
    
    return NextResponse.json({ success: true, id: stat.id });
  } catch (error) {
    console.error('Error recording stat:', error);
    return NextResponse.json(
      { error: 'Failed to record stat' },
      { status: 500 }
    );
  }
}

/**
 * GET /api/stats - 获取统计摘要
 */
export async function GET(request: NextRequest) {
  try {
    ensureInit();
    
    const searchParams = request.nextUrl.searchParams;
    const period = (searchParams.get('period') || 'all') as 'day' | 'week' | 'month' | 'all';
    
    const summary = getUsageSummary(period);
    
    return NextResponse.json(summary);
  } catch (error) {
    console.error('Error fetching stats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch stats' },
      { status: 500 }
    );
  }
}
