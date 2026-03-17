import { getDatabase } from './sqlite';

// 统计记录类型
export interface UsageStatRecord {
  id: number;
  session_id: string | null;
  action: string;
  params_n: number | null;
  params_model_type: string | null;
  params_sparsity: number | null;
  params_tokens: number | null;
  params_gpu_count: number | null;
  params_utilization: number | null;
  result_loss: number | null;
  result_time_days: number | null;
  result_flops: number | null;
  selected_gpu_ids: string | null;
  created_at: string;
}

// 记录统计
export function recordStat(params: {
  action: string;
  session_id?: string;
  params_n?: number;
  params_model_type?: string;
  params_sparsity?: number;
  params_tokens?: number;
  params_gpu_count?: number;
  params_utilization?: number;
  result_loss?: number;
  result_time_days?: number;
  result_flops?: number;
  selected_gpu_ids?: number[];
}): UsageStatRecord {
  const db = getDatabase();
  
  const stmt = db.prepare(`
    INSERT INTO usage_stats (session_id, action, params_n, params_model_type, params_sparsity, params_tokens, params_gpu_count, params_utilization, result_loss, result_time_days, result_flops, selected_gpu_ids)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  
  const result = stmt.run(
    params.session_id || null,
    params.action,
    params.params_n || null,
    params.params_model_type || null,
    params.params_sparsity || null,
    params.params_tokens || null,
    params.params_gpu_count || null,
    params.params_utilization || null,
    params.result_loss || null,
    params.result_time_days || null,
    params.result_flops || null,
    params.selected_gpu_ids ? JSON.stringify(params.selected_gpu_ids) : null
  );
  
  return {
    id: result.lastInsertRowid as number,
    session_id: params.session_id || null,
    action: params.action,
    params_n: params.params_n || null,
    params_model_type: params.params_model_type || null,
    params_sparsity: params.params_sparsity || null,
    params_tokens: params.params_tokens || null,
    params_gpu_count: params.params_gpu_count || null,
    params_utilization: params.params_utilization || null,
    result_loss: params.result_loss || null,
    result_time_days: params.result_time_days || null,
    result_flops: params.result_flops || null,
    selected_gpu_ids: params.selected_gpu_ids ? JSON.stringify(params.selected_gpu_ids) : null,
    created_at: new Date().toISOString(),
  };
}

// 统计摘要
export interface StatsSummary {
  total_calculations: number;
  total_page_views: number;
  avg_model_params: number | null;
  total_training_hours: number;
}

// 获取统计摘要
export function getUsageSummary(period: 'day' | 'week' | 'month' | 'all' = 'week'): StatsSummary {
  const db = getDatabase();
  
  // 计算时间范围
  let timeCondition = '';
  switch (period) {
    case 'day':
      timeCondition = "created_at >= datetime('now', '-1 day')";
      break;
    case 'week':
      timeCondition = "created_at >= datetime('now', '-7 days')";
      break;
    case 'month':
      timeCondition = "created_at >= datetime('now', '-30 days')";
      break;
    case 'all':
    default:
      timeCondition = '1=1';
      break;
  }
  
  // 计算次数
  const calcStmt = db.prepare(`
    SELECT COUNT(*) as count FROM usage_stats 
    WHERE action = 'calculation' AND ${timeCondition}
  `);
  const calcResult = calcStmt.get() as { count: number };
  
  // 页面访问次数
  const viewStmt = db.prepare(`
    SELECT COUNT(*) as count FROM usage_stats 
    WHERE action = 'page_view' AND ${timeCondition}
  `);
  const viewResult = viewStmt.get() as { count: number };
  
  // 平均模型参数
  const avgStmt = db.prepare(`
    SELECT AVG(params_n) as avg FROM usage_stats 
    WHERE action = 'calculation' AND params_n IS NOT NULL AND ${timeCondition}
  `);
  const avgResult = avgStmt.get() as { avg: number | null };
  
  // 总训练时间
  const timeStmt = db.prepare(`
    SELECT SUM(result_time_days) as total FROM usage_stats 
    WHERE action = 'calculation' AND result_time_days IS NOT NULL AND ${timeCondition}
  `);
  const timeResult = timeStmt.get() as { total: number | null };
  
  return {
    total_calculations: calcResult.count,
    total_page_views: viewResult.count,
    avg_model_params: avgResult.avg,
    total_training_hours: (timeResult.total || 0) * 24,
  };
}
