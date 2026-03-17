import { getDatabase } from './sqlite';

// GPU 数据类型
export interface GPURecord {
  id: number;
  gpu_id: string;
  name: string;
  brand: string | null;
  tflops: number | null;
  memory_gb: number | null;
  price_usd: number | null;
  launch_price_usd: number | null;
  architecture: string | null;
  release_date: string | null;
  tdp: number | null;
  is_active: boolean;
  raw_data: string | null;
  created_at: string;
  updated_at: string;
}

export interface GPUInsert {
  gpu_id: string;
  name: string;
  brand?: string | null;
  tflops?: number | null;
  memory_gb?: number | null;
  price_usd?: number | null;
  launch_price_usd?: number | null;
  architecture?: string | null;
  release_date?: string | null;
  tdp?: number | null;
  is_active?: boolean;
  raw_data?: string | null;
}

// 查询参数
export interface GetGPUsParams {
  minMemory?: number;
  minTflops?: number;
  brand?: string;
  isActive?: boolean;
  limit?: number;
  offset?: number;
}

// 获取 GPU 列表
export function getGPUs(params: GetGPUsParams = {}): GPURecord[] {
  const db = getDatabase();
  
  let sql = 'SELECT * FROM gpus WHERE 1=1';
  const conditions: string[] = [];
  const values: (string | number)[] = [];
  
  if (params.minMemory !== undefined) {
    conditions.push('memory_gb >= ?');
    values.push(params.minMemory);
  }
  
  if (params.minTflops !== undefined) {
    conditions.push('tflops >= ?');
    values.push(params.minTflops);
  }
  
  if (params.brand) {
    conditions.push('brand = ?');
    values.push(params.brand);
  }
  
  if (params.isActive !== undefined) {
    conditions.push('is_active = ?');
    values.push(params.isActive ? 1 : 0);
  }
  
  if (conditions.length > 0) {
    sql += ' AND ' + conditions.join(' AND ');
  }
  
  sql += ' ORDER BY name ASC';
  
  if (params.limit) {
    sql += ' LIMIT ?';
    values.push(params.limit);
    
    if (params.offset) {
      sql += ' OFFSET ?';
      values.push(params.offset);
    }
  }
  
  const stmt = db.prepare(sql);
  const rows = stmt.all(...values) as GPURecord[];
  
  // 转换 is_active 为布尔值
  return rows.map(row => ({
    ...row,
    is_active: Boolean(row.is_active),
  }));
}

// 获取 GPU 数量
export function getGPUCount(): number {
  const db = getDatabase();
  const stmt = db.prepare('SELECT COUNT(*) as count FROM gpus');
  const result = stmt.get() as { count: number };
  return result.count;
}

// 根据 ID 获取 GPU
export function getGPUById(id: number): GPURecord | null {
  const db = getDatabase();
  const stmt = db.prepare('SELECT * FROM gpus WHERE id = ?');
  const row = stmt.get(id) as GPURecord | undefined;
  
  if (!row) return null;
  
  return {
    ...row,
    is_active: Boolean(row.is_active),
  };
}

// 根据 gpu_id 获取 GPU
export function getGPUByGpuId(gpuId: string): GPURecord | null {
  const db = getDatabase();
  const stmt = db.prepare('SELECT * FROM gpus WHERE gpu_id = ?');
  const row = stmt.get(gpuId) as GPURecord | undefined;
  
  if (!row) return null;
  
  return {
    ...row,
    is_active: Boolean(row.is_active),
  };
}

// 插入 GPU
export function insertGPU(gpu: GPUInsert): GPURecord {
  const db = getDatabase();
  
  const stmt = db.prepare(`
    INSERT INTO gpus (gpu_id, name, brand, tflops, memory_gb, price_usd, launch_price_usd, architecture, release_date, tdp, is_active, raw_data)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  
  const result = stmt.run(
    gpu.gpu_id,
    gpu.name,
    gpu.brand || null,
    gpu.tflops || null,
    gpu.memory_gb || null,
    gpu.price_usd || null,
    gpu.launch_price_usd || null,
    gpu.architecture || null,
    gpu.release_date || null,
    gpu.tdp || null,
    gpu.is_active !== false ? 1 : 0,
    gpu.raw_data || null
  );
  
  return getGPUById(result.lastInsertRowid as number)!;
}

// 批量插入 GPU（使用事务）
export function insertGPUs(gpus: GPUInsert[]): number {
  const db = getDatabase();
  
  const stmt = db.prepare(`
    INSERT OR REPLACE INTO gpus (gpu_id, name, brand, tflops, memory_gb, price_usd, launch_price_usd, architecture, release_date, tdp, is_active, raw_data)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  
  const insertMany = db.transaction((items: GPUInsert[]) => {
    for (const gpu of items) {
      stmt.run(
        gpu.gpu_id,
        gpu.name,
        gpu.brand || null,
        gpu.tflops || null,
        gpu.memory_gb || null,
        gpu.price_usd || null,
        gpu.launch_price_usd || null,
        gpu.architecture || null,
        gpu.release_date || null,
        gpu.tdp || null,
        gpu.is_active !== false ? 1 : 0,
        gpu.raw_data || null
      );
    }
  });
  
  insertMany(gpus);
  
  return gpus.length;
}

// 更新 GPU
export function updateGPU(id: number, updates: Partial<GPUInsert>): GPURecord | null {
  const db = getDatabase();
  
  const fields: string[] = [];
  const values: (string | number | null)[] = [];
  
  for (const [key, value] of Object.entries(updates)) {
    if (key === 'is_active') {
      fields.push('is_active = ?');
      values.push(value ? 1 : 0);
    } else if (key !== 'gpu_id') {
      fields.push(`${key} = ?`);
      values.push(value as string | number | null);
    }
  }
  
  if (fields.length === 0) return getGPUById(id);
  
  fields.push("updated_at = datetime('now')");
  values.push(id);
  
  const stmt = db.prepare(`UPDATE gpus SET ${fields.join(', ')} WHERE id = ?`);
  stmt.run(...values);
  
  return getGPUById(id);
}

// 删除所有 GPU
export function deleteAllGPUs(): void {
  const db = getDatabase();
  db.exec('DELETE FROM gpus');
}

// 价格历史
export interface GPUPriceRecord {
  id: number;
  gpu_id: string;
  price_usd: number;
  source: string | null;
  recorded_at: string;
}

export function getGPUPriceHistory(gpuId: string): GPUPriceRecord[] {
  const db = getDatabase();
  const stmt = db.prepare(`
    SELECT * FROM gpu_prices 
    WHERE gpu_id = ? 
    ORDER BY recorded_at DESC
  `);
  return stmt.all(gpuId) as GPUPriceRecord[];
}

export function addGPUPrice(gpuId: string, priceUsd: number, source?: string): GPUPriceRecord {
  const db = getDatabase();
  
  const stmt = db.prepare(`
    INSERT INTO gpu_prices (gpu_id, price_usd, source)
    VALUES (?, ?, ?)
  `);
  
  const result = stmt.run(gpuId, priceUsd, source || null);
  
  return {
    id: result.lastInsertRowid as number,
    gpu_id: gpuId,
    price_usd: priceUsd,
    source: source || null,
    recorded_at: new Date().toISOString(),
  };
}
