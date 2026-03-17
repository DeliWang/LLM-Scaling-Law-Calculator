import { initDatabase, getDatabase } from './sqlite';
import { getGPUCount, insertGPUs, type GPUInsert } from './sqlite-gpus';
import fs from 'fs';
import path from 'path';

// CSV 行解析（处理引号内的逗号）
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current.trim());
  
  return result;
}

// 解析显存大小（如 "8 GB" -> 8）
function parseMemorySize(value: string): number {
  if (!value) return 0;
  const match = value.match(/(\d+(?:\.\d+)?)/);
  return match ? parseFloat(match[1]) : 0;
}

// 解析价格（如 "$1,499" -> 1499）
function parsePrice(value: string): number | null {
  if (!value || value === 'unknown' || value === '') return null;
  const match = value.replace(/[$,]/g, '').match(/(\d+(?:\.\d+)?)/);
  return match ? parseFloat(match[1]) : null;
}

// 解析 TFLOPS（如 "82.58 TFLOPS" -> 82.58）
function parseTFLOPS(value: string): number {
  if (!value || value === 'unknown' || value === '') return 0;
  const match = value.match(/(\d+(?:\.\d+)?)/);
  return match ? parseFloat(match[1]) : 0;
}

// 解析 TDP（如 "350 W" -> 350）
function parseTDP(value: string): number | null {
  if (!value || value === 'unknown' || value === '') return null;
  const match = value.match(/(\d+)/);
  return match ? parseInt(match[1]) : null;
}

// 解析日期（如 "Mar 23rd, 1987" -> "1987-03-23"）
function parseDate(value: string): string | null {
  if (!value || value === 'unknown' || value === '') return null;
  try {
    // 处理 "Mar 23rd, 1987" 格式
    const cleaned = value.replace(/(\d+)(st|nd|rd|th)/, '$1');
    const date = new Date(cleaned);
    if (isNaN(date.getTime())) return null;
    return date.toISOString().split('T')[0];
  } catch {
    return null;
  }
}

// 生成唯一 GPU ID
function generateGPUId(brand: string, name: string): string {
  const cleanBrand = brand.toLowerCase().replace(/[^a-z0-9]/g, '_');
  const cleanName = name.toLowerCase().replace(/[^a-z0-9]/g, '_');
  return `${cleanBrand}_${cleanName}`;
}

// 从 CSV 导入 GPU 数据
export function importGPUDataFromCSV(): number {
  // 初始化数据库
  initDatabase();
  
  // 检查是否已有数据
  const existingCount = getGPUCount();
  if (existingCount > 0) {
    console.log(`Database already has ${existingCount} GPUs, skipping import`);
    return existingCount;
  }
  
  // 读取 CSV 文件
  const csvPath = path.join(process.cwd(), 'data/gpu_1986-2026.csv');
  
  if (!fs.existsSync(csvPath)) {
    console.error('GPU CSV file not found:', csvPath);
    console.warn('Please ensure data/gpu_1986-2026.csv exists');
    return 0;
  }
  
  const csvContent = fs.readFileSync(csvPath, 'utf-8');
  const lines = csvContent.split('\n');
  
  if (lines.length < 2) {
    console.error('CSV file is empty or has no data rows');
    return 0;
  }
  
  // 解析表头
  const headers = parseCSVLine(lines[0]);
  const headerMap: Record<string, number> = {};
  headers.forEach((header, index) => {
    headerMap[header] = index;
  });
  
  // 需要的字段
  const requiredFields = [
    'Brand',
    'Name',
    'Theoretical Performance__FP32 (float)',
    'Memory__Memory Size',
    'Graphics Card__Launch Price',
    'Graphics Processor__Architecture',
    'Graphics Card__Release Date',
    'Board Design__TDP',
  ];
  
  // 检查必需字段
  for (const field of requiredFields) {
    if (headerMap[field] === undefined) {
      console.error(`Missing required field in CSV: ${field}`);
      return 0;
    }
  }
  
  // 解析数据行
  const gpuRecords: GPUInsert[] = [];
  const seenIds = new Set<string>();
  
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    try {
      const values = parseCSVLine(line);
      
      const brand = values[headerMap['Brand']] || '';
      const name = values[headerMap['Name']] || '';
      
      if (!brand || !name) continue;
      
      const gpuId = generateGPUId(brand, name);
      
      // 去重
      if (seenIds.has(gpuId)) continue;
      seenIds.add(gpuId);
      
      const tflops = parseTFLOPS(values[headerMap['Theoretical Performance__FP32 (float)']]);
      const memoryGb = parseMemorySize(values[headerMap['Memory__Memory Size']]);
      const priceUsd = parsePrice(values[headerMap['Graphics Card__Launch Price']]);
      const architecture = values[headerMap['Graphics Processor__Architecture']] || null;
      const releaseDate = parseDate(values[headerMap['Graphics Card__Release Date']]);
      const tdp = parseTDP(values[headerMap['Board Design__TDP']]);
      
      // 只保留有算力数据的 GPU
      if (tflops <= 0) continue;
      
      gpuRecords.push({
        gpu_id: gpuId,
        name: `${brand} ${name}`,
        brand: brand,
        tflops: tflops,
        memory_gb: memoryGb,
        price_usd: priceUsd,
        launch_price_usd: priceUsd,
        architecture: architecture,
        release_date: releaseDate,
        tdp: tdp,
        is_active: tflops > 0 && memoryGb > 0,
      });
    } catch (error) {
      // 跳过解析错误的行
      console.warn(`Failed to parse line ${i + 1}: ${error}`);
    }
  }
  
  // 批量插入
  if (gpuRecords.length > 0) {
    const inserted = insertGPUs(gpuRecords);
    console.log(`Imported ${inserted} GPUs from CSV to SQLite database`);
    
    // 更新系统配置
    const db = getDatabase();
    const stmt = db.prepare(`
      INSERT OR REPLACE INTO system_config (key, value, description)
      VALUES ('last_import', ?, 'Last GPU data import timestamp')
    `);
    stmt.run(new Date().toISOString());
    
    return inserted;
  }
  
  return 0;
}

// 获取导入状态
export function getImportStatus(): { lastImport: string | null; count: number } {
  const db = getDatabase();
  
  const configStmt = db.prepare("SELECT value FROM system_config WHERE key = 'last_import'");
  const config = configStmt.get() as { value: string } | undefined;
  
  return {
    lastImport: config?.value || null,
    count: getGPUCount(),
  };
}
