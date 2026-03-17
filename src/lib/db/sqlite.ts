import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';

// 数据库文件路径
const DATA_DIR = path.join(process.cwd(), 'data');
const DB_PATH = path.join(DATA_DIR, 'scaling_law.db');

// 确保数据目录存在
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// 创建数据库连接（单例）
let db: Database.Database | null = null;

export function getDatabase(): Database.Database {
  // 检查数据库文件是否存在，如果不存在则重置连接
  if (db && !fs.existsSync(DB_PATH)) {
    try {
      db.close();
    } catch {
      // 忽略关闭错误
    }
    db = null;
  }
  
  if (!db) {
    db = new Database(DB_PATH);
    // 启用外键约束
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');
  }
  return db;
}

// 初始化数据库表
export function initDatabase(): void {
  const db = getDatabase();
  
  // GPU表
  db.exec(`
    CREATE TABLE IF NOT EXISTS gpus (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      gpu_id TEXT UNIQUE NOT NULL,
      name TEXT NOT NULL,
      brand TEXT,
      tflops REAL,
      memory_gb REAL,
      price_usd REAL,
      launch_price_usd REAL,
      architecture TEXT,
      release_date TEXT,
      tdp INTEGER,
      is_active INTEGER DEFAULT 1,
      raw_data TEXT,
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now'))
    );
    
    CREATE INDEX IF NOT EXISTS idx_gpus_gpu_id ON gpus(gpu_id);
    CREATE INDEX IF NOT EXISTS idx_gpus_is_active ON gpus(is_active);
    CREATE INDEX IF NOT EXISTS idx_gpus_brand ON gpus(brand);
  `);
  
  // GPU价格历史表
  db.exec(`
    CREATE TABLE IF NOT EXISTS gpu_prices (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      gpu_id TEXT NOT NULL,
      price_usd REAL NOT NULL,
      source TEXT,
      recorded_at TEXT DEFAULT (datetime('now')),
      FOREIGN KEY (gpu_id) REFERENCES gpus(gpu_id) ON DELETE CASCADE
    );
    
    CREATE INDEX IF NOT EXISTS idx_gpu_prices_gpu_id ON gpu_prices(gpu_id);
    CREATE INDEX IF NOT EXISTS idx_gpu_prices_recorded_at ON gpu_prices(recorded_at);
  `);
  
  // 数据集表
  db.exec(`
    CREATE TABLE IF NOT EXISTS datasets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      description TEXT,
      tokens_count INTEGER,
      language TEXT,
      source_url TEXT,
      license TEXT,
      is_recommended INTEGER DEFAULT 0,
      created_at TEXT DEFAULT (datetime('now')),
      updated_at TEXT DEFAULT (datetime('now'))
    );
  `);
  
  // 使用统计表
  db.exec(`
    CREATE TABLE IF NOT EXISTS usage_stats (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      action TEXT NOT NULL,
      params_n REAL,
      params_model_type TEXT,
      params_sparsity REAL,
      params_tokens REAL,
      params_gpu_count INTEGER,
      params_utilization REAL,
      result_loss REAL,
      result_time_days REAL,
      result_flops REAL,
      selected_gpu_ids TEXT,
      created_at TEXT DEFAULT (datetime('now'))
    );
    
    CREATE INDEX IF NOT EXISTS idx_usage_stats_action ON usage_stats(action);
    CREATE INDEX IF NOT EXISTS idx_usage_stats_created_at ON usage_stats(created_at);
  `);
  
  // 系统配置表
  db.exec(`
    CREATE TABLE IF NOT EXISTS system_config (
      key TEXT PRIMARY KEY,
      value TEXT,
      description TEXT,
      updated_at TEXT DEFAULT (datetime('now'))
    );
  `);
  
  console.log('Database initialized successfully');
}

// 关闭数据库连接
export function closeDatabase(): void {
  if (db) {
    db.close();
    db = null;
  }
}

// 重置数据库（用于测试）
export function resetDatabase(): void {
  closeDatabase();
  if (fs.existsSync(DB_PATH)) {
    fs.unlinkSync(DB_PATH);
  }
  // 同时删除 WAL 文件
  const walPath = DB_PATH + '-wal';
  const shmPath = DB_PATH + '-shm';
  if (fs.existsSync(walPath)) fs.unlinkSync(walPath);
  if (fs.existsSync(shmPath)) fs.unlinkSync(shmPath);
  
  initDatabase();
}
