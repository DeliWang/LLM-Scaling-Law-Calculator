#!/usr/bin/env node
/**
 * 从 CSV 导入 GPU 数据到 SQLite 数据库
 * 用法: npx tsx scripts/seed-gpus.ts
 */

import path from 'path';

// 设置工作目录
process.chdir(path.join(__dirname, '..'));

// 动态导入模块
const { importGPUDataFromCSV, getImportStatus } = require('../src/lib/db/import-gpus');

console.log('=== GPU Data Seeding Script ===\n');

// 执行导入
const count = importGPUDataFromCSV();

// 显示结果
const status = getImportStatus();
console.log('\n=== Import Status ===');
console.log(`Total GPUs: ${status.count}`);
console.log(`Last Import: ${status.lastImport || 'Never'}`);

if (count > 0) {
  console.log(`\n✅ Successfully imported ${count} GPUs`);
} else if (status.count > 0) {
  console.log('\n✅ Database already contains GPU data');
} else {
  console.log('\n⚠️ No GPU data was imported');
  console.log('Please ensure data/gpu_1986-2026.csv exists');
}
