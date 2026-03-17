'use client';

import { useState, useEffect, useRef } from 'react';

// GPU接口定义
export interface GPU {
  id: string;
  name: string;
  tfops: number;
  utilization: number;
  memory_gb: number;
  price_usd: number | null;
}

// 统计数据接口
export interface Stats {
  total_calculations: number;
  total_page_views: number;
  avg_model_params: number | null;
  total_training_hours: number;
}

// 全局缓存，防止热更新时重置
let cachedGPUs: GPU[] | null = null;
let cachedStats: Stats | null = null;

export function useGPUs() {
  const [gpus, setGpus] = useState<GPU[]>(() => {
    // 初始化时使用缓存
    return cachedGPUs || [];
  });
  const [loading, setLoading] = useState(!cachedGPUs);
  const loadingRef = useRef(false);

  useEffect(() => {
    // 已经有缓存或正在加载，跳过
    if (cachedGPUs || loadingRef.current) {
      setLoading(false);
      return;
    }

    loadingRef.current = true;

    fetch('/api/gpus?isActive=true')
      .then(res => res.json())
      .then(data => {
        if (data.success && data.data) {
          cachedGPUs = data.data;
          setGpus(data.data);
        }
      })
      .catch(err => {
        console.error('Failed to load GPUs:', err);
      })
      .finally(() => {
        setLoading(false);
        loadingRef.current = false;
      });
  }, []); // 空依赖，只在挂载时执行一次

  return {
    gpus,
    loading,
    reload: () => {
      cachedGPUs = null;
      loadingRef.current = false;
    },
  };
}

// 记录使用统计（带节流）- 使用全局变量持久化节流时间
let lastStatTime = 0;
const STAT_THROTTLE = 5000; // 5秒节流
let isRecording = false; // 防止并发请求

export async function recordStat(stat: {
  action: 'page_view' | 'calculation' | 'gpu_selection';
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
}): Promise<boolean> {
  const now = Date.now();
  
  // 节流检查
  if (now - lastStatTime < STAT_THROTTLE) {
    return true; // 节流中，静默跳过
  }
  
  // 防止并发
  if (isRecording) {
    return true;
  }
  
  lastStatTime = now;
  isRecording = true;

  try {
    await fetch('/api/stats', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(stat),
    });
    return true;
  } catch {
    return false;
  } finally {
    isRecording = false;
  }
}

// 获取使用统计
export function useStats() {
  const [stats, setStats] = useState<Stats>(() => {
    // 不使用缓存，每次都重新获取
    return { total_calculations: 0, total_page_views: 0, avg_model_params: null, total_training_hours: 0 };
  });
  const loadingRef = useRef(false);

  useEffect(() => {
    if (loadingRef.current) return;

    loadingRef.current = true;

    fetch('/api/stats')
      .then(res => res.json())
      .then(data => {
        cachedStats = data;
        setStats(data);
      })
      .catch(err => {
        console.error('Failed to load stats:', err);
      })
      .finally(() => {
        loadingRef.current = false;
      });
  }, []);

  return { stats };
}
