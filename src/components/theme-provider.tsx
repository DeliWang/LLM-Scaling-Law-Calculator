'use client';

import { createContext, useContext, useEffect, useState, useCallback } from 'react';

type Theme = 'light' | 'dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// 根据时间判断是否应该使用暗色主题
function shouldUseDarkByTime(): boolean {
  const hour = new Date().getHours();
  // 晚上7点到早上7点使用暗色主题
  return hour >= 19 || hour < 7;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('dark');
  const [mounted, setMounted] = useState(false);

  // 初始化主题 - 只根据时间判断，不读取 localStorage
  useEffect(() => {
    setTheme(shouldUseDarkByTime() ? 'dark' : 'light');
    setMounted(true);
  }, []);

  // 应用主题到 DOM - 不保存到 localStorage
  useEffect(() => {
    if (!mounted) return;

    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
    } else {
      root.classList.remove('dark');
      root.classList.add('light');
    }
    // 不再保存到 localStorage
  }, [theme, mounted]);

  // 切换主题
  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
