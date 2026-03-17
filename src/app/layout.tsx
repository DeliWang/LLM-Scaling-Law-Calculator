import type { Metadata } from 'next';
import { Inspector } from 'react-dev-inspector';
import { ThemeProvider } from '@/components/theme-provider';
import './globals.css';

export const metadata: Metadata = {
  title: {
    default: 'LLM Scaling Calculator',
    template: '%s | 大模型缩放律计算器',
  },
  description:
    '基于论文 arXiv:2508.06617 的大模型缩放律计算器，智能推算训练数据量、算力需求与训练时长',
  keywords: [
    'LLM',
    'Scaling Law',
    '大模型',
    '缩放律',
    '计算器',
    'GPU',
    '训练时间',
    '算力估算',
  ],
  authors: [{ name: 'LLM Scaling Calculator Team' }],
  openGraph: {
    title: 'LLM Scaling Calculator | 大模型缩放律计算器',
    description:
      '基于论文的大模型缩放律计算器，输入模型参数量与训练目标，智能推算数据量、算力需求与训练时长',
    type: 'website',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const isDev = process.env.NODE_ENV === 'development';

  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className={`antialiased`}>
        <ThemeProvider>
          {isDev && <Inspector />}
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
