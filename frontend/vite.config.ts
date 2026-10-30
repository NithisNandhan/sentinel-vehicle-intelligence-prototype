import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const apiTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8000';
const mediaTarget = process.env.VITE_MEDIA_PROXY_TARGET || 'http://localhost:8888';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    allowedHosts: ['localhost', '127.0.0.1'],
    port: 5173,
    proxy: {
      '/v1': {
        target: apiTarget,
        changeOrigin: false,
        ws: true,
      },
      '/media': {
        target: mediaTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/media/, ''),
      },
    },
  },
});
