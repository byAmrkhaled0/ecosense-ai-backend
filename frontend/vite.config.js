import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const backendUrl = "https://ecosense-backend.vercel.app";
const sensorModelUrl = "https://amr2004-ecosense-ai.hf.space";
const combinedModelUrl = "https://Amrkhaled2004.pythonanywhere.com";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const useProxy = env.VITE_USE_DEV_PROXY === "true" || env.VITE_API_BASE_URL === "/api";
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 3000,
      proxy: useProxy
        ? {
            "/api": {
              target: backendUrl,
              changeOrigin: true,
              secure: true,
            },
            "/ai-model": {
              target: sensorModelUrl,
              changeOrigin: true,
              secure: true,
              rewrite: (path) => path.replace(/^\/ai-model/, ""),
            },
            "/ai-combined": {
              target: combinedModelUrl,
              changeOrigin: true,
              secure: true,
              rewrite: (path) => path.replace(/^\/ai-combined/, ""),
            },
          }
        : undefined,
    },
    build: {
      chunkSizeWarningLimit: 700,
      rollupOptions: {
        output: {
          manualChunks: {
            react: ["react", "react-dom", "react-router-dom"],
            charts: ["recharts"],
            vendor: ["axios", "lucide-react"],
          },
        },
      },
    },
  };
});
