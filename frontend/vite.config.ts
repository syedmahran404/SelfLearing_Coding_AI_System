import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API base is read at runtime via VITE_API_BASE_URL (see src/api/client.ts).
// In dev we proxy /api to the backend so the frontend can run on its own port.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "0.0.0.0",
    proxy: {
      "/api": {
        target: process.env.VITE_API_BASE_URL ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
