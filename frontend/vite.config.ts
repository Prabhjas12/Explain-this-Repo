import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development (npm run dev) the UI runs on :5173 and forwards /api to the Python server on :8000.
// For normal use, `npm run build` writes dist/, which the Python server serves itself.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1200 },
});
