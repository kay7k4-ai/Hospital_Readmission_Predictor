import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, every API path is forwarded to the FastAPI backend on port 8000.
const API = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/predict": API,
      "/health": API,
      "/model": API,
    },
  },
});
