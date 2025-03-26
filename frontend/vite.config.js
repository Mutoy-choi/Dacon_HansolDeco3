import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,  // 3000번 포트 사용
    strictPort: true,
    host: "localhost",
  }
});
