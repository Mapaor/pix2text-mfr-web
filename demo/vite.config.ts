import { defineConfig } from "vite";
import fs from "node:fs";
import path from "node:path";

function preparePublicAssets(): void {
  const publicDir = path.resolve("demo/public");
  const ortDir = path.join(publicDir, "ort");
  fs.mkdirSync(ortDir, { recursive: true });

  const ortPackageDir = path.resolve("node_modules/onnxruntime-web/dist");
  for (const file of fs.readdirSync(ortPackageDir)) {
    if (file.startsWith("ort-wasm") && file.endsWith(".wasm")) {
      fs.copyFileSync(path.join(ortPackageDir, file), path.join(ortDir, file));
    }
  }
  for (const file of fs.readdirSync(ortDir)) {
    if (file.startsWith("ort-wasm") && file.endsWith(".mjs")) {
      fs.unlinkSync(path.join(ortDir, file));
    }
  }

  const modelDir = path.resolve("models");
  const publicModelDir = path.join(publicDir, "pix2text-mfr-1.5");
  if (fs.existsSync(modelDir) && !fs.existsSync(publicModelDir)) {
    fs.symlinkSync(path.join(modelDir, "pix2text-mfr-1.5"), publicModelDir, "dir");
  }
}

preparePublicAssets();

export default defineConfig({
  root: "demo",
  publicDir: "public",
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    allowedHosts: ["hp-server.cod-owl.ts.net"], // Add your allowed host here (in my case my Tailscale host.net domain)
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
