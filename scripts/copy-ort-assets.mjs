import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ortDist = resolve(root, "node_modules/onnxruntime-web/dist");
const assetsDir = resolve(root, "dist/assets");
const assets = [
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
];

await mkdir(assetsDir, { recursive: true });
for (const asset of assets) {
  await copyFile(resolve(ortDist, asset), resolve(assetsDir, asset));
}
