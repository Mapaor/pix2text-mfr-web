import { MfrRecognizer } from "../src/index";
import wasmModuleUrl from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url";
import "./style.css";

const imageInput = document.querySelector<HTMLInputElement>("#formula-image");
const recognizeButton = document.querySelector<HTMLButtonElement>("#recognize");
const copyButton = document.querySelector<HTMLButtonElement>("#copy-result");
const webgpuButton = document.querySelector<HTMLButtonElement>("#webgpu");
const wasmButton = document.querySelector<HTMLButtonElement>("#wasm");
const backend = document.querySelector<HTMLParagraphElement>("#backend");
const status = document.querySelector<HTMLParagraphElement>("#status");
const result = document.querySelector<HTMLPreElement>("#result");

if (!imageInput || !recognizeButton || !copyButton || !webgpuButton || !wasmButton || !backend || !status || !result) {
  throw new Error("Demo markup is incomplete.");
}

let recognizer: MfrRecognizer | undefined;
let selectedBackend: "webgpu" | "wasm" = "wasm";
let backendWasSelected = false;
const modelBaseUrl = import.meta.env.VITE_MFR_MODEL_BASE_URL
  ?? (import.meta.env.DEV
    ? "/pix2text-mfr-1.5"
    : "https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/main");
const backendReady = navigator.gpu?.requestAdapter().then((adapter) => Boolean(adapter)) ?? Promise.resolve(false);
backendReady.then((available) => {
  webgpuButton.disabled = !available;
  if (available && !backendWasSelected) selectBackend("webgpu");
});

function selectBackend(backendName: "webgpu" | "wasm"): void {
  selectedBackend = backendName;
  webgpuButton.classList.toggle("selected", backendName === "webgpu");
  wasmButton.classList.toggle("selected", backendName === "wasm");
  recognizer = undefined;
}

webgpuButton.addEventListener("click", () => {
  backendWasSelected = true;
  selectBackend("webgpu");
});
wasmButton.addEventListener("click", () => {
  backendWasSelected = true;
  selectBackend("wasm");
});

imageInput.addEventListener("change", () => {
  recognizeButton.disabled = !imageInput.files?.length;
  result.textContent = "";
  status.textContent = imageInput.files?.length
    ? "Ready to recognize."
    : "Choose an image to begin.";
});

document.addEventListener("paste", (event) => {
  const image = Array.from(event.clipboardData?.items ?? [])
    .find((item) => item.type.startsWith("image/"))
    ?.getAsFile();
  if (!image) return;

  const transfer = new DataTransfer();
  transfer.items.add(image);
  imageInput.files = transfer.files;
  recognizeButton.disabled = false;
  result.textContent = "";
  status.textContent = "Pasted image ready to recognize.";
});

recognizeButton.addEventListener("click", async () => {
  const image = imageInput.files?.[0];
  if (!image) return;

  recognizeButton.disabled = true;
  try {
    backend.hidden = false;
    if (selectedBackend === "webgpu" && !(await backendReady)) {
      throw new Error("WebGPU is not available in this browser.");
    }
    const executionProviders = [selectedBackend];
    backend.textContent = `Backend: ${selectedBackend === "webgpu" ? "WebGPU" : "WASM"}`;
    if (!recognizer) {
      status.textContent = "Loading model...";
      recognizer = await MfrRecognizer.load({
        modelBaseUrl,
        wasmPaths: {
          mjs: wasmModuleUrl,
          wasm: "/ort/ort-wasm-simd-threaded.jsep.wasm",
        },
        executionProviders,
        cacheName: "pix2text-mfr-1.5",
      });
    }

    status.textContent = "Recognizing...";
    const startedAt = performance.now();
    const recognition = await recognizer.recognize(image);
    const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(2);
    result.textContent = recognition.text;
    copyButton.disabled = false;
    status.textContent = `Done (${recognition.tokenIds.length} tokens, ${elapsedSeconds}s).`;
  } catch (error) {
    status.textContent = "Recognition failed.";
    result.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    recognizeButton.disabled = false;
  }
});

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(result.textContent);
    copyButton.classList.add("copied");
    window.setTimeout(() => copyButton.classList.remove("copied"), 1000);
  } catch (error) {
    status.textContent = "Copy failed.";
    result.textContent = error instanceof Error ? error.message : String(error);
  }
});
