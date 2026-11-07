import { MfrRecognizer } from "../src/index";
import wasmModuleUrl from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url";
import "./style.css";

const imageInput = document.querySelector<HTMLInputElement>("#formula-image");
const recognizeButton = document.querySelector<HTMLButtonElement>("#recognize");
const backend = document.querySelector<HTMLParagraphElement>("#backend");
const status = document.querySelector<HTMLParagraphElement>("#status");
const result = document.querySelector<HTMLPreElement>("#result");

if (!imageInput || !recognizeButton || !backend || !status || !result) {
  throw new Error("Demo markup is incomplete.");
}

let recognizer: MfrRecognizer | undefined;
const modelBaseUrl = import.meta.env.VITE_MFR_MODEL_BASE_URL
  ?? (import.meta.env.DEV
    ? "/pix2text-mfr-1.5"
    : "https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/main");
const backendReady = navigator.gpu?.requestAdapter().then((adapter) => Boolean(adapter)) ?? Promise.resolve(false);

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
    const webgpuAvailable = await backendReady;
    const executionProviders = webgpuAvailable ? ["webgpu"] : ["wasm"];
    backend.textContent = `Backend: ${webgpuAvailable ? "WebGPU" : "WASM"}`;
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
    const recognition = await recognizer.recognize(image);
    result.textContent = recognition.text;
    status.textContent = `Done (${recognition.tokenIds.length} tokens).`;
  } catch (error) {
    status.textContent = "Recognition failed.";
    result.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    recognizeButton.disabled = false;
  }
});
