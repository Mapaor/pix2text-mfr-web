import { MfrRecognizer } from "../src/index";
import wasmModuleUrl from "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url";
import "./style.css";

const imageInput = document.querySelector<HTMLInputElement>("#formula-image");
const recognizeButton = document.querySelector<HTMLButtonElement>("#recognize");
const status = document.querySelector<HTMLParagraphElement>("#status");
const result = document.querySelector<HTMLPreElement>("#result");

if (!imageInput || !recognizeButton || !status || !result) {
  throw new Error("Demo markup is incomplete.");
}

let recognizer: MfrRecognizer | undefined;

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
    if (!recognizer) {
      status.textContent = "Loading model...";
      recognizer = await MfrRecognizer.load({
        modelBaseUrl: "/pix2text-mfr-1.5",
        wasmPaths: {
          mjs: wasmModuleUrl,
          wasm: "/ort/ort-wasm-simd-threaded.wasm",
        },
        executionProviders: ["wasm"],
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
