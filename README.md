# pix2text-mfr-web

Client-side Math Formula Recognition (MFR) for Pix2Text MFR-1.5, using
[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html).

## Install

```bash
npm install pix2text-mfr-web
```

The npm package exposes a small TypeScript API for the Pix2Text MFR-1.5
encoder-decoder model. So this is npm package is for the API, not the model itself. The model and tokenizer assets are hosted on HuggingFace.

## Quick start

```ts
import { MfrRecognizer } from "pix2text-mfr-web";

const recognizer = await MfrRecognizer.load({
  modelBaseUrl:
    "https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/main",
});

const result = await recognizer.recognize(imageBlob);
console.log(result.text);
```

Note: `imageBlob` can be a browser `Blob`, `ImageData`, or a Transformers.js
`RawImage`.

## API

### `MfrRecognizer.load(options)`

Loads the ONNX encoder, decoder, and tokenizer assets.

```ts
interface MfrLoadOptions {
  modelBaseUrl: string;
  executionProviders?: ExecutionProviderConfig[];
  wasmPaths?: string | { mjs: string; wasm: string };
  cacheName?: string;
}
```

- `modelBaseUrl` is the directory containing the model assets.
- `executionProviders` defaults to `["webgpu", "wasm"]`, preferring WebGPU and
  falling back to WASM.
- `wasmPaths` optionally overrides ONNX Runtime's WASM module and binary URLs.
- `cacheName` sets the name for the browser Cache Storage.

The model directory must contain:

```text
encoder_model.onnx
decoder_model.onnx
config.json
generation_config.json
preprocessor_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```

### `recognizer.recognize(image, options?)`

Recognizes one formula image.

```ts
interface RecognizeOptions {
  maxNewTokens?: number;
}

interface RecognitionResult {
  text: string;
  tokenIds: number[];
}
```

The default maximum is 1024 generated tokens. Decoding is greedy and stops
when the EOS token is produced.

### `MfrRecognizer.clearCache(cacheName?)`

Deletes the model cache.

```ts
await MfrRecognizer.clearCache("pix2text-mfr-1.5");
```

## Hosting the model

The original Pix2Text MFR-1.5 ONNX files work with ONNX Runtime Web without any
browser-specific conversion needed so one can use the original model directory of
Hugging Face, a CDN, or a custom object storage. I also intend to publish a quantized version of the model to HF as well.

If you are on linux, I recommend using a virtual environment on `/opt/pix2text/venv`. Then the Hugging Face model can be downloaded locally with:

```bash
source /opt/pix2text/venv/bin/activate
npm run download:model
```

For production pin a model revision instead of main:

```ts
const recognizer = await MfrRecognizer.load({
  modelBaseUrl:
    "https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/<revision>",
  cacheName: "pix2text-mfr-1.5-<revision>",
});
```

Notes for it to work:

The host must serve `.onnx` files as binary responses, needs to handle CORS to allow `GET` requests and serve the .json and .onnx files we listed before.

## Caching

- The encoder and decoder are stored in the browser's Cache Storage.
- Transformers.js also uses browser Cache Storage. 

## Notes on current state

- The default execution providers are WebGPU followed by WASM fallback.
- WebGPU is selected by ONNX Runtime Web at runtime; the ONNX model files do
  not need a separate WebGPU conversion.
- Images are converted to RGB, resized to 384×384, and normalized with mean
  and standard deviation `[0.5, 0.5, 0.5]` (just like pix2text does).
- Currently no confidence scores or post-processing are implemented.

## Demo

The repository includes a Vite demo under `demo/` useful for testing in development. The demo is excluded from the published npm package.

```bash
npm install
npm run download:model
npm run demo
```

The local demo uses `/pix2text-mfr-1.5` (local model that needs to be manually downloaded as explained before). In production set `VITE_MFR_MODEL_BASE_URL` to a public model URL, if not set it defaults to https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/main.

You can check a live version of the demo here: [pix2text-mfr-web.vercel.app](https://pix2text-mfr-web.vercel.app/).

## License

[MIT](/LICENSE)
