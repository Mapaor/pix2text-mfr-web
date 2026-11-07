# pix2text-mfr-web

Client-side Math Formula Recognition (MFR) for Pix2Text MFR-1.5, using
[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html).

## Status

The package exposes a small TypeScript API for the exported Pix2Text
encoder-decoder model.

The downloaded MFR-1.5 graphs have these interfaces:

```text
encoder: pixel_values [1, 3, 384, 384] -> last_hidden_state [1, 578, 384]
decoder: input_ids [1, sequence] +
         encoder_hidden_states [1, 578, 384] -> logits [1, sequence, 1868]
```

The model files are not included in this repository. Download them from the
[Pix2Text MFR-1.5 model repository](https://huggingface.co/breezedeus/pix2text-mfr-1.5).

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

Model weights are runtime assets and must not be committed to this repository.
Download the model directly from Hugging Face:

```bash
source /opt/pix2text/venv/bin/activate
npm run download:model
```

The script downloads the default model to `models/pix2text-mfr-1.5/`. You can
override the repository and destination:

```bash
./scripts/download-model.sh breezedeus/pix2text-mfr-1.5 ./models/mfr-1.5
```

## Install

```bash
npm install
npm run build
```

## Demo

The repository includes a small Vite demo. It is committed for development
and validation but is excluded from the published npm package.

```bash
npm run download:model
npm run demo
```

The demo listens on all network interfaces at port `5173`. Open
`http://localhost:5173/` locally, or use the machine's Tailscale hostname,
for example `http://hp-server.cod-owl.ts.net:5173/`. Select a formula image
and click **Recognize**. The demo serves the downloaded model and ONNX
Runtime's WASM binaries as static files and runs recognition in the browser.
For a production demo build:

```bash
npm run demo:build
```

## Usage

```ts
import { MfrRecognizer } from "pix2text-mfr-web";

const recognizer = await MfrRecognizer.load({
  modelBaseUrl: "https://cdn.example.com/pix2text-mfr-1.5",
});

const result = await recognizer.recognize(imageBlob);
console.log(result.text);
```

`modelBaseUrl` must point to a directory containing the files listed above.
For local development, serve the repository directory over HTTP rather than
opening an HTML file directly:

```bash
npx serve .
```

Vite's `publicDir` serves the model directory at the site root, so the demo
uses `/pix2text-mfr-1.5` as the model URL.

## Publishing the model to a CDN

The downloaded directory is already the complete static asset bundle. Upload
its contents to a public CDN or object-storage prefix, preserving the file
names:

```text
https://cdn.example.com/pix2text/mfr-1.5/
  encoder_model.onnx
  decoder_model.onnx
  tokenizer.json
  tokenizer_config.json
  config.json
  generation_config.json
  preprocessor_config.json
  special_tokens_map.json
```

The CDN must allow browser `GET` requests from the application origin
(`Access-Control-Allow-Origin`) and must serve `.onnx` files as binary data.
The application does not upload images or send inference requests to a
server: model downloads and recognition happen in the user's browser.

```ts
const recognizer = await MfrRecognizer.load({
  modelBaseUrl: "https://cdn.example.com/pix2text/mfr-1.5",
  executionProviders: ["wasm"],
});

const file = document.querySelector<HTMLInputElement>("#formula")!.files![0];
const { text } = await recognizer.recognize(file);
```

The first load fetches the model and tokenizer assets. ONNX Runtime and the
tokenizer can then use browser caching; applications should reuse the loaded
recognizer instead of loading it for every image. Pinning a versioned CDN path
such as `/mfr/1.5.0/` prevents model updates from changing existing clients.

The library also stores the encoder and decoder in the browser's persistent
Cache Storage. After the first successful load, closing and reopening the page
reuses the cached model bytes instead of downloading them again. Tokenizer
assets use Transformers.js browser caching as well. Use a versioned
`modelBaseUrl` when publishing a new model:

```ts
const recognizer = await MfrRecognizer.load({
  modelBaseUrl: "https://cdn.example.com/pix2text/mfr/1.5.0",
  cacheName: "pix2text-mfr-1.5.0",
});
```

To force a fresh download during development:

```ts
await MfrRecognizer.clearCache("pix2text-mfr-1.5.0");
```

The package uses the WASM execution provider:

```ts
const recognizer = await MfrRecognizer.load({
  modelBaseUrl: "/models/pix2text-mfr-1.5",
  executionProviders: ["wasm"],
});
```

Images are converted to RGB, resized to 384×384, and normalized with mean and
standard deviation `[0.5, 0.5, 0.5]`. Decoding starts with token `1`, stops at
EOS token `2`, and defaults to a maximum of 1024 generated tokens.

## Limitations

- Decoding is greedy; beam search and confidence scores are not implemented.
- Post-processing is not yet ported from Pix2Text's `latex_ocr.py`.
- The package currently loads one encoder and one decoder session per recognizer.

## Verifying a model deployment

Before publishing a model directory, verify all of the following:

1. The model directory contains the eight required files listed above.
2. `encoder_model.onnx` and `decoder_model.onnx` are binary files, not Git LFS
   pointer text files.
3. The CDN serves every file with successful `GET` requests and allows CORS
   from the application origin.
4. A browser smoke test loads one formula image and returns non-empty LaTeX.
5. The browser result matches a server-side reference run using the same image.

For a quick local smoke test, download a formula example, serve the web
project over HTTP, and run the usage example from a browser. Do not open the
HTML file with `file://`; browser model and WASM requests require HTTP.

The server-side reference can be run with the Pix2Text virtual environment:

```bash
source /opt/pix2text/venv/bin/activate
python - <<'PY'
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import TrOCRProcessor

model = "models/pix2text-mfr-1.5"
image = "../pix2text-server/Pix2Text/docs/examples/formula.jpg"
processor = TrOCRProcessor.from_pretrained(model, use_fast=False)
pixels = processor(
    images=Image.open(image).convert("RGB"), return_tensors="np"
).pixel_values.astype("float32")
hidden = ort.InferenceSession(
    f"{model}/encoder_model.onnx"
).run(None, {"pixel_values": pixels})[0]
decoder = ort.InferenceSession(f"{model}/decoder_model.onnx")
tokens = [1]
for _ in range(1024):
    logits = decoder.run(None, {
        "input_ids": np.asarray([tokens], dtype=np.int64),
        "encoder_hidden_states": hidden,
    })[0]
    next_token = int(np.argmax(logits[0, -1]))
    if next_token == 2:
        break
    tokens.append(next_token)
print(processor.tokenizer.decode(tokens, skip_special_tokens=True))
PY
```

The exact text is expected to be stable for a pinned model revision. Minor
differences between browser and server runtimes should be investigated by
comparing token IDs first, then decoded text.
