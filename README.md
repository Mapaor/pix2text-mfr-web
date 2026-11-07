# pix2text-mfr-web

The goal of this repo is to port Pix2Text MFR-1.5 open source model for Math Formula Recognition to ONNX Runtime for Web, so it can be used in client-side websites.

## Model details

The model is already ONNX, it's a TrOCR-style encoder-decoder:

Vision encoder: DeiT, 12 layers, 384 hidden size
Text decoder: 6-layer Transformer decoder, 256 hidden size
Input: RGB image resized to 384×384, normalized with mean/std 0.5
Output: LaTeX tokens, vocabulary size 1868
Maximum output: 1024 tokens
Graphs: separate encoder_model.onnx and decoder_model.onnx

## Portability plan

Load the encoder and decoder with onnxruntime-web.
Implement the 384×384 image preprocessing.
Run the encoder once.
Implement autoregressive decoding:
start token 1
repeatedly run the decoder
select the next token
stop at EOS token 2
Decode tokens using the supplied XLM-R tokenizer assets
Reproduce Pix2Text’s LaTeX cleanup functions from latex_ocr.py

ONNX Runtime Web does not tokenize text, we'll need Transformers.js or a custom tokenizer for that.

WebGPU should be intended, as it will probably be faster, but we'll implement it later on.

The model could be cached with IndexedDB or similar.

First we'll go with the original FP32  graphs and then consider FP16 or IINT8 quantization.

## Desgin
We should create a Typescript API that uses onnxruntime-web, which initialized internally a WASM backend that uses encoder_model.onxx + decoder_model.onnx

onnxruntime-web already provides the WASM runtime.

We should put our source code inside src/ (example recognizer.ts, encoder-decoder.ts, image-preproceessor.ts, tokenized.ts, postprocess.ts, etc.) and the model inside models/pix2text-mfr-1.5/

We should probably also create a demo/.

We should make the Model URL configurable (if people want to use their own CDN).

```js
await MfrRecognizer.load({
  modelBaseUrl: "https://cdn.example.com/pix2text-mfr-1.5",
});
```
