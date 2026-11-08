import * as ort from "onnxruntime-web";
import { AutoTokenizer, env, RawImage } from "@huggingface/transformers";

export interface MfrLoadOptions {
  modelBaseUrl: string;
  executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
  wasmPaths?: string | { mjs: string; wasm: string };
  cacheName?: string;
}

export interface RecognizeOptions {
  maxNewTokens?: number;
}

export interface RecognitionResult {
  text: string;
  tokenIds: number[];
}

const DEFAULT_MAX_NEW_TOKENS = 1024;
const BOS_TOKEN_ID = 1;
const EOS_TOKEN_ID = 2;
const DEFAULT_CACHE_NAME = "pix2text-mfr-web";
const DEFAULT_WASM_PATHS = {
  mjs: new URL("./assets/ort-wasm-simd-threaded.jsep.mjs", import.meta.url).href,
  wasm: new URL("./assets/ort-wasm-simd-threaded.jsep.wasm", import.meta.url).href,
};

function url(base: string, file: string): string {
  return `${base.replace(/\/+$/, "")}/${file}`;
}

async function imageToTensor(image: Blob | ImageData | RawImage): Promise<ort.Tensor> {
  const source = image instanceof RawImage
    ? image
    : image instanceof ImageData
      ? new RawImage(image.data, image.width, image.height, 4)
      : await RawImage.fromBlob(image);
  const resized = (await source.resize(384, 384)).rgb();
  const pixels = resized.data;
  const tensor = new Float32Array(3 * 384 * 384);

  for (let i = 0; i < 384 * 384; i += 1) {
    tensor[i] = (pixels[i * 3] / 255 - 0.5) / 0.5;
    tensor[384 * 384 + i] = (pixels[i * 3 + 1] / 255 - 0.5) / 0.5;
    tensor[2 * 384 * 384 + i] = (pixels[i * 3 + 2] / 255 - 0.5) / 0.5;
  }

  return new ort.Tensor("float32", tensor, [1, 3, 384, 384]);
}

export class MfrRecognizer {
  private constructor(
    private readonly encoder: ort.InferenceSession,
    private readonly decoder: ort.InferenceSession,
    private readonly tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>,
  ) {}

  static async load(options: MfrLoadOptions): Promise<MfrRecognizer> {
    ort.env.wasm.wasmPaths = options.wasmPaths ?? DEFAULT_WASM_PATHS;
    env.useBrowserCache = true;
    const providers = options.executionProviders ?? ["webgpu", "wasm"];
    const base = options.modelBaseUrl;
    const cacheName = options.cacheName ?? DEFAULT_CACHE_NAME;
    const normalizedBase = base.replace(/\/+$/, "");
    const separator = normalizedBase.lastIndexOf("/");
    const tokenizerModelId = normalizedBase.slice(separator + 1) || "mfr-1.5";
    env.remoteHost = `${normalizedBase.slice(0, separator + 1)}`;
    env.remotePathTemplate = "{model}";
    const encoderUrl = url(base, "encoder_model.onnx");
    const decoderUrl = url(base, "decoder_model.onnx");
    const [encoder, decoder, tokenizer] = await Promise.all([
      loadSession(encoderUrl, cacheName, providers),
      loadSession(decoderUrl, cacheName, providers),
      AutoTokenizer.from_pretrained(tokenizerModelId),
    ]);
    return new MfrRecognizer(encoder, decoder, tokenizer);
  }

  static async clearCache(cacheName = DEFAULT_CACHE_NAME): Promise<boolean> {
    if (typeof caches === "undefined") return false;
    return caches.delete(cacheName);
  }

  async recognize(image: Blob | ImageData | RawImage, options: RecognizeOptions = {}): Promise<RecognitionResult> {
    const pixelValues = await imageToTensor(image);
    const encoderInput = this.encoder.inputNames[0];
    const encoderOutput = await this.encoder.run({ [encoderInput]: pixelValues });
    const encoderHiddenStates = encoderOutput[this.encoder.outputNames[0]];
    const tokenIds = [BOS_TOKEN_ID];
    const maxNewTokens = options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS;

    for (let step = 0; step < maxNewTokens; step += 1) {
      const inputIds = new ort.Tensor("int64", BigInt64Array.from(tokenIds, BigInt), [1, tokenIds.length]);
      const decoderInputs: Record<string, ort.Tensor> = {};
      for (const name of this.decoder.inputNames) {
        if (name === "input_ids") decoderInputs[name] = inputIds;
        else if (name === "encoder_hidden_states") decoderInputs[name] = encoderHiddenStates;
        else if (name === "attention_mask") {
          decoderInputs[name] = new ort.Tensor("int64", BigInt64Array.from(tokenIds, () => 1n), [1, tokenIds.length]);
        }

      }

      const output = await this.decoder.run(decoderInputs);
      const logits = output[this.decoder.outputNames[0]];
      const nextToken = argmaxLastToken(logits);
      if (nextToken === EOS_TOKEN_ID) break;
      tokenIds.push(nextToken);
    }

    return { text: this.tokenizer.decode(tokenIds, { skip_special_tokens: true }), tokenIds };
  }
}

async function loadSession(
  modelUrl: string,
  cacheName: string,
  executionProviders: ort.InferenceSession.ExecutionProviderConfig[],
): Promise<ort.InferenceSession> {
  if (typeof caches === "undefined") {
    return ort.InferenceSession.create(modelUrl, { executionProviders });
  }

  const cache = await caches.open(cacheName);
  let response = await cache.match(modelUrl);
  if (!response) {
    response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to download model asset (${response.status}): ${modelUrl}`);
    }
    await cache.put(modelUrl, response.clone());
  }

  return ort.InferenceSession.create(await response.arrayBuffer(), { executionProviders });
}

function argmaxLastToken(logits: ort.Tensor): number {
  const dims = logits.dims;
  if (dims.length !== 3) throw new Error(`Expected decoder logits with 3 dimensions, got ${dims.length}.`);
  const vocabSize = dims[2];
  const offset = (dims[1] - 1) * vocabSize;
  const values = logits.data as Float32Array;
  let best = 0;
  for (let i = 1; i < vocabSize; i += 1) {
    if (values[offset + i] > values[offset + best]) best = i;
  }
  return best;
}
