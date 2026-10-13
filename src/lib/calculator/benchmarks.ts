import { calculate, runtimeFor } from "./engine";
import type { CalculatorInput, SpecMethod } from "./types";
import { GPUS, MODELS } from "../data";
import benchmarks from "../data/benchmarks.json";

/** One published measurement (src/lib/data/benchmarks.json), used to fit and test the speed model. */
export interface Benchmark {
  gpu: string;
  model: string;
  quant: string;
  runtime: string;
  /** decode_tps: tokens/s per sequence; prefill_tps: prompt tokens/s; ttft_ms: time to first token. */
  metric: "decode_tps" | "prefill_tps" | "ttft_ms";
  /** Context (decode) or prompt (prefill, ttft) length. */
  tokens: number;
  /** Sequences decoded at once. */
  batch?: number;
  /** GPUs, tensor parallel. */
  gpus?: number;
  /** Speculative decoding as run, with vLLM's measured acceptance length (tokens per verification step). */
  spec?: { method: SpecMethod; tokens: number; acceptance_length: number };
  value: number;
  /** A link, or a description for contributed runs. */
  source: string;
}

/** Per-token acceptance a whose expected tokens per step, (1 − a^(k+1)) / (1 − a), equals the measured length. */
export function acceptanceRate(length: number, k: number) {
  let lo = 0;
  let hi = 0.99;
  for (let i = 0; i < 50; i++) {
    const a = (lo + hi) / 2;
    if ((1 - a ** (k + 1)) / (1 - a) < length) lo = a;
    else hi = a;
  }
  return lo;
}

export const BENCHMARKS = benchmarks as Benchmark[];

/** What the engine predicts for a benchmark row, in the row's own unit. */
export function predict(b: Benchmark): number {
  const gpu = GPUS.find((g) => g.key === b.gpu)!;
  const input: CalculatorInput = {
    calc_mode: "inference",
    gpu_key: b.gpu,
    model_variant: MODELS.find((m) => m.slug === b.model)!,
    quantization: b.quant,
    batch_size: 1,
    seq_length: b.tokens,
    concurrent_users: b.batch ?? 1,
    num_gpus: b.gpus ?? 1,
    tp_degree: b.gpus ?? 1,
    framework_preset: b.runtime,
    ...(b.spec && {
      enable_speculative_decoding: true,
      speculative_method: b.spec.method,
      speculative_draft_tokens: b.spec.tokens,
      speculative_acceptance_rate: acceptanceRate(b.spec.acceptance_length, b.spec.tokens),
    }),
  } as CalculatorInput;
  const out = calculate(input, gpu);
  if (b.metric === "decode_tps") return out.per_user_tps ?? NaN;
  if (b.metric === "ttft_ms") return out.ttft_ms ?? NaN;
  // llama-bench times prompt processing alone, without the request overhead included in time to first token.
  return b.tokens / (((out.ttft_ms ?? NaN) - (runtimeFor(b.runtime, gpu).ttftMs ?? 15)) / 1000);
}

/** Typical error: exp(mean |log(predicted / measured)|) − 1, so 0.2 means predictions are off by about 20%. */
export function typicalError(rows: Benchmark[]) {
  const logs = rows.map((b) => Math.abs(Math.log(predict(b) / b.value)));
  return Math.exp(logs.reduce((s, x) => s + x, 0) / logs.length) - 1;
}
