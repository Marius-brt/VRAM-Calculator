import { CalculatorInput, CalculatorOutput, GpuSpec, ModelVariant } from "./types";
import speedFit from "../data/speed-fit.json";

/**
 * VRAM, speed and cost model. Pure functions, no I/O: runs in the browser and on the server.
 *
 * Memory: weights (bits per weight per format) + KV cache (per layer type) + activations (one prefill
 * chunk) + runtime overhead, against what the runtime leaves usable on each GPU.
 * Speed: decoding is bandwidth-bound (weights + the experts a batch touches + KV reads, plus
 * tensor-parallel all-reduces); prefill is compute-bound. Fine-tuning uses the usual bytes per
 * parameter and the activation formula from Korthikanti et al. (2022).
 * Speed constants start from typical values per runtime and are fitted to published benchmarks per GPU class
 * (scripts/fit-speed.ts, src/lib/data/benchmarks.json); they are not measurements of your system.
 */

const GiB = 1024 ** 3;

/** Average stored bits per weight, including quantization scales. GGUF figures are whole-file averages. */
export const WEIGHT_BITS: Record<string, number> = {
  fp32: 32,
  fp16: 16,
  bf16: 16,
  fp8: 8,
  int8: 8.1,
  q8_0: 8.5,
  q8: 8.5,
  nvfp6: 6.5,
  q6_k: 6.56,
  q5_k_m: 5.7,
  q5_k_s: 5.54,
  q4_k_m: 4.85,
  q4_k_s: 4.6,
  q4_0: 4.54,
  nvfp4: 4.5,
  int4: 4.5,
  exl2: 4.5,
  mxfp4: 4.25,
  awq: 4.25,
  gptq: 4.25,
  q3_k_m: 3.91,
  q2_k: 3.0,
};
const isGguf = (q: string) => /^q\d/.test(q);

export const KV_BYTES: Record<string, number> = { auto: 2, fp16: 2, bf16: 2, fp8: 1, int8: 1, int4: 0.5 };

interface Runtime {
  /** Share of GPU memory the runtime will use (e.g. vLLM --gpu-memory-utilization). */
  reserve: number;
  overheadGb: number;
  /** Achieved share of peak memory bandwidth while decoding. */
  bwEff: number;
  /** Fixed cost per decode step and per layer (scheduling, kernel launches). */
  stepMs: number;
  layerUs: number;
  /** Extra cost per MoE layer at small batches: routing and many small expert matmuls. */
  moeLayerUs: number;
  prefillMfu: number;
  /** Tokens processed per prefill chunk (sizes the activation buffer). */
  chunk: number;
  /** Fixed part of time to first token: scheduling, tokenization, sampling (default 15 ms). */
  ttftMs?: number;
}

// Starting values. FITTED (below) replaces the speed constants of runtimes that have published benchmarks.
export const RUNTIMES: Record<string, Runtime> = {
  vllm: { reserve: 0.9, overheadGb: 1.5, bwEff: 0.72, stepMs: 0.6, layerUs: 4, moeLayerUs: 50, prefillMfu: 0.45, chunk: 8192 },
  sglang: { reserve: 0.88, overheadGb: 1.5, bwEff: 0.74, stepMs: 0.5, layerUs: 4, moeLayerUs: 40, prefillMfu: 0.45, chunk: 8192 },
  tensorrt_llm: {
    reserve: 0.9,
    overheadGb: 1.2,
    bwEff: 0.78,
    stepMs: 0.4,
    layerUs: 3,
    moeLayerUs: 35,
    prefillMfu: 0.55,
    chunk: 8192,
  },
  llama_cpp: { reserve: 1, overheadGb: 0.4, bwEff: 0.75, stepMs: 0.5, layerUs: 8, moeLayerUs: 50, prefillMfu: 0.7, chunk: 512 },
  ollama: { reserve: 1, overheadGb: 0.5, bwEff: 0.72, stepMs: 1, layerUs: 10, moeLayerUs: 55, prefillMfu: 0.65, chunk: 512 },
  hf_transformers: {
    reserve: 1,
    overheadGb: 0.8,
    bwEff: 0.5,
    stepMs: 5,
    layerUs: 60,
    moeLayerUs: 150,
    prefillMfu: 0.3,
    chunk: 4096,
  },
  generic: { reserve: 0.9, overheadGb: 1, bwEff: 0.7, stepMs: 1, layerUs: 10, moeLayerUs: 60, prefillMfu: 0.4, chunk: 4096 },
};

/**
 * Speed constants fitted to src/lib/data/benchmarks.json by scripts/fit-speed.ts: per runtime, values for every GPU
 * ("all") and per GPU class. llama.cpp's kernels, for one, run far below the tensor-core peak of datacenter GPUs.
 */
export const FITTED: Record<string, Record<string, Partial<Runtime>>> = speedFit;
export const gpuClass = (gpu: GpuSpec) => `${gpu.vendor}-${gpu.category}`;

export function runtimeFor(name: string, gpu: GpuSpec): Runtime {
  const key = name in RUNTIMES ? name : "generic";
  return { ...RUNTIMES[key], ...FITTED[key]?.all, ...FITTED[key]?.[gpuClass(gpu)] };
}

/** Unsloth's own 4-bit and 16-bit loader; other formats go to vLLM (`unsloth run --engine vllm`). */
export const UNSLOTH_LOADER = new Set(["fp16", "bf16", "fp32", "int4"]);
/**
 * The runtime that does the work. `unsloth run` serves GGUF files with llama-server, 16-bit and 4-bit checkpoints
 * with its Transformers-based loader and other formats with vLLM, so it takes their constants.
 */
export const backendOf = (preset: string, quant: string) =>
  preset !== "unsloth" ? preset : isGguf(quant) ? "llama_cpp" : UNSLOTH_LOADER.has(quant) ? "hf_transformers" : "vllm";

/** GPU-to-GPU links: bandwidth per direction (GB/s) and all-reduce latency (µs). */
export const LINKS: Record<string, { gbps: number; us: number }> = {
  nvlink_gen6: { gbps: 1800, us: 4 },
  nvlink_gen5: { gbps: 900, us: 5 },
  nvlink_gen4: { gbps: 450, us: 6 },
  nvlink_gen3: { gbps: 300, us: 7 },
  pcie5: { gbps: 64, us: 15 },
  pcie4: { gbps: 32, us: 20 },
  infiniband_ndr: { gbps: 50, us: 15 },
  infiniband_hdr: { gbps: 25, us: 18 },
  infiniband_edr: { gbps: 12.5, us: 20 },
  ethernet400g: { gbps: 50, us: 25 },
  ethernet200g: { gbps: 25, us: 28 },
  ethernet100g: { gbps: 12.5, us: 30 },
  ethernet25g: { gbps: 3.1, us: 45 },
  ethernet10g: { gbps: 1.25, us: 60 },
  thunderbolt5: { gbps: 10, us: 40 },
};

/** Choices shown in the UI; labels quote the usual bidirectional figures. */
export const GPU_LINK_OPTIONS: [string, string][] = [
  ["nvlink_gen5", "NVLink 5 (1.8 TB/s, B200/B300)"],
  ["nvlink_gen4", "NVLink 4 (900 GB/s, H100/H200)"],
  ["nvlink_gen3", "NVLink 3 (600 GB/s, A100)"],
  ["pcie5", "PCIe 5 (128 GB/s)"],
  ["pcie4", "PCIe 4 (64 GB/s)"],
];
export const NETWORK_OPTIONS: [string, string][] = [
  ["infiniband_ndr", "InfiniBand NDR (400 Gb/s)"],
  ["infiniband_hdr", "InfiniBand HDR (200 Gb/s)"],
  ["ethernet400g", "400G Ethernet (RoCE)"],
  ["ethernet200g", "200G Ethernet"],
  ["ethernet100g", "100G Ethernet"],
  ["ethernet25g", "25G Ethernet"],
  ["ethernet10g", "10G Ethernet"],
  ["thunderbolt5", "Thunderbolt 5 (80 Gb/s, Macs)"],
];

/** The link GPUs of this type are usually connected with. */
export function defaultGpuLink(gpu?: Pick<GpuSpec, "key" | "category">) {
  const key = gpu?.key ?? "";
  if (/^(b200|b300|gb200)/.test(key)) return "nvlink_gen5";
  if (/^(h100_|h200_|gh200|h800|h20_)/.test(key)) return "nvlink_gen4";
  if (/^(a100_|a800)/.test(key)) return "nvlink_gen3";
  return /5090|blackwell|rtx_pro|mi3|mi35/.test(key) ? "pcie5" : "pcie4";
}

/** Host memory bandwidth for layers offloaded to the CPU (dual-channel DDR5). */
export const CPU_GBPS = 60;

export const isEncoderTask = (m?: ModelVariant) => m?.task === "embedding" || m?.task === "reranking";

const n = (v: unknown, fallback: number) => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : fallback);
const round = (v: number, d = 2) => Number(v.toFixed(d));

/** Model shape with defaults filled in. */
export function geometry(m: ModelVariant) {
  const layers = n(m.num_of_layers, 32);
  const hidden = n(m.hidden_size, 4096);
  const heads = n(m.num_attention_heads, 32);
  const kvHeads = n(m.num_key_value_heads, heads);
  const headDim = n(m.attention_head_dim, hidden / heads);
  const vocab = n(m.vocab_size, 32000);
  const total = n(m.num_of_params, 7) * 1e9;
  const moe = m.architecture === "moe" && n(m.num_experts, 0) > 1;
  // Parameters used per token; without a published figure, assume experts hold nearly all weights.
  const active = moe
    ? Math.min(total, m.active_params ? m.active_params * 1e9 : (total / n(m.num_experts, 8)) * n(m.num_active_experts, 2))
    : total;
  const mla = String(m.attention_structure ?? "").toUpperCase() === "MLA";
  const sliding = Math.min(layers, n(m.sliding_layers, 0));
  const state = Math.min(layers - sliding, n(m.state_layers, 0));
  const kvLayers = Math.min(layers - sliding - state, m.kv_layers ?? layers - sliding - state);
  return {
    layers,
    hidden,
    heads,
    kvHeads,
    headDim,
    vocab,
    total,
    active,
    moe,
    experts: n(m.num_experts, 1),
    perToken: Math.max(1, n(m.num_active_experts, 1) - n(m.num_of_shared_experts, 0)),
    ffn: moe
      ? n(m.moe_ffn_intermediate_size, n(m.ffn_intermediate_size, hidden * 3.5)) * n(m.num_active_experts, 2)
      : n(m.ffn_intermediate_size, hidden * 3.5),
    embed: vocab * hidden * (m.tie_word_embeddings ? 1 : 2),
    mla,
    /** Cached values per token per layer (elements). */
    kvPerTokenLayer: mla ? n(m.kv_latent_dim, 576) : 2 * kvHeads * headDim,
    kvLayers,
    slidingLayers: sliding,
    window: n(m.sliding_window_size, 4096),
    stateLayers: state,
    stateBytes: n(m.state_bytes_per_layer, hidden * 128 * 4),
  };
}

type Geometry = ReturnType<typeof geometry>;

/** Bytes of weights for a model at a given precision; embeddings and the output head stay 16-bit outside GGUF. */
export function weightBytes(g: Geometry, quant: string) {
  const bits = WEIGHT_BITS[quant] ?? 16;
  if (isGguf(quant) || bits >= 16) return (g.total * bits) / 8;
  return ((g.total - g.embed) * bits) / 8 + g.embed * 2;
}

/** KV cache (and recurrent state) of one sequence, bytes, before tensor-parallel sharding. */
function kvBytesPerSeq(g: Geometry, tokens: number, kvBytes: number) {
  return (
    (g.kvLayers * tokens + g.slidingLayers * Math.min(tokens, g.window)) * g.kvPerTokenLayer * kvBytes +
    g.stateLayers * g.stateBytes
  );
}

/** Share of the KV cache each tensor-parallel GPU holds: heads split until each GPU has one; MLA latents are replicated. */
const kvShard = (g: Geometry, tp: number) => (g.mla ? 1 : Math.ceil(g.kvHeads / tp) / g.kvHeads);

/** Time of one all-reduce of `bytes` across `tp` GPUs over `link`, seconds. */
function allReduce(bytes: number, tp: number, link: { gbps: number; us: number }) {
  if (tp <= 1) return 0;
  return (link.us * (1 + Math.log2(tp) / 2)) / 1e6 + ((2 * (tp - 1)) / tp) * (bytes / (link.gbps * 1e9));
}

function status(pct: number): CalculatorOutput["memory_status"] {
  return pct > 100 ? "insufficient" : pct >= 90 ? "very_high" : pct >= 75 ? "high" : pct >= 50 ? "moderate" : "okay";
}

function usableShare(gpu: GpuSpec, runtime: Runtime) {
  // macOS lets the GPU wire about 75% of unified memory (two thirds on smaller machines).
  if (gpu.type === "apple_silicon") return Math.min(runtime.reserve, gpu.memory > 36 ? 0.75 : 0.67);
  return runtime.reserve;
}

function energy(input: CalculatorInput, gpu: GpuSpec, gpus: number, load: number) {
  const power = Math.round(n(gpu.tdp_watts, 300) * gpus * load);
  const dailyKwh = round((power * 24) / 1000);
  const price = gpu.hourly_price_usd != null ? gpu.hourly_price_usd * gpus : null;
  return {
    estimated_power_draw: power,
    daily_kwh: dailyKwh,
    monthly_co2_kg: round((dailyKwh * 30 * n(input.carbon_intensity, 380)) / 1000, 1),
    electricity_cost_month: Math.round(dailyKwh * 30 * n(input.electricity_price, 0.15)),
    hourly_cost_usd: price != null ? round(price) : null,
    monthly_cost_usd: price != null ? Math.round(price * 24 * 30) : null,
  };
}

function breakdown(details: Record<string, number>) {
  const total = Object.values(details).reduce((s, v) => s + v, 0) || 1;
  const entries = Object.entries(details).filter(([, v]) => v > 0.005);
  return {
    memory_details: Object.fromEntries(entries.map(([k, v]) => [k, round(v)])),
    memory_breakdown: entries.map(([label, v]) => ({ label, size_gb: round(v), value: round((v / total) * 100, 1) })),
  };
}

/** Apply custom hardware overrides from the input. */
function hardware(input: CalculatorInput, gpu: GpuSpec): GpuSpec {
  return {
    ...gpu,
    memory: n(input.custom_vram, gpu.memory),
    bandwidth_gbs: n(input.custom_bandwidth, gpu.bandwidth_gbs),
    fp16_tflops: n(input.custom_tflops, gpu.fp16_tflops),
  };
}

export function calculate(input: CalculatorInput, rawGpu: GpuSpec): CalculatorOutput {
  const gpu = hardware(input, rawGpu);
  if (isEncoderTask(input.model_variant)) return encoder(input, gpu);
  return input.calc_mode === "finetuning" ? finetune(input, gpu) : inference(input, gpu);
}

// ─── Inference ────────────────────────────────────────────────────────────────

/**
 * Speculative decoding drafter: bytes it adds to each copy, and for each drafting pass the bytes it reads and the layers
 * it runs. Learned drafters are sized from the target: an EAGLE-3 head is one decoder layer plus a 32K-token output head,
 * DFlash about five layers that draft a whole block in one pass, MTP one more layer of the model; MTP and DFlash also
 * run the target's output head. A separate draft model is taken as 16 layers.
 */
function drafter(input: CalculatorInput, g: Geometry, bytesPerParam: number, k: number) {
  const layer = 12 * g.hidden ** 2;
  const head = g.vocab * g.hidden * 2;
  switch (input.speculative_method ?? "draft") {
    case "ngram":
    case "suffix":
      return { bytes: 0, read: 0, passes: 0, layers: 0 };
    case "mtp":
      return {
        bytes: (g.total / g.layers) * bytesPerParam,
        read: (g.active / g.layers) * bytesPerParam + head,
        passes: k,
        layers: 1,
      };
    case "eagle3": {
      const p = layer + 32000 * g.hidden;
      return { bytes: p * 2, read: p * 2, passes: k, layers: 1 };
    }
    case "dflash":
      return { bytes: 5 * layer * 2, read: 5 * layer * 2 + head, passes: 1, layers: 5 };
    default: {
      const p = n(input.speculative_draft_model_params, 0.5) * 1e9;
      return { bytes: p * 2, read: p * 2, passes: k, layers: 16 };
    }
  }
}

function inference(input: CalculatorInput, gpu: GpuSpec): CalculatorOutput {
  const m = input.model_variant;
  const g = geometry(m);
  const quant = (input.quantization || "fp16").toLowerCase();
  const backend = backendOf(input.framework_preset ?? "vllm", quant);
  const runtime = runtimeFor(backend, gpu);
  const warnings: string[] = [];

  const tp = Math.max(1, Math.round(input.tp_degree || 1));
  const pp = Math.max(1, Math.round(input.pp_degree || 1));
  const perCopy = tp * pp;
  const gpus = Math.max(perCopy, Math.round(input.num_gpus || perCopy));
  const replicas = Math.max(1, Math.floor(gpus / perCopy));
  const seq = Math.max(1, input.seq_length || 2048);
  const users = Math.max(1, input.concurrent_users || 1);
  // Sequences each copy decodes at once.
  const seqs = Math.ceil(users / replicas) * Math.max(1, input.batch_size || 1);

  if (perCopy > gpus) warnings.push(`Tensor × pipeline parallel needs ${perCopy} GPUs but the setup has ${gpus}.`);
  if (m.context_length && seq > m.context_length)
    warnings.push(`Context is longer than the model's ${m.context_length.toLocaleString()} tokens.`);
  if (tp > 1 && g.heads % tp !== 0) warnings.push(`${g.heads} attention heads don't split evenly over tensor parallel ${tp}.`);
  if (input.framework_preset === "unsloth" && backend === "hf_transformers" && seqs > 1)
    warnings.push("Unsloth's own loader answers one request at a time; a GGUF format serves several users at once.");

  // ── Memory per GPU of one copy
  const wBytes = weightBytes(g, quant);
  const offload = input.enable_offloading ? Math.min(0.95, Math.max(0, input.percentage_offload ?? 0.3)) : 0;
  const weightsGpu = (wBytes * (1 - offload)) / perCopy;

  const kvb = KV_BYTES[input.kv_cache_quantization ?? "auto"] ?? 2;
  const prefix = input.enable_prefix_caching ? Math.min(0.95, Math.max(0, input.shared_prefix_ratio ?? 0)) : 0;
  // With prefix caching the shared part of the prompt is stored once per copy.
  const kvSeq = kvBytesPerSeq(g, seq, kvb);
  const kvAll = kvSeq * (seqs * (1 - prefix) + (prefix > 0 ? prefix : 0));
  const kvGpuFull = (kvAll * kvShard(g, tp)) / pp;
  const kvOffload = input.enable_offloading && input.offload_kv_cache ? 0.7 : 0;
  const kvGpu = kvGpuFull * (1 - kvOffload);

  // Activations of one prefill chunk plus logits for every running sequence.
  const chunk = Math.min(runtime.chunk, seq * seqs);
  const actGpu = (chunk * (4 * g.hidden + (2 * g.ffn) / tp) * 2 + (seqs * g.vocab * 4) / tp) / pp;
  const overheadGpu = runtime.overheadGb * GiB + (perCopy > 1 ? 0.5 * GiB : 0);
  // Without fused attention kernels the full score matrix of a prefill chunk is materialized (FP32).
  const eagerAttention =
    input.optimization_config?.flash_attention === false &&
    ["llama_cpp", "ollama", "hf_transformers", "generic"].includes(backend);
  const scoresGpu = eagerAttention ? (chunk * seq * g.heads * 4) / tp / pp : 0;

  const draftTokens = n(input.speculative_draft_tokens, 4);
  const spec = !!input.enable_speculative_decoding;
  const draft = spec ? drafter(input, g, wBytes / g.total, draftTokens) : { bytes: 0, read: 0, passes: 0, layers: 0 };
  const draftGpu = draft.bytes / tp;

  const perGpu = weightsGpu + kvGpu + actGpu + scoresGpu + overheadGpu + draftGpu;
  const usableGpu = gpu.memory * GiB * usableShare(gpu, runtime);
  const pct = (perGpu / usableGpu) * 100;
  const fits = pct <= 100;

  // ── Decode speed of one copy with `b` sequences in flight
  const intraLink = LINKS[input.interconnect_type ?? ""] ?? LINKS[defaultGpuLink(gpu)];
  const netLink = LINKS[input.inter_node_interconnect ?? ""] ?? LINKS.ethernet100g;
  const perNode = n(input.gpus_per_node, gpus);
  const tpLink = tp > perNode ? netLink : intraLink; // tensor parallel across nodes rides the network
  const hopLink = (input.num_nodes ?? 1) > 1 ? netLink : intraLink;
  const bw = gpu.bandwidth_gbs * 1e9 * runtime.bwEff;
  const flops = gpu.fp16_tflops * 1e12;
  const bytesPerParam = wBytes / g.total;
  const routedPerExpert = g.moe ? (g.total - g.active) / Math.max(1, g.experts - g.perToken) : 0;
  const alwaysRead = g.active - g.perToken * routedPerExpert;

  // One forward step over `tokens` tokens of `b` sequences (more than b when speculative decoding verifies drafts).
  const stepSeconds = (b: number, tokens = b) => {
    // MoE: a batch of tokens touches more experts than one token does.
    const touched = g.moe ? g.experts * (1 - Math.pow(1 - g.perToken / g.experts, tokens)) : 0;
    // The input embedding table is only looked up, not streamed (the output head is).
    const lookupOnly = g.embed > g.vocab * g.hidden ? g.vocab * g.hidden : 0;
    const paramsRead = (g.moe ? alwaysRead + touched * routedPerExpert : g.total) - lookupOnly;
    const weightTime = (paramsRead * bytesPerParam * (1 - offload)) / tp / bw;
    const kvTime = (kvSeq * b * kvShard(g, tp) * (1 - kvOffload)) / bw;
    const cpuTime = (paramsRead * bytesPerParam * offload + kvSeq * b * kvOffload) / (CPU_GBPS * 1e9);
    const compute = (2 * g.active * tokens) / (flops * tp * 0.6);
    const comm = 2 * g.layers * allReduce(tokens * g.hidden * 2, tp, tpLink);
    const hops = (pp - 1) * allReduce(tokens * g.hidden * 2, 2, hopLink);
    const fixed = (runtime.stepMs + (g.layers * (runtime.layerUs + (g.moe ? runtime.moeLayerUs : 0))) / 1000) / 1000;
    return Math.max(weightTime + kvTime, compute) + cpuTime + comm + hops + fixed;
  };

  // Speculative decoding: draft k tokens, then verify them in one step over k + 1 tokens per sequence; a step yields
  // (1 − a^(k+1)) / (1 − a) tokens on average (Leviathan et al.). Large batches make verification compute-bound.
  const draftSeconds =
    draft.passes * (draft.read / tp / bw + (draft.layers * (runtime.layerUs + (g.moe ? runtime.moeLayerUs : 0))) / 1e6);
  const speed = (b: number) => {
    const plain = 1 / stepSeconds(b);
    if (!spec) return plain;
    const a = Math.min(0.99, n(input.speculative_acceptance_rate, 0.7));
    const k = draftTokens;
    // Checking drafts is not free beyond the step itself (rejection sampling, KV writes, scheduling): about 3% each.
    const VERIFY_COST = 0.03;
    const drafted = (1 - Math.pow(a, k + 1)) / (1 - a) / ((stepSeconds(b, b * (k + 1)) + draftSeconds) * (1 + VERIFY_COST * k));
    // Measured gains shrink with batch size faster than this roofline does, as kernels don't overlap perfectly.
    return plain * (1 + (drafted / plain - 1) / (1 + b / 8));
  };

  const perUser = fits ? speed(seqs) : null;
  const single = fits ? speed(1) : null;

  // Prefill: compute-bound for long prompts, weight-read-bound for short ones; FP8/FP4 tensor cores speed it up.
  const lowBits = (quant === "fp8" && gpu.fp8) || (/^(nvfp4|mxfp4)$/.test(quant) && gpu.fp4);
  const prefillFlops = flops * tp * (quant === "fp8" && gpu.fp8 ? 1.6 : lowBits ? 2.4 : 1);
  const attnFlops = 2 * (g.kvLayers * seq * seq + g.slidingLayers * seq * Math.min(seq, g.window)) * g.heads * g.headDim;
  const prefillCompute = (2 * g.active * seq + attnFlops) / (prefillFlops * runtime.prefillMfu);
  const prefillReads = (Math.ceil(seq / runtime.chunk) * wBytes) / tp / bw;
  const prefillComm =
    2 * g.layers * allReduce(Math.min(seq, runtime.chunk) * g.hidden * 2, tp, tpLink) * Math.ceil(seq / runtime.chunk);
  const ttft = fits ? Math.max(prefillCompute, prefillReads) + prefillComm + (runtime.ttftMs ?? 15) / 1000 : null;

  // ── KV capacity: how many more users fit
  const freeForKv = usableGpu - (weightsGpu + actGpu + scoresGpu + overheadGpu + draftGpu);
  const kvPerUserGpu = (kvSeq * Math.max(1, input.batch_size || 1) * kvShard(g, tp) * (1 - kvOffload)) / pp;
  const maxUsersPerCopy = kvPerUserGpu > 0 ? Math.max(0, Math.floor(freeForKv / kvPerUserGpu)) : 9999;
  const maxUsers = maxUsersPerCopy * replicas;
  const throughput = perUser != null ? perUser * seqs * replicas : null;
  const OUTPUT_TOKENS = 512;

  const memGb = (bytes: number) => (bytes * gpus) / GiB;
  const { memory_details, memory_breakdown } = breakdown({
    "Model Weights": memGb(weightsGpu),
    "KV Cache": memGb(kvGpu),
    Activations: memGb(actGpu),
    "Attention Scores": memGb(scoresGpu),
    "Framework Overhead": memGb(overheadGpu),
    [input.speculative_method === "mtp" ? "MTP Layers" : "Draft Model"]: memGb(draftGpu),
  });

  return {
    vram_usage: round((perGpu * gpus) / GiB),
    total_gpu_memory: gpu.memory * gpus,
    usable_gpu_memory: round((usableGpu * gpus) / GiB),
    per_gpu_usage: round(perGpu / GiB),
    per_gpu_usable: round(usableGpu / GiB),
    gpu_count: gpus,
    replicas,
    vram_percentage: round(Math.min(100, pct), 1),
    actual_vram_percentage: round(pct, 1),
    memory_status: status(pct),
    memory_details,
    memory_breakdown,
    offloaded_memory: round((wBytes * offload + kvAll * kvOffload) / GiB),
    estimated_latency_tps: single != null ? round(single, 1) : null,
    per_user_tps: perUser != null ? round(perUser, 1) : null,
    estimated_throughput_tps: throughput != null ? round(throughput, 1) : null,
    ms_per_token: perUser != null ? round(1000 / perUser, 1) : null,
    ttft_ms: ttft != null ? Math.round(ttft * 1000) : null,
    concurrency: {
      maxUsers,
      kvPerUserGb: round((kvPerUserGpu * perCopy) / GiB, 3),
      kvFreeGb: round((Math.max(0, freeForKv - kvGpu) * gpus) / GiB),
      requestsPerSecond: throughput != null ? round(throughput / OUTPUT_TOKENS, 2) : 0,
      preemptionRisk: users > maxUsers ? "high" : users > maxUsers * 0.8 ? "moderate" : "none",
    },
    ...energy(input, gpu, gpus, 0.75),
    warnings,
  };
}

// ─── Fine-tuning ──────────────────────────────────────────────────────────────

export const TRAINERS: Record<string, { mfu: number; chunkedLoss: boolean; pureBf16: boolean; offloadCheckpoints: boolean }> = {
  unsloth: { mfu: 0.5, chunkedLoss: true, pureBf16: false, offloadCheckpoints: true },
  torchtune: { mfu: 0.42, chunkedLoss: true, pureBf16: true, offloadCheckpoints: false },
  hf_trainer: { mfu: 0.35, chunkedLoss: false, pureBf16: false, offloadCheckpoints: false },
  generic: { mfu: 0.35, chunkedLoss: false, pureBf16: false, offloadCheckpoints: false },
};

function finetune(input: CalculatorInput, gpu: GpuSpec): CalculatorOutput {
  const m = input.model_variant;
  const g = geometry(m);
  const trainer = TRAINERS[input.framework_preset ?? "unsloth"] ?? TRAINERS.generic;
  const method = input.finetuning_method ?? "lora";
  const precision = (input.fine_tuning_quantization || "bf16").toLowerCase();
  const opt = input.optimization_config ?? {};
  const gpus = Math.max(1, Math.round(input.num_gpus || 1));
  const zero = gpus > 1 ? (opt.zero_stage ?? 0) : 0;
  const seq = Math.max(1, input.seq_length || 2048);
  const micro = Math.max(1, input.batch_size || 1);
  const ckpt = opt.gradient_checkpointing !== false;
  const flash = opt.flash_attention !== false;
  const eightBitOpt = !!(opt.use_8bit_optimizer || opt.use_paged_optimizer);
  const warnings: string[] = [];

  let weights: number,
    grads: number,
    states: number,
    adapters = 0;
  if (method === "full") {
    const wb = precision === "fp32" ? 4 : precision === "fp8" ? 1 : 2;
    const master = wb < 4 && !trainer.pureBf16 ? 4 : 0; // fp32 master copy in mixed precision
    const moments = eightBitOpt ? 2 : trainer.pureBf16 ? 4 : 8;
    weights = (g.total * wb) / (zero >= 3 ? gpus : 1);
    grads = (g.total * Math.max(2, wb)) / (zero >= 2 ? gpus : 1);
    states = (g.total * (master + moments)) / (zero >= 1 ? gpus : 1);
  } else {
    weights = method === "qlora" ? (g.total - g.embed) * 0.516 + g.embed * 2 : g.total * 2;
    // LoRA on every linear layer: q, k, v, o and the MLP's gate, up and down projections.
    const r = n(input.lora_rank, 16);
    const q = g.heads * g.headDim;
    const kv = g.kvHeads * g.headDim;
    const perLayer = r * (g.hidden + q + 2 * (g.hidden + kv) + (q + g.hidden) + 2 * (g.hidden + g.ffn) + (g.ffn + g.hidden));
    const adapterParams = g.layers * perLayer;
    adapters = adapterParams * 2;
    grads = adapterParams * 2;
    states = adapterParams * (4 + (eightBitOpt ? 2 : 8));
  }

  // Activations kept for the backward pass (Korthikanti et al. 2022): 34·s·b·h bytes per layer with flash
  // attention, plus 5·a·s²·b without it; checkpointing keeps only each layer's input and recomputes one layer.
  const perLayerFull = seq * micro * g.hidden * 34 + (flash ? 0 : 5 * g.heads * seq * seq * micro);
  let activations = ckpt ? g.layers * seq * micro * g.hidden * 2 + perLayerFull : g.layers * perLayerFull;
  if (ckpt && trainer.offloadCheckpoints) activations = activations * 0.3;
  const logits = seq * micro * g.vocab * (trainer.chunkedLoss ? 0.5 : 6);
  const overhead = 1 * GiB + (gpus > 1 ? 0.5 * GiB : 0);

  const perGpu = weights + grads + states + adapters + activations + logits + overhead;
  const usableGpu = gpu.memory * GiB * (gpu.type === "apple_silicon" ? (gpu.memory > 36 ? 0.75 : 0.67) : 0.95);
  const pct = (perGpu / usableGpu) * 100;
  const fits = pct <= 100;

  // Training FLOPs per token: forward 2P, backward 4P (2P without weight gradients for LoRA), +2P recompute.
  const flopsPerToken = (method === "full" ? 6 : 4) * g.active + (ckpt ? 2 * g.active : 0);
  const fp8Boost = precision === "fp8" && gpu.fp8 ? 1.4 : 1;
  const qloraCost = method === "qlora" ? 0.75 : 1;
  const tps = fits ? (gpus * gpu.fp16_tflops * 1e12 * trainer.mfu * fp8Boost * qloraCost) / flopsPerToken : null;
  const samplesPerSec = tps != null ? tps / seq : null;
  const accum = Math.max(1, input.gradient_accumulation_steps || 1);
  const samples = n(input.num_samples, 10000) * n(input.num_epochs, 1);

  const { memory_details, memory_breakdown } = breakdown({
    "Model Weights": (weights * gpus) / GiB,
    "LoRA Adapters": (adapters * gpus) / GiB,
    Gradients: (grads * gpus) / GiB,
    "Optimizer States": (states * gpus) / GiB,
    Activations: ((activations + logits) * gpus) / GiB,
    "Framework Overhead": (overhead * gpus) / GiB,
  });
  if (method === "full" && zero === 0 && gpus > 1)
    warnings.push("Without ZeRO every GPU holds a full copy of the optimizer states.");

  return {
    vram_usage: round((perGpu * gpus) / GiB),
    total_gpu_memory: gpu.memory * gpus,
    usable_gpu_memory: round((usableGpu * gpus) / GiB),
    per_gpu_usage: round(perGpu / GiB),
    per_gpu_usable: round(usableGpu / GiB),
    gpu_count: gpus,
    replicas: gpus,
    vram_percentage: round(Math.min(100, pct), 1),
    actual_vram_percentage: round(pct, 1),
    memory_status: status(pct),
    memory_details,
    memory_breakdown,
    offloaded_memory: 0,
    estimated_latency_tps: null,
    per_user_tps: null,
    estimated_throughput_tps: null,
    ms_per_token: null,
    ttft_ms: null,
    training_tps: tps != null ? Math.round(tps) : null,
    samples_per_second: samplesPerSec != null ? round(samplesPerSec, 3) : null,
    steps_per_second: samplesPerSec != null ? round(samplesPerSec / (micro * accum * gpus), 4) : null,
    total_training_time_hours: samplesPerSec ? round(samples / samplesPerSec / 3600, 2) : null,
    ...energy(input, gpu, gpus, 0.9),
    warnings,
  };
}

// ─── Embedding & reranking ────────────────────────────────────────────────────

/**
 * One forward pass per input (or query/document pair), no KV cache. Throughput is compute-bound at large
 * batches and weight-read-bound at small ones. The batch is batch_size × concurrent_users.
 */
function encoder(input: CalculatorInput, gpu: GpuSpec): CalculatorOutput {
  const m = input.model_variant;
  const g = geometry(m);
  const gpus = Math.max(1, Math.round(input.num_gpus || 1));
  const seq = Math.max(1, input.seq_length || 512);
  const batch = Math.max(1, (input.batch_size || 1) * (input.concurrent_users || 1));
  const quant = (input.quantization || "fp16").toLowerCase();

  const wBytes = (g.total * (WEIGHT_BITS[quant] ?? 16)) / 8;
  const activations = batch * seq * (4 * g.hidden + 2 * g.ffn) * 2;
  const overhead = 0.75 * GiB;
  const perGpu = wBytes / gpus + activations / gpus + overhead;
  const usableGpu = gpu.memory * GiB * (gpu.type === "apple_silicon" ? (gpu.memory > 36 ? 0.75 : 0.67) : 0.92);
  const pct = (perGpu / usableGpu) * 100;

  // FLOPs per token: 2 per weight plus attention scores over the whole input.
  const flopsPerToken = 2 * g.total + 4 * g.layers * seq * g.hidden;
  const mfu = g.total < 2e8 ? 0.25 : g.total < 2e9 ? 0.4 : 0.5; // tiny models are launch-bound
  const compute = gpu.fp16_tflops * 1e12 * gpus * (gpus > 1 ? 0.85 : 1) * mfu;
  const bw = gpu.bandwidth_gbs * 1e9 * gpus * 0.7;
  const pass = (tokens: number) => Math.max((tokens * flopsPerToken) / compute, wBytes / bw) + 0.002;
  const batchSec = pass(batch * seq);

  const { memory_details, memory_breakdown } = breakdown({
    "Model Weights": wBytes / GiB,
    Activations: activations / GiB,
    "Framework Overhead": (overhead * gpus) / GiB,
  });

  return {
    task: m.task,
    vram_usage: round((perGpu * gpus) / GiB),
    total_gpu_memory: gpu.memory * gpus,
    usable_gpu_memory: round((usableGpu * gpus) / GiB),
    per_gpu_usage: round(perGpu / GiB),
    per_gpu_usable: round(usableGpu / GiB),
    gpu_count: gpus,
    replicas: 1,
    vram_percentage: round(Math.min(100, pct), 1),
    actual_vram_percentage: round(pct, 1),
    memory_status: status(pct),
    memory_details,
    memory_breakdown,
    offloaded_memory: 0,
    items_per_second: round(batch / batchSec, 1),
    tokens_per_second_processed: Math.round((batch * seq) / batchSec),
    batch_latency_ms: Math.round(batchSec * 1000),
    single_latency_ms: round(pass(seq) * 1000, 1),
    estimated_latency_tps: null,
    per_user_tps: null,
    estimated_throughput_tps: null,
    ms_per_token: null,
    ttft_ms: null,
    ...energy(input, gpu, gpus, 0.6),
    warnings:
      m.context_length && seq > m.context_length ? [`Inputs are longer than the model's ${m.context_length} tokens.`] : [],
  };
}
