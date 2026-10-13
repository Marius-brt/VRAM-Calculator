import test from "node:test";
import assert from "node:assert/strict";
import { calculate } from "../lib/calculator/engine.ts";
import { planPool } from "../lib/calculator/planner.ts";
import { specFromConfig } from "../lib/calculator/hf.ts";
import { launch, trainCommand } from "../components/calculator/LaunchCommand.tsx";
import { BENCHMARKS, predict, typicalError } from "../lib/calculator/benchmarks.ts";
import models from "../lib/data/models.json";
import gpus from "../lib/data/gpus.json";

const model = (slug) => {
  const m = models.find((x) => x.slug === slug);
  assert.ok(m, `catalog has ${slug}`);
  return m;
};
const gpu = (key) => gpus.find((g) => g.key === key);
const infer = (slug, gpuKey, extra = {}) =>
  calculate(
    {
      calc_mode: "inference",
      gpu_key: gpuKey,
      model_variant: model(slug),
      quantization: "fp16",
      batch_size: 1,
      seq_length: 8192,
      concurrent_users: 1,
      num_gpus: 1,
      tp_degree: 1,
      framework_preset: "vllm",
      ...extra,
    },
    gpu(gpuKey),
  );

test("Llama 3.1 8B BF16 on one H100 matches commonly reported figures", () => {
  const out = infer("llama-3-1-8b-instruct", "h100_80");
  assert.ok(Math.abs(out.memory_details["Model Weights"] - 14.96) < 0.2, `weights ${out.memory_details["Model Weights"]}`);
  assert.ok(Math.abs(out.memory_details["KV Cache"] - 1) < 0.05, `8K tokens of KV cache ${out.memory_details["KV Cache"]}`);
  assert.ok(
    out.estimated_latency_tps > 100 && out.estimated_latency_tps < 180,
    `single stream ${out.estimated_latency_tps} tok/s`,
  );
  assert.equal(out.memory_status, "okay");
});

test("Lower precision means smaller weights; GGUF Q4_K_M is about 4.85 bits per weight", () => {
  const w = (q) => infer("llama-3-1-8b-instruct", "h100_80", { quantization: q }).memory_details["Model Weights"];
  assert.ok(w("fp16") > w("fp8") && w("fp8") > w("awq"));
  assert.ok(Math.abs(w("q4_k_m") - (8.03e9 * 4.85) / 8 / 1024 ** 3) < 0.3, `Q4_K_M weights ${w("q4_k_m")}`);
});

test("KV cache follows the attention design: sliding windows and linear-attention layers keep it small", () => {
  const long = { seq_length: 32768 };
  // gpt-oss: half the layers use a 128-token window.
  const oss = infer("gpt-oss-120b", "h100_80", { ...long, quantization: "mxfp4" }).memory_details["KV Cache"];
  // Qwen3.5-122B: 3 of 4 layers are linear attention with a fixed state.
  const qwen = infer("qwen3-5-122b-a10b", "h100_80", { ...long, quantization: "awq" }).memory_details["KV Cache"];
  const llama70 = infer("llama-3-3-70b-instruct", "h100_80", { ...long, quantization: "awq" }).memory_details["KV Cache"];
  assert.ok(oss < llama70 / 4, `gpt-oss ${oss} vs Llama 70B ${llama70}`);
  assert.ok(qwen < llama70 / 4, `Qwen3.5 ${qwen} vs Llama 70B ${llama70}`);
});

test("Tensor parallel across nodes is much slower than inside a node", () => {
  const big = {
    quantization: "q4_k_m",
    concurrent_users: 1,
    inter_node_interconnect: "infiniband_hdr",
    interconnect_type: "nvlink_gen5",
  };
  const insideNodes = infer("llama-3-1-405b-instruct", "b200_180", {
    ...big,
    num_gpus: 64,
    num_nodes: 8,
    gpus_per_node: 8,
    tp_degree: 8,
    pp_degree: 8,
  });
  const acrossNodes = infer("llama-3-1-405b-instruct", "b200_180", {
    ...big,
    num_gpus: 64,
    num_nodes: 64,
    gpus_per_node: 1,
    tp_degree: 64,
    pp_degree: 1,
  });
  assert.ok(
    acrossNodes.estimated_latency_tps < insideNodes.estimated_latency_tps / 1.5,
    `TP64 across nodes ${acrossNodes.estimated_latency_tps} vs TP8 x PP8 ${insideNodes.estimated_latency_tps} tok/s`,
  );
  assert.ok(acrossNodes.estimated_latency_tps < 150, `cross-node TP ${acrossNodes.estimated_latency_tps} tok/s`);
});

test("More users share the bandwidth: per-user speed drops, total throughput rises", () => {
  const one = infer("llama-3-1-8b-instruct", "h100_80", { seq_length: 2048, concurrent_users: 1 });
  const many = infer("llama-3-1-8b-instruct", "h100_80", { seq_length: 2048, concurrent_users: 32 });
  assert.ok(many.per_user_tps < one.per_user_tps);
  assert.ok(many.estimated_throughput_tps > one.estimated_throughput_tps * 4);
});

test("Fine-tuning: QLoRA < LoRA < full fine-tuning, and ZeRO-3 shards across GPUs", () => {
  const ft = (method, extra = {}) =>
    calculate(
      {
        calc_mode: "finetuning",
        gpu_key: "h100_80",
        model_variant: model("llama-3-1-8b-instruct"),
        batch_size: 1,
        seq_length: 2048,
        num_gpus: 1,
        finetuning_method: method,
        framework_preset: "unsloth",
        ...extra,
      },
      gpu("h100_80"),
    ).per_gpu_usage;
  assert.ok(ft("qlora") < ft("lora") && ft("lora") < ft("full"));
  assert.ok(ft("qlora") > 5 && ft("qlora") < 11, `QLoRA 8B ${ft("qlora")} GB`);
  const zero3 = ft("full", { num_gpus: 4, framework_preset: "hf_trainer", optimization_config: { zero_stage: 3 } });
  assert.ok(zero3 < ft("full", { framework_preset: "hf_trainer" }) / 2.5);
});

test("Embedding models: no KV cache, throughput in embeddings/s, batching helps", () => {
  const embed = (batch) => infer("bge-m3", "4090_24", { seq_length: 512, batch_size: batch });
  assert.equal(embed(1).memory_details["KV Cache"], undefined);
  assert.equal(embed(1).estimated_latency_tps, null);
  assert.ok(embed(64).items_per_second > embed(1).items_per_second);
});

// ─── Planner ──────────────────────────────────────────────────────────────────

const g24 = [
  { key: "g24", label: "24GB", vendor: "nvidia", type: "discrete", memory: 24, bandwidth_gbs: 1000, fp16_tflops: 100 },
];
/** Fake engine: 2 bytes per parameter at FP16, split over tp GPUs, speed falling with users per copy. */
const fake = (i) => {
  const bytes = { fp16: 2, fp8: 1, awq: 0.5 }[i.quantization];
  const n = i.num_gpus;
  const perGpu = (i.model_variant.num_of_params * bytes) / n + 1;
  return {
    per_gpu_usage: perGpu,
    per_gpu_usable: 24,
    gpu_count: n,
    memory_status: perGpu > 24 ? "insufficient" : "okay",
    memory_details: { "KV Cache": 0.5 * n },
    per_user_tps: (100 * n) / i.concurrent_users,
    estimated_throughput_tps: 100 * n,
    ttft_ms: 100,
  };
};
const pm = (id, params, extra = {}) => ({
  id,
  model: { name: id, num_of_params: params },
  contextLength: 4096,
  users: 1,
  quant: "auto",
  ...extra,
});
const node = (count) => [{ id: "n", gpuKey: "g24", count, link: "pcie4" }];

test("planPool: largest first, precision steps down to fit, then quality is won back", () => {
  const plan = planPool({ network: "ethernet100g", nodes: node(2), models: [pm("small", 8), pm("big", 20)] }, g24, fake);
  const by = Object.fromEntries(plan.placements.map((p) => [p.modelId, p]));
  assert.equal(plan.unplaced.length, 0);
  assert.equal(by.big.quant, "fp8");
  assert.equal(by.small.quant, "fp16");
});

test("planPool: a lone model on a 2-GPU node gets tensor parallel 2", () => {
  const plan = planPool({ network: "ethernet100g", nodes: node(2), models: [pm("m", 8)] }, g24, fake);
  assert.equal(plan.placements[0].tp, 2);
});

test("planPool: speed targets add GPUs; idle GPUs get extra copies for busy models", () => {
  // 8 users at 100 tok/s per GPU group: one GPU gives 12.5 tok/s each, the target needs 40.
  const plan = planPool(
    { network: "ethernet100g", nodes: node(4), models: [pm("m", 4, { users: 8, target: { minSpeed: 40 } })] },
    g24,
    fake,
  );
  const p = plan.placements[0];
  assert.equal(p.meetsTarget, true);
  assert.ok(p.gpus.length >= 4, `uses ${p.gpus.length} GPUs`);
  // Without a target, idle GPUs still go to extra copies once TP 2 is taken.
  const busy = planPool({ network: "ethernet100g", nodes: node(4), models: [pm("m", 4, { users: 8 })] }, g24, fake).placements[0];
  assert.ok(busy.copies.length >= 2, `${busy.copies.length} copies`);
});

test("planPool: respects pinned GPUs and tensor parallel", () => {
  const plan = planPool(
    {
      network: "ethernet100g",
      nodes: node(4),
      models: [
        pm("a", 5, { quant: "fp16" }),
        pm("pinned", 5, { quant: "fp16", pin: { nodeId: "n", gpus: [3] } }),
        pm("wide", 5, { quant: "fp16", pin: { tp: 2 } }),
      ],
    },
    g24,
    fake,
  );
  const by = Object.fromEntries(plan.placements.map((p) => [p.modelId, p]));
  assert.deepEqual(by.pinned.gpus, [{ nodeId: "n", index: 3 }]);
  assert.equal(by.wide.tp, 2);
});

test("planPool: a pinned copy count is kept, each copy on its own GPUs", () => {
  const plan = planPool(
    { network: "ethernet100g", nodes: node(4), models: [pm("m", 2, { users: 8, pin: { copies: 3 } })] },
    g24,
    fake,
  );
  const p = plan.placements[0];
  assert.equal(p.copies.length, 3);
  assert.equal(new Set(p.gpus.map((g) => g.index)).size, p.gpus.length, "no GPU shared between copies");
});

test("launch: one copy per GPU group, with its own devices and port", () => {
  const llama = model("llama-3-1-8b-instruct");
  const base = { tp: 2, pp: 1, users: 16, contextLength: 8192, devices: [2, 3], copy: 1 };
  const { cmd } = launch(llama, { ...base, runtime: "vllm", quant: "fp8" });
  assert.match(cmd, /^CUDA_VISIBLE_DEVICES=2,3 vllm serve meta-llama\/Llama-3.1-8B-Instruct --tensor-parallel-size 2 /);
  assert.match(cmd, /--max-num-seqs 16 --quantization fp8 --port 8001$/);
  const gguf = launch(llama, { ...base, runtime: "llama_cpp", quant: "q4_k_m", tp: 1, users: 4, devices: null, copy: null }).cmd;
  assert.equal(gguf, "llama-server -m Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 32768 -np 4 -ngl all --port 8080");
});

test("launch: speculative decoding and offloading map to each runtime's flags, or to a note", () => {
  const llama = model("llama-3-1-8b-instruct");
  const base = {
    quant: "fp16",
    tp: 1,
    pp: 1,
    users: 1,
    contextLength: 8192,
    devices: null,
    copy: null,
    spec: { method: "eagle3", tokens: 4 },
  };
  assert.match(
    launch(llama, { ...base, runtime: "vllm" }).cmd,
    /--speculative-config '\{"method":"eagle3","model":"<eagle3-head>","num_speculative_tokens":4\}'/,
  );
  assert.match(
    launch(llama, { ...base, runtime: "sglang" }).cmd,
    /--speculative-algorithm EAGLE3 --speculative-draft-model-path <eagle3-head> --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5/,
  );
  assert.match(
    launch(llama, { ...base, runtime: "tensorrt_llm" }).cmd,
    /decoding_type: Eagle\n  max_draft_len: 4\n  speculative_model: <eagle3-head>\nEOF\ntrtllm-serve .* --config config.yaml/,
  );
  const suffix = launch(llama, { ...base, runtime: "sglang", spec: { method: "suffix", tokens: 8 } });
  assert.doesNotMatch(suffix.cmd, /speculative/);
  assert.ok(suffix.notes.some((n) => n.includes("doesn't support suffix decoding")));
  const offload = launch(llama, { ...base, runtime: "vllm", spec: undefined, offload: { fraction: 0.5, kvGb: 2.2 } }).cmd;
  assert.match(offload, /--cpu-offload-gb 8 --kv-offloading-size 3 --kv-offloading-backend native/);
});

test("trainCommand: TRL with ZeRO-3 over 4 GPUs", () => {
  const cmd = trainCommand({
    calc_mode: "finetuning",
    framework_preset: "hf_trainer",
    model_variant: model("llama-3-1-8b-instruct"),
    finetuning_method: "qlora",
    lora_rank: 16,
    batch_size: 2,
    seq_length: 2048,
    num_gpus: 4,
    optimization_config: { zero_stage: 3 },
  }).cmd;
  assert.match(
    cmd,
    /^trl sft --model_name_or_path meta-llama\/Llama-3.1-8B-Instruct --dataset_name <your-dataset> --use_peft --lora_r 16 --lora_alpha 32 --load_in_4bit /,
  );
  assert.match(cmd, /--accelerate_config zero3 --num_processes 4 --output_dir/);
});

// ─── Hugging Face config parsing ──────────────────────────────────────────────

test("Hugging Face parser: text_config, hybrid attention, shared experts and name hints", () => {
  // Trimmed from Qwen/Qwen3.5-122B-A10B's config.json.
  const config = {
    vision_config: {},
    text_config: {
      hidden_size: 3072,
      num_hidden_layers: 48,
      num_attention_heads: 32,
      num_key_value_heads: 2,
      head_dim: 256,
      vocab_size: 248320,
      max_position_embeddings: 262144,
      num_experts: 256,
      num_experts_per_tok: 8,
      moe_intermediate_size: 1024,
      shared_expert_intermediate_size: 1024,
      linear_num_value_heads: 64,
      linear_key_head_dim: 128,
      linear_value_head_dim: 128,
      layer_types: [...Array(36).fill("linear_attention"), ...Array(12).fill("full_attention")],
    },
  };
  const s = specFromConfig("Qwen/Qwen3.5-122B-A10B", config, {
    safetensors: { parameters: { BF16: 125086490096 }, total: 125086490096 },
  });
  assert.equal(s.num_of_params, 125.09);
  assert.equal(s.active_params, 10);
  assert.equal(s.kv_layers, 12);
  assert.equal(s.state_layers, 36);
  assert.equal(s.num_active_experts, 9);
  assert.equal(s.provider, "Alibaba");
});

test("Hugging Face parser: Gemma 3 sliding-window pattern", () => {
  const s = specFromConfig("google/gemma-3-27b-it", {
    text_config: {
      hidden_size: 5376,
      num_hidden_layers: 62,
      num_attention_heads: 32,
      num_key_value_heads: 16,
      head_dim: 128,
      sliding_window: 1024,
      sliding_window_pattern: 6,
    },
  });
  assert.equal(s.kv_layers, 10);
  assert.equal(s.sliding_layers, 52);
  assert.equal(s.sliding_window_size, 1024);
});

test("Speed predictions stay close to published benchmarks", () => {
  // Fitted error after `bun scripts/fit-speed.ts --write`. GPUs left out of the fit (--cv) are off by about 10% when
  // decoding and 25% on llama.cpp prompt processing.
  const LIMITS = { decode_tps: 0.12, prefill_tps: 0.22, ttft_ms: 0.1 };
  for (const runtime of new Set(BENCHMARKS.map((b) => b.runtime)))
    for (const [metric, limit] of Object.entries(LIMITS)) {
      const rows = BENCHMARKS.filter((b) => b.runtime === runtime && b.metric === metric);
      if (!rows.length) continue;
      const error = typicalError(rows);
      assert.ok(error < limit, `${runtime} ${metric}: typical error ${(error * 100).toFixed(0)}% over ${rows.length} benchmarks`);
    }
  for (const b of BENCHMARKS) {
    const ratio = predict(b) / b.value;
    assert.ok(
      ratio > 0.4 && ratio < 2.5,
      `${b.runtime} ${b.gpu} ${b.quant} ${b.metric}: predicted ${ratio.toFixed(2)}× the measured value`,
    );
  }
});
