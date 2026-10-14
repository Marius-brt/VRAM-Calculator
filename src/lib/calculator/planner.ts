import { calculate, isEncoderTask } from "./engine";
import { CalculatorInput, CalculatorOutput, GpuSpec, ModelTask, ModelVariant } from "./types";

/**
 * Pool planner: places several models on a set of (possibly different) nodes.
 *
 * Strategy, kept deliberately simple and explainable:
 * 1. Pinned models first (node, GPUs or tensor parallel size as the user set them), then largest first.
 *    Each model takes the fewest GPUs that fit and meet its targets (minimum speed per user, maximum
 *    time to first token): more tensor parallelism or extra copies when one GPU group is too slow.
 *    Only when no single node works does a model span identical nodes with pipeline parallelism.
 * 2. "Auto" precision starts every model at the highest quality and steps all of them down together
 *    while that places more models or meets more targets; then each model, smallest first, steps
 *    back up if nothing gets worse.
 * 3. Idle GPUs: a single-GPU model takes a second idle GPU (tensor parallel 2), then remaining idle GPUs
 *    host extra copies of whichever model gains the most speed per user. A pinned copy count is kept as is.
 *
 * ponytail: greedy, not optimal; an exact packer (ILP) would squeeze in more at scale.
 */

export interface PoolNode {
  id: string;
  gpuKey: string;
  count: number;
  /** Link between the GPUs inside this node. */
  link: string;
}

/** The Simple view's serving options, per model; anything left out keeps the defaults (runtime picked by GPU). */
export type ServingOptions = Pick<
  CalculatorInput,
  | "framework_preset"
  | "kv_cache_quantization"
  | "enable_prefix_caching"
  | "shared_prefix_ratio"
  | "enable_offloading"
  | "percentage_offload"
  | "offload_kv_cache"
  | "enable_speculative_decoding"
  | "speculative_method"
  | "speculative_draft_tokens"
  | "speculative_draft_model_params"
  | "speculative_acceptance_rate"
  | "optimization_config"
>;

export interface PoolModel {
  id: string;
  model: ModelVariant;
  /** Context length (tokens) for chat models; input length for embedding/reranking models. */
  contextLength: number;
  /** Concurrent users for chat models; batch size for embedding/reranking models. */
  users: number;
  /** "auto" or a fixed quantization key. */
  quant: string;
  /** Optional user choices; anything left out is decided by the planner. */
  pin?: { nodeId?: string; gpus?: number[]; tp?: number; copies?: number };
  /** Service targets: tokens/s per user (items/s for embedding models) and time to first token. */
  target?: { minSpeed?: number; maxTtftMs?: number };
  /** Chat models only: embedding and reranking models always run on TEI with defaults. */
  options?: ServingOptions;
}

export interface PoolConfig {
  nodes: PoolNode[];
  /** Link between nodes. */
  network: string;
  models: PoolModel[];
}

export interface GpuRef {
  nodeId: string;
  index: number;
}

export interface Placement {
  modelId: string;
  task?: ModelTask;
  runtime: string;
  quant: string;
  tp: number;
  pp: number;
  /** GPUs of each copy of the model. */
  copies: GpuRef[][];
  /** All GPUs used, flattened. */
  gpus: GpuRef[];
  perGpuGb: number;
  /** Part of perGpuGb that is KV cache (0 for embedding/reranking models). */
  kvPerGpuGb: number;
  /** Weights and KV cache kept in CPU memory, per copy (GB). */
  offloadedGb: number;
  /** Share of each GPU's memory to hand this model, e.g. vLLM --gpu-memory-utilization. */
  memoryFraction: number;
  usersPerCopy: number;
  perUserTps: number | null;
  /** All copies together. */
  throughputTps: number | null;
  ttftMs: number | null;
  /** Embedding/reranking: embeddings or pairs scored per second, all copies together. */
  itemsPerSecond: number | null;
  /** null when the model has no target. */
  meetsTarget: boolean | null;
}

export interface GpuUsage extends GpuRef {
  memory: number;
  usable: number;
  used: number;
}

export interface PoolPlan {
  placements: Placement[];
  unplaced: { modelId: string; reason: string }[];
  gpus: GpuUsage[];
  /** Tokens per second across all chat models and copies. */
  totalTps: number;
}

export type Evaluate = (input: CalculatorInput, gpu: GpuSpec) => CalculatorOutput;

/** Highest quality first. vLLM serves FP8/AWQ; Macs run llama.cpp with GGUF files; TEI serves embeddings in FP16. */
export const QUANT_LADDER: Record<string, string[]> = {
  vllm: ["fp16", "fp8", "awq"],
  llama_cpp: ["fp16", "q8_0", "q4_k_m"],
  ollama: ["fp16", "q8_0", "q4_k_m"],
  hf_transformers: ["fp16", "int8", "int4"],
  tei: ["fp16"],
};
const MAX_LEVEL = 2;
const TP_SIZES = [1, 2, 4, 8, 16];
export const MAX_COPIES = 8;

export const runtimeFor = (gpu: GpuSpec, model?: ModelVariant) =>
  isEncoderTask(model) ? "tei" : gpu.vendor === "apple" ? "llama_cpp" : "vllm";

/** The runtime the user picked for a chat model, else the one for its GPU. */
const runtimeOf = (gpu: GpuSpec, m: PoolModel) =>
  (!isEncoderTask(m.model) && m.options?.framework_preset) || runtimeFor(gpu, m.model);

const round = (n: number, d = 2) => Number(n.toFixed(d));

export function planPool(config: PoolConfig, gpuSpecs: GpuSpec[], evaluate: Evaluate = calculate): PoolPlan {
  const nodes = config.nodes
    .map((n) => ({ ...n, gpu: gpuSpecs.find((g) => g.key === n.gpuKey)! }))
    .filter((n) => n.gpu && n.gpu.memory > 0 && n.count > 0)
    // A Mac is one GPU.
    .map((n) => ({ ...n, count: n.gpu.type === "apple_silicon" ? 1 : n.count }));
  type Node = (typeof nodes)[number];
  const usableByGpu = new Map<string, number>();
  const cache = new Map<string, ReturnType<typeof run>>();

  function run(m: PoolModel, gpu: GpuSpec, quant: string, tp: number, pp: number, link: string, users: number) {
    const encoder = isEncoderTask(m.model);
    const out = evaluate(
      {
        calc_mode: "inference",
        kv_cache_quantization: "auto",
        ...(encoder ? {} : m.options),
        gpu_key: gpu.key,
        model_variant: m.model,
        quantization: quant,
        batch_size: 1,
        seq_length: m.contextLength,
        concurrent_users: users,
        num_gpus: tp * pp,
        tp_degree: tp,
        pp_degree: pp,
        num_nodes: pp > 1 ? pp : 1,
        gpus_per_node: pp > 1 ? tp : tp * pp,
        framework_preset: runtimeOf(gpu, m) as CalculatorInput["framework_preset"],
        interconnect_type: link,
        inter_node_interconnect: config.network,
      },
      gpu,
    );
    // Models sharing a GPU must all fit under the strictest runtime reserve.
    const usable = Math.min(out.per_gpu_usable, usableByGpu.get(gpu.key) ?? Infinity);
    usableByGpu.set(gpu.key, usable);
    const speed = encoder ? (out.items_per_second ?? null) : out.per_user_tps;
    const meets =
      !m.target?.minSpeed && !m.target?.maxTtftMs
        ? null
        : out.memory_status !== "insufficient" &&
          (!m.target.minSpeed || (speed ?? 0) >= m.target.minSpeed) &&
          (!m.target.maxTtftMs || encoder || (out.ttft_ms ?? Infinity) <= m.target.maxTtftMs);
    return { out, usable, perGpu: out.per_gpu_usage, kvPerGpu: (out.memory_details["KV Cache"] ?? 0) / out.gpu_count, meets };
  }
  /** One copy of the model on tp × pp GPUs of type `gpu`, serving `users`. */
  const need = (m: PoolModel, gpu: GpuSpec, quant: string, tp: number, pp: number, link: string, users: number) => {
    const key = [m.id, gpu.key, quant, tp, pp, link, users].join("|");
    if (!cache.has(key)) cache.set(key, run(m, gpu, quant, tp, pp, link, users));
    return cache.get(key)!;
  };

  const ladderFor = (gpu: GpuSpec, m: PoolModel) => QUANT_LADDER[runtimeOf(gpu, m)] ?? QUANT_LADDER.vllm;
  const quantFor = (m: PoolModel, gpu: GpuSpec, level: number) =>
    m.quant === "auto" ? ladderFor(gpu, m)[Math.min(level, ladderFor(gpu, m).length - 1)] : m.quant;

  type Used = Map<string, number[]>;
  type Need = ReturnType<typeof need>;

  const placement = (
    m: PoolModel,
    r: Need,
    node: Node,
    quant: string,
    tp: number,
    pp: number,
    copies: GpuRef[][],
  ): Placement => ({
    modelId: m.id,
    task: m.model.task,
    runtime: runtimeOf(node.gpu, m),
    quant,
    tp,
    pp,
    copies,
    gpus: copies.flat(),
    perGpuGb: round(r.perGpu),
    kvPerGpuGb: round(r.kvPerGpu),
    offloadedGb: round(r.out.offloaded_memory),
    memoryFraction: Math.min(0.99, Math.ceil((r.perGpu / node.gpu.memory) * 100 + 1) / 100),
    usersPerCopy: Math.ceil(m.users / copies.length),
    perUserTps: r.out.per_user_tps,
    throughputTps: r.out.estimated_throughput_tps != null ? round(r.out.estimated_throughput_tps * copies.length, 1) : null,
    ttftMs: r.out.ttft_ms,
    itemsPerSecond: r.out.items_per_second != null ? round(r.out.items_per_second * copies.length, 1) : null,
    meetsTarget: r.meets,
  });

  /** k GPUs in a node with room for `perGpu` each, tightest fit first. */
  const pickGpus = (used: number[], usable: number, perGpu: number, k: number) => {
    const fits = used
      .map((u, i) => ({ i, free: usable - u }))
      .filter((g) => g.free >= perGpu)
      .sort((a, b) => a.free - b.free)
      .slice(0, k);
    return fits.length === k ? fits : null;
  };

  /** The user's pins, dropping any that point at nodes or GPUs that no longer exist. */
  const pinsOf = (m: PoolModel) => {
    const node = nodes.find((n) => n.id === m.pin?.nodeId);
    const gpus = node && m.pin?.gpus?.length ? [...new Set(m.pin.gpus)].filter((i) => i < node.count).sort((a, b) => a - b) : [];
    // GPUs picked by hand hold exactly one copy.
    const copies = gpus.length
      ? 1
      : m.pin?.copies
        ? Math.min(MAX_COPIES, Math.max(1, Math.round(Number(m.pin.copies) || 1)))
        : undefined;
    return { node, gpus: gpus.length ? gpus : null, tp: gpus.length ? gpus.length : m.pin?.tp, copies };
  };

  /** GPUs for `copies` copies of tp GPUs each, on nodes with the same GPU type as `first`. */
  const allocate = (used: Used, first: Node, r: Need, tp: number, copies: number, fixed: number[] | null) => {
    const trial = new Map([...used].map(([k, v]) => [k, [...v]]));
    const sameType = [first, ...nodes.filter((n) => n !== first && n.gpuKey === first.gpuKey)];
    const out: GpuRef[][] = [];
    // Copies of one model never share a GPU: they would split its bandwidth instead of adding to it.
    const taken = (node: Node, u: number[]) =>
      u.map((x, i) => (out.flat().some((g) => g.nodeId === node.id && g.index === i) ? Infinity : x));
    for (let c = 0; c < copies; c++) {
      const hit = (fixed ? [first] : sameType).find((node) => {
        const u = trial.get(node.id)!;
        const picked = fixed
          ? fixed.every((i) => u[i] + r.perGpu <= r.usable) && fixed.map((i) => ({ i }))
          : tp <= node.count && pickGpus(taken(node, u), r.usable, r.perGpu, tp);
        if (!picked) return false;
        picked.forEach((g) => (u[g.i] += r.perGpu));
        out.push(picked.map((g) => ({ nodeId: node.id, index: g.i })));
        return true;
      });
      if (!hit) return null;
    }
    return out;
  };

  const place = (m: PoolModel, level: number, used: Used): Placement | null => {
    const pin = pinsOf(m);
    const hasTarget = !!(m.target?.minSpeed || m.target?.maxTtftMs);
    // Shapes in order of GPUs used; extra copies only for models with targets (idle GPUs get them later).
    const copyCounts = pin.copies
      ? [pin.copies]
      : Array.from({ length: hasTarget && !pin.gpus ? MAX_COPIES : 1 }, (_, i) => i + 1);
    const shapes = (pin.tp ? [pin.tp] : TP_SIZES)
      .flatMap((tp) => copyCounts.map((copies) => ({ tp, copies })))
      .sort((a, b) => a.tp * a.copies - b.tp * b.copies || a.copies - b.copies);

    let fallback: Placement | null = null;
    for (const { tp, copies } of shapes) {
      let best: { slack: number; p: Placement } | null = null;
      for (const n of pin.node ? [pin.node] : nodes) {
        if (tp > n.count) continue;
        const quant = quantFor(m, n.gpu, level);
        const r = need(m, n.gpu, quant, tp, 1, n.link, Math.ceil(m.users / copies));
        if (r.out.memory_status === "insufficient") continue;
        const groups = allocate(used, n, r, tp, copies, pin.gpus);
        if (!groups) continue;
        const slack = groups.flat().reduce((s, g) => s + r.usable - used.get(g.nodeId)![g.index] - r.perGpu, 0);
        if (!best || slack < best.slack) best = { slack, p: placement(m, r, n, quant, tp, 1, groups) };
      }
      if (!best) continue;
      if (best.p.meetsTarget !== false) return best.p;
      fallback ??= best.p;
    }
    if (fallback || pin.node || (pin.copies ?? 1) > 1) return fallback;

    // Across identical nodes: pipeline stages over the network, tensor parallel inside each node.
    const byType = new Map<string, Node[]>();
    for (const n of nodes) byType.set(n.gpuKey, [...(byType.get(n.gpuKey) ?? []), n]);
    const spans: { group: Node[]; pp: number; tp: number }[] = [];
    for (const group of byType.values()) {
      for (let pp = 2; pp <= group.length; pp++) for (const tp of pin.tp ? [pin.tp] : TP_SIZES) spans.push({ group, pp, tp });
    }
    spans.sort((a, b) => a.pp * a.tp - b.pp * b.tp);
    for (const { group, pp, tp } of spans) {
      const gpu = group[0].gpu;
      const quant = quantFor(m, gpu, level);
      const r = need(m, gpu, quant, tp, pp, group[0].link, m.users);
      if (r.out.memory_status === "insufficient") continue;
      const stages = group
        .map((n) => ({ n, picked: tp <= n.count ? pickGpus(used.get(n.id)!, r.usable, r.perGpu, tp) : null }))
        .filter((s) => s.picked)
        .slice(0, pp);
      if (stages.length < pp) continue;
      return placement(m, r, group[0], quant, tp, pp, [
        stages.flatMap((s) => s.picked!.map((g) => ({ nodeId: s.n.id, index: g.i }))),
      ]);
    }
    return null;
  };

  // Pinned models first (they have no choice), then largest first, counting every pinned copy.
  const size = (m: PoolModel) => (m.model.num_of_params ?? 0) * (pinsOf(m).copies ?? 1);
  const order = [...config.models].sort((a, b) => Number(!!b.pin?.nodeId) - Number(!!a.pin?.nodeId) || size(b) - size(a));

  const attempt = (levels: Map<string, number>) => {
    const used: Used = new Map(nodes.map((n) => [n.id, Array(n.count).fill(0)]));
    const placements: Placement[] = [];
    const unplaced: string[] = [];
    for (const m of order) {
      const p = place(m, levels.get(m.id)!, used);
      if (!p) {
        unplaced.push(m.id);
        continue;
      }
      placements.push(p);
      for (const g of p.gpus) used.get(g.nodeId)![g.index] += p.perGpuGb;
    }
    return { placements, unplaced, used, missed: placements.filter((p) => p.meetsTarget === false).length };
  };
  type Attempt = ReturnType<typeof attempt>;
  // Better = fewer models left out, then fewer missed targets.
  const better = (a: Attempt, b: Attempt) =>
    a.unplaced.length < b.unplaced.length || (a.unplaced.length === b.unplaced.length && a.missed < b.missed);
  const noWorse = (a: Attempt, b: Attempt) => a.unplaced.every((id) => b.unplaced.includes(id)) && a.missed <= b.missed;

  // Step everyone down together while that helps.
  let levels = new Map(config.models.map((m) => [m.id, 0]));
  let plan = attempt(levels);
  for (let level = 1; level <= MAX_LEVEL && (plan.unplaced.length > 0 || plan.missed > 0); level++) {
    const next = new Map(config.models.map((m) => [m.id, level]));
    const trial = attempt(next);
    if (better(trial, plan)) {
      plan = trial;
      levels = next;
    }
  }

  // Then win back quality, smallest model first, as long as nothing gets worse.
  for (let improved = true; improved;) {
    improved = false;
    for (const m of [...order].reverse()) {
      const level = levels.get(m.id)!;
      if (m.quant !== "auto" || level === 0) continue;
      const next = new Map(levels).set(m.id, level - 1);
      const trial = attempt(next);
      if (noWorse(trial, plan)) {
        plan = trial;
        levels = next;
        improved = true;
      }
    }
  }

  const usedOn = (nodeId: string) => plan.used.get(nodeId)!;
  const idleIn = (node: Node, exclude: number[] = []) =>
    usedOn(node.id).flatMap((u, i) => (u === 0 && !exclude.includes(i) ? [i] : []));
  const nodeOf = (id: string) => nodes.find((n) => n.id === id)!;
  const modelOf = (id: string) => config.models.find((m) => m.id === id)!;

  // Idle GPUs, step 1: single-GPU models take a second, unused GPU in their node (tensor parallel 2).
  for (const m of order) {
    const i = plan.placements.findIndex((p) => p.modelId === m.id);
    const p = plan.placements[i];
    // Embedding servers (TEI) have no tensor parallelism; pinned models keep the user's choice.
    if (!p || p.tp !== 1 || p.pp !== 1 || p.copies.length !== 1 || isEncoderTask(m.model) || m.pin?.tp || m.pin?.gpus?.length)
      continue;
    const own = p.gpus[0];
    const node = nodeOf(own.nodeId);
    const idle = idleIn(node, [own.index])[0];
    if (idle == null) continue;
    const r = need(m, node.gpu, p.quant, 2, 1, node.link, m.users);
    const used = usedOn(node.id);
    if (r.out.memory_status === "insufficient" || used[own.index] - p.perGpuGb + r.perGpu > r.usable) continue;
    const spread = placement(m, r, node, p.quant, 2, 1, [[own, { nodeId: node.id, index: idle }]]);
    used[own.index] += spread.perGpuGb - p.perGpuGb;
    used[idle] += spread.perGpuGb;
    plan.placements[i] = spread;
  }

  // Idle GPUs, step 2: extra copies for the model whose users gain the most speed from splitting the load.
  const speedOf = (p: Placement) => p.perUserTps ?? p.itemsPerSecond ?? 0;
  for (;;) {
    let best: { i: number; gain: number; next: Placement } | null = null;
    plan.placements.forEach((p, i) => {
      const m = modelOf(p.modelId);
      if (p.pp !== 1 || m.pin?.gpus?.length || m.pin?.copies || p.copies.length >= Math.min(MAX_COPIES, m.users)) return;
      const node = nodeOf(p.copies[0][0].nodeId);
      const host = nodes.find((n) => n.gpuKey === node.gpuKey && idleIn(n).length >= p.tp);
      if (!host) return;
      const r = need(m, node.gpu, p.quant, p.tp, 1, node.link, Math.ceil(m.users / (p.copies.length + 1)));
      if (r.out.memory_status === "insufficient") return;
      const group = idleIn(host)
        .slice(0, p.tp)
        .map((index) => ({ nodeId: host.id, index }));
      const next = placement(m, r, node, p.quant, p.tp, 1, [...p.copies, group]);
      const gain = speedOf(next) / Math.max(1e-9, speedOf(p));
      if (gain > 1.05 && (!best || gain > best.gain)) best = { i, gain, next };
    });
    if (!best) break;
    const { i, next } = best as { i: number; next: Placement };
    // Every copy's GPUs now hold the footprint of a copy serving fewer users.
    const old = plan.placements[i];
    for (const g of old.gpus) usedOn(g.nodeId)[g.index] -= old.perGpuGb;
    for (const g of next.gpus) usedOn(g.nodeId)[g.index] += next.perGpuGb;
    plan.placements[i] = next;
  }

  function whyUnplaced(m: PoolModel) {
    if (nodes.length === 0) return "Add a node with GPUs first.";
    const pin = pinsOf(m);
    if ((pin.copies ?? 1) > 1) return `${pin.copies} copies don't fit; lower the copies or set them back to Auto.`;
    if (pin.node) {
      const k = pin.tp ?? Math.max(...TP_SIZES.filter((t) => t <= pin.node!.count));
      const r = need(m, pin.node.gpu, quantFor(m, pin.node.gpu, MAX_LEVEL), k, 1, pin.node.link, m.users);
      return `Needs about ${Math.round(r.perGpu)} GB on each of ${k} GPU${k > 1 ? "s" : ""} in node ${nodes.indexOf(pin.node) + 1}; free up GPUs there or set the GPUs back to Auto.`;
    }
    // Compare the smallest footprint we tried against the roomiest node.
    const roomiest = nodes
      .map((n) => ({ n, free: usedOn(n.id).reduce((s, u) => s + (usableByGpu.get(n.gpuKey) ?? n.gpu.memory) - u, 0) }))
      .sort((a, b) => b.free - a.free)[0];
    const k = Math.max(...TP_SIZES.filter((t) => t <= roomiest.n.count));
    let smallest = { gb: Infinity, quant: "" };
    for (let level = 0; level <= MAX_LEVEL; level++) {
      const quant = quantFor(m, roomiest.n.gpu, level);
      const gb = need(m, roomiest.n.gpu, quant, k, 1, roomiest.n.link, m.users).perGpu * k;
      if (gb < smallest.gb) smallest = { gb, quant };
    }
    return `Needs about ${Math.round(smallest.gb).toLocaleString()} GB even at ${smallest.quant.toUpperCase()}; the roomiest node has ${Math.round(roomiest.free)} GB free.`;
  }

  return {
    placements: plan.placements,
    unplaced: plan.unplaced.map((modelId) => ({ modelId, reason: whyUnplaced(modelOf(modelId)) })),
    totalTps: round(
      plan.placements.reduce((s, p) => s + (p.throughputTps ?? 0), 0),
      1,
    ),
    gpus: nodes.flatMap((n) =>
      usedOn(n.id).map((u, index) => ({
        nodeId: n.id,
        index,
        memory: n.gpu.memory,
        // GPU types no model evaluated yet fall back to the common 90% runtime reserve.
        usable: round(usableByGpu.get(n.gpuKey) ?? n.gpu.memory * 0.9),
        used: round(u),
      })),
    ),
  };
}
