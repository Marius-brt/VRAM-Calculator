/**
 * Builds src/lib/data/models.json from Hugging Face: config.json for the architecture, the model API for
 * parameter counts, release date and popularity. Re-run to refresh: bun scripts/build-catalog.ts
 * Optional: HF_TOKEN=... to read gated configs directly instead of through public mirrors.
 */
import { CHAT_ORGS, specFromConfig } from "../src/lib/calculator/hf";
import type { ModelTask, ModelVariant } from "../src/lib/calculator/types";

const OUT = new URL("../src/lib/data/models.json", import.meta.url);
const headers: Record<string, string> = process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {};
const previous: ModelVariant[] = await Bun.file(OUT)
  .json()
  .catch(() => []);

// Quantized copies, adapters, research checkpoints and non-chat models.
const SKIP =
  /(fp8|fp4|awq|gptq|gguf|int4|int8|nvfp4|mxfp4|mlx|bnb|4bit|8bit|w4a16|w8a8|w4a4|onnx|eagle|dflash|dspark|-mtp|guard|reward|safety|prm\b|embed|rerank|retriev|tts|asr|audio|speech|ocr|-base\b|-pt\b|midtrain|stage\d|-sft\b|-rl\b|rloo|teacher|draft|lora|adapter|qat|unquantized|bench)/i;
const CHAT_PIPELINES = new Set(["text-generation", "image-text-to-text"]);

// Always included, whatever their current rank.
const CLASSICS = [
  "meta-llama/Llama-3.1-8B-Instruct",
  "meta-llama/Llama-3.1-70B-Instruct",
  "meta-llama/Llama-3.1-405B-Instruct",
  "meta-llama/Llama-3.3-70B-Instruct",
  "meta-llama/Llama-3.2-3B-Instruct",
  "meta-llama/Llama-4-Scout-17B-16E-Instruct",
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
  "deepseek-ai/DeepSeek-V3",
  "deepseek-ai/DeepSeek-R1",
  "mistralai/Mixtral-8x7B-Instruct-v0.1",
  "mistralai/Mistral-7B-Instruct-v0.3",
  "Qwen/Qwen2.5-7B-Instruct",
  "Qwen/Qwen2.5-72B-Instruct",
  "Qwen/Qwen3-8B",
  "Qwen/Qwen3-32B",
  "Qwen/Qwen3-30B-A3B",
  "Qwen/Qwen3-235B-A22B",
  "Qwen/Qwen3.5-122B-A10B",
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
  "google/gemma-3-27b-it",
  "microsoft/phi-4",
  "moonshotai/Kimi-K2-Instruct",
  "zai-org/GLM-4.6",
  "MiniMaxAI/MiniMax-M2",
];

// Embedding and reranking models (served by Text Embeddings Inference).
const ENCODERS: Record<string, ModelTask> = {
  "BAAI/bge-m3": "embedding",
  "BAAI/bge-large-en-v1.5": "embedding",
  "BAAI/bge-base-en-v1.5": "embedding",
  "BAAI/bge-small-en-v1.5": "embedding",
  "Qwen/Qwen3-Embedding-0.6B": "embedding",
  "Qwen/Qwen3-Embedding-4B": "embedding",
  "Qwen/Qwen3-Embedding-8B": "embedding",
  "Alibaba-NLP/gte-Qwen2-7B-instruct": "embedding",
  "nomic-ai/nomic-embed-text-v1.5": "embedding",
  "mixedbread-ai/mxbai-embed-large-v1": "embedding",
  "Snowflake/snowflake-arctic-embed-l-v2.0": "embedding",
  "jinaai/jina-embeddings-v3": "embedding",
  "intfloat/multilingual-e5-large-instruct": "embedding",
  "intfloat/e5-mistral-7b-instruct": "embedding",
  "nvidia/NV-Embed-v2": "embedding",
  "google/embeddinggemma-300m": "embedding",
  "sentence-transformers/all-MiniLM-L6-v2": "embedding",
  "BAAI/bge-reranker-v2-m3": "reranking",
  "BAAI/bge-reranker-large": "reranking",
  "BAAI/bge-reranker-base": "reranking",
  "BAAI/bge-reranker-v2-gemma": "reranking",
  "Qwen/Qwen3-Reranker-0.6B": "reranking",
  "Qwen/Qwen3-Reranker-4B": "reranking",
  "Qwen/Qwen3-Reranker-8B": "reranking",
  "jinaai/jina-reranker-v2-base-multilingual": "reranking",
  "mixedbread-ai/mxbai-rerank-large-v1": "reranking",
  "cross-encoder/ms-marco-MiniLM-L-6-v2": "reranking",
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const get = async (url: string) => {
  for (let attempt = 0; attempt < 5; attempt++) {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(30_000) }).catch(() => null);
    if (res?.ok) return res.json();
    if (res && res.status !== 429 && res.status < 500) return null;
    await sleep((res?.status === 429 ? 10_000 : 2_000) * (attempt + 1)); // back off harder on rate limits
  }
  return null;
};

async function discover(org: string) {
  const since = Date.now() - 180 * 864e5;
  const lists = await Promise.all(
    ["downloads", "createdAt"].map((sort) =>
      get(`https://huggingface.co/api/models?author=${org}&sort=${sort}&direction=-1&limit=${sort === "downloads" ? 40 : 25}`),
    ),
  );
  const found = lists
    .flat()
    .filter(Boolean)
    .filter((m: any) => CHAT_PIPELINES.has(m.pipeline_tag) && !SKIP.test(m.id.split("/")[1]))
    .filter((m: any) => m.downloads >= 3000 || (Date.parse(m.createdAt) > since && m.downloads >= 500))
    .map((m: any) => m.id as string);
  if (lists.every(Boolean)) return found;
  // Hugging Face didn't answer: keep the org's models from the current catalog rather than dropping them.
  console.warn(`  couldn't list ${org}; keeping its current models`);
  const kept = previous.filter((m) => (m.task ?? "generation") === "generation" && m.hf_repo_id?.split("/")[0] === org);
  return [...found, ...kept.map((m) => m.hf_repo_id!)];
}

/** Gated repos (Llama, Gemma...) expose config.json only after accepting a license; public mirrors carry the same file. */
async function config(repo: string) {
  const name = repo.split("/")[1];
  const mirrors = ["unsloth", "NousResearch"].flatMap((org) => [
    `${org}/${name}`,
    `${org}/Meta-${name}`,
    `${org}/Meta-${name}-bnb-4bit`,
  ]);
  for (const source of [repo, ...mirrors]) {
    const cfg = await get(`https://huggingface.co/${source}/resolve/main/config.json`);
    if (cfg) return cfg;
  }
  return null;
}

async function build(repo: string, task: ModelTask): Promise<ModelVariant | null> {
  const fields = ["safetensors", "createdAt", "downloads", "likes"].map((f) => `expand%5B%5D=${f}`).join("&");
  const [cfg, info] = await Promise.all([config(repo), get(`https://huggingface.co/api/models/${repo}?${fields}`)]);
  if (!cfg || !(cfg.hidden_size || cfg.text_config?.hidden_size || cfg.d_model || cfg.dim)) return null;
  return { ...specFromConfig(repo, cfg, info ?? {}), task, is_preset: 1 };
}

async function pool<T, R>(items: T[], size: number, fn: (item: T) => Promise<R>) {
  const out: R[] = [];
  for (let i = 0; i < items.length; i += size) out.push(...(await Promise.all(items.slice(i, i + size).map(fn))));
  return out;
}

// Listing an org can take seconds, so a few go at once.
const discovered = (await pool(CHAT_ORGS, 4, discover)).flat();
const chat = [...new Set([...CLASSICS, ...discovered])].filter((r) => !(r in ENCODERS));
console.log(`${chat.length} chat models, ${Object.keys(ENCODERS).length} embedding/reranking models`);

const jobs: [string, ModelTask][] = [...chat.map((r) => [r, "generation"] as [string, ModelTask]), ...Object.entries(ENCODERS)];
const built = await pool(jobs, 4, async ([repo, task]) => {
  const spec = await build(repo, task);
  if (!spec) console.warn("  skipped", repo);
  return spec;
});

const seen = new Set<string>();
const models = built
  .filter((m): m is ModelVariant => !!m && !seen.has(m.slug!) && !!seen.add(m.slug!))
  .sort((a, b) => a.provider!.localeCompare(b.provider!) || a.num_of_params! - b.num_of_params!);
// A much smaller result means Hugging Face throttled us; keep the existing catalog.
if (models.length < previous.length * 0.8 && !process.argv.includes("--force")) {
  console.error(`Only ${models.length} models (was ${previous.length}); not overwriting. Re-run later or pass --force.`);
  process.exit(1);
}
await Bun.write(OUT, JSON.stringify(models, null, 1) + "\n");
console.log(`${models.length} models written to ${OUT.pathname}`);
