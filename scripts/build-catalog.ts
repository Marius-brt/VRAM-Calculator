/**
 * Builds src/lib/data/models.json from Hugging Face: config.json for the architecture, the model API for
 * parameter counts, release date and popularity. Re-run to refresh: bun scripts/build-catalog.ts
 * Optional: HF_TOKEN=... to read gated configs directly instead of through public mirrors.
 */
import { CHAT_ORGS, PROVIDERS, specFromConfig } from "../src/lib/calculator/hf";
import type { ModelTask, ModelVariant } from "../src/lib/calculator/types";

const OUT = new URL("../src/lib/data/models.json", import.meta.url);
// HF_TOKEN, or the token `hf auth login` saved: signed-in clients get a higher rate limit than anonymous ones.
const token =
  process.env.HF_TOKEN ||
  (await Bun.file(`${process.env.HF_HOME ?? `${process.env.HOME}/.cache/huggingface`}/token`)
    .text()
    .then((t) => t.trim())
    .catch(() => ""));
const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
const previous: ModelVariant[] = await Bun.file(OUT)
  .json()
  .catch(() => []);

// Quantized copies, adapters, research checkpoints and non-chat models.
const SKIP =
  /(fp8|fp4|awq|gptq|gguf|int4|int8|nvfp4|mxfp4|mlx|bnb|4bit|8bit|w4a16|w8a8|w4a4|onnx|eagle|dflash|dspark|-mtp|guard|reward|safety|prm\b|embed|rerank|retriev|tts|asr|audio|speech|ocr|-base\b|midtrain|stage\d|-sft\b|-rl\b|rloo|teacher|draft|lora|adapter|qat|unquantized|bench|random|dummy|-assistant\b|shield|-bf16\b|\bt5\b)/i;
// Lowercase "-pt" is a pre-trained base (gemma-3-4b-pt); Baidu's "-PT" means PyTorch weights (ERNIE-4.5-21B-A3B-PT).
const skip = (name: string) => SKIP.test(name) || /-pt\b/.test(name);
// "any-to-any": models that also take audio or video, such as Gemma 4 E2B/E4B/12B and Qwen3-Omni. Untagged repos
// count too (most of Mistral's): the config check in build() turns away anything that isn't a language model.
const CHAT_PIPELINES = new Set(["text-generation", "image-text-to-text", "any-to-any"]);
const chatTag = (tag?: string | null) => !tag || CHAT_PIPELINES.has(tag);

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
/** JSON as Python writes it: some configs hold Infinity or NaN. Anything else unreadable counts as missing. */
const parse = (text: string) => {
  try {
    return JSON.parse(text.replace(/\bNaN\b/g, "null").replace(/\bInfinity\b/g, "1e999"));
  } catch {
    return null;
  }
};
const get = async (url: string) => {
  for (let attempt = 0; attempt < 5; attempt++) {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(30_000) }).catch(() => null);
    if (res?.ok) return parse(await res.text());
    if (res && res.status !== 429 && res.status < 500) return null;
    await sleep((res?.status === 429 ? 10_000 : 2_000) * (attempt + 1)); // back off harder on rate limits
  }
  return null;
};

// Hugging Face allows 500 API calls per 5 minutes without a token, so each model's record comes with the listing
// that found it rather than from a call of its own.
const FIELDS = ["safetensors", "createdAt", "downloads", "likes", "pipeline_tag"].map((f) => `expand%5B%5D=${f}`).join("&");
const infos = new Map<string, Promise<any>>();
/** A model's API record, fetched once (models.dev candidates are checked with it, then built from it). */
const info = (repo: string) => {
  if (!infos.has(repo)) infos.set(repo, get(`https://huggingface.co/api/models/${repo}?${FIELDS}`));
  return infos.get(repo)!;
};

async function discover(org: string) {
  const since = Date.now() - 180 * 864e5;
  const lists = await Promise.all(
    ["downloads", "createdAt"].map((sort) =>
      // Generous limits: quantized copies and adapters fill many slots before SKIP drops them.
      get(
        `https://huggingface.co/api/models?author=${org}&sort=${sort}&direction=-1&limit=${sort === "downloads" ? 100 : 50}&${FIELDS}`,
      ),
    ),
  );
  for (const m of lists.flat().filter(Boolean)) if (!infos.has(m.id)) infos.set(m.id, Promise.resolve(m));
  const found = lists
    .flat()
    .filter(Boolean)
    .filter((m: any) => chatTag(m.pipeline_tag) && !skip(m.id.split("/")[1]))
    .filter((m: any) => m.downloads >= 3000 || (Date.parse(m.createdAt) > since && m.downloads >= 500))
    .map((m: any) => m.id as string);
  if (lists.every(Boolean)) return found;
  // Hugging Face didn't answer: keep the org's models from the current catalog rather than dropping them.
  console.warn(`  couldn't list ${org}; keeping its current models`);
  const kept = previous.filter((m) => (m.task ?? "generation") === "generation" && m.hf_repo_id?.split("/")[0] === org);
  return [...found, ...kept.map((m) => m.hf_repo_id!)];
}

/**
 * Open-weight models that inference providers serve, from models.dev. Many providers name them by their Hugging Face
 * repo (google/gemma-4-E4B-it); the ones Hugging Face knows as chat models are kept, under their exact repo name.
 * Only ids that can be repos are checked: those of the "huggingface" provider, and those under a lab's own org name
 * (others use their own names, like deepseek/deepseek-chat). Models the org listings already found are skipped.
 */
async function fromModelsDev(found: string[]) {
  const data = await get("https://models.dev/api.json");
  if (!data) return (console.warn("  couldn't read models.dev"), []);
  const labs = new Set(Object.keys(PROVIDERS).map((o) => o.toLowerCase()));
  const seen = new Set(found.map((id) => id.toLowerCase()));
  const ids = new Map<string, string>();
  for (const [name, provider] of Object.entries<any>(data))
    for (const m of Object.values<any>(provider.models ?? {})) {
      const [org, repo] = m.id.split("/");
      if (!m.open_weights || !/^[\w.-]+\/[\w.-]+$/.test(m.id) || skip(repo) || seen.has(m.id.toLowerCase())) continue;
      if (name === "huggingface" || labs.has(org.toLowerCase())) ids.set(m.id.toLowerCase(), m.id);
    }
  const checked = await pool([...ids.values()], 8, info);
  return checked
    .filter((i) => i?.id && chatTag(i.pipeline_tag) && !skip(i.id.split("/")[1]) && i.downloads >= 500)
    .map((i) => i.id as string);
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
  const [cfg, record] = await Promise.all([config(repo), info(repo)]);
  // Configs without a transformer's shape (diffusion, speech-only...) can't be sized.
  const sized = (c: any) => c && (c.hidden_size || c.d_model || c.dim || c.n_embd);
  const parts = [cfg, cfg?.text_config, cfg?.thinker_config?.text_config, cfg?.llm_config, cfg?.language_config];
  if (!cfg || !parts.some(sized)) return null;
  // An untagged repo counts as a chat model only with a text-generation head (not BERT, ELECTRA, time series...).
  if (
    task === "generation" &&
    !record?.pipeline_tag &&
    !/CausalLM|ConditionalGeneration|LMHeadModel/.test(String(cfg.architectures))
  )
    return null;
  const spec = specFromConfig(repo, cfg, record ?? {});
  // Unsized repos, and chat "models" under 50M parameters (test fixtures), are left out.
  const big = task === "generation" ? (spec.num_of_params ?? 0) >= 0.05 : !!spec.num_of_params;
  return big ? { ...spec, task, is_preset: 1 } : null;
}

async function pool<T, R>(items: T[], size: number, fn: (item: T) => Promise<R>) {
  const out: R[] = [];
  for (let i = 0; i < items.length; i += size) out.push(...(await Promise.all(items.slice(i, i + size).map(fn))));
  return out;
}

// Listing an org can take seconds, so a few go at once.
const discovered = (await pool(CHAT_ORGS, 4, discover)).flat();
const served = await fromModelsDev(discovered);
const chat = [...new Set([...CLASSICS, ...discovered, ...served])].filter((r) => !(r in ENCODERS));
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
