import type { ModelVariant } from "./types";

const FLOAT_DTYPES = /^(BF16|F16|F32|F64|F8)/;

/** Hugging Face org → display name for the provider. */
export const PROVIDERS: Record<string, string> = {
  "meta-llama": "Meta",
  Qwen: "Alibaba",
  "deepseek-ai": "DeepSeek",
  mistralai: "Mistral AI",
  google: "Google",
  microsoft: "Microsoft",
  openai: "OpenAI",
  nvidia: "NVIDIA",
  "ibm-granite": "IBM",
  allenai: "Ai2",
  moonshotai: "Moonshot AI",
  "zai-org": "Z.ai",
  THUDM: "Z.ai",
  MiniMaxAI: "MiniMax",
  tencent: "Tencent",
  baidu: "Baidu",
  "stepfun-ai": "StepFun",
  XiaomiMiMo: "Xiaomi",
  LiquidAI: "Liquid AI",
  HuggingFaceTB: "Hugging Face",
  CohereLabs: "Cohere",
  inclusionAI: "Ant Group",
  "ServiceNow-AI": "ServiceNow",
  "01-ai": "01.AI",
  internlm: "Shanghai AI Lab",
  openbmb: "OpenBMB",
  tiiuae: "TII",
  "arcee-ai": "Arcee",
  NousResearch: "Nous Research",
  "swiss-ai": "Swiss AI",
  "LGAI-EXAONE": "LG AI Research",
  upstage: "Upstage",
  "ByteDance-Seed": "ByteDance",
  ai21labs: "AI21 Labs",
  "meituan-longcat": "Meituan",
  kakaocorp: "Kakao",
  "naver-hyperclovax": "NAVER",
  sarvamai: "Sarvam AI",
  "utter-project": "EuroLLM",
  speakleash: "SpeakLeash",
  stabilityai: "Stability AI",
  bigcode: "BigCode",
  Unbabel: "Unbabel",
  Skywork: "Skywork",
  thinkingmachines: "Thinking Machines",
  // Embedding and reranking model publishers.
  BAAI: "BAAI",
  "Alibaba-NLP": "Alibaba",
  "nomic-ai": "Nomic AI",
  "mixedbread-ai": "Mixedbread",
  Snowflake: "Snowflake",
  jinaai: "Jina AI",
  intfloat: "Microsoft",
  "sentence-transformers": "Sentence Transformers",
  "cross-encoder": "Sentence Transformers",
};

/** Labs whose chat models the catalog build scans. */
export const CHAT_ORGS = Object.keys(PROVIDERS).slice(0, Object.keys(PROVIDERS).indexOf("BAAI"));

export interface HfInfo {
  safetensors?: { parameters?: Record<string, number>; total?: number };
  createdAt?: string;
  downloads?: number;
  likes?: number;
}

/** Fixed per-sequence state of one recurrent layer (linear attention, Mamba/SSM, short convolution), in bytes. */
function stateBytes(cfg: any, hidden: number) {
  if (cfg.linear_num_value_heads && cfg.linear_key_head_dim && cfg.linear_value_head_dim) {
    return cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim * 4; // fp32 state
  }
  const mambaHeads = cfg.mamba_n_heads ?? cfg.mamba_num_heads;
  const mambaHeadDim = cfg.mamba_d_head ?? cfg.mamba_head_dim;
  const ssm = cfg.mamba_d_state ?? cfg.ssm_state_size ?? cfg.state_size;
  if (mambaHeads && mambaHeadDim && ssm) return mambaHeads * mambaHeadDim * ssm * 4;
  if (cfg.conv_L_cache) return cfg.conv_L_cache * hidden * 2; // LFM2 short convolutions
  return hidden * 128 * 4;
}

/**
 * Turn a Hugging Face config.json into a model spec for the engine.
 * `info` is the model's API record (parameter count, dates, popularity), when available.
 */
export function specFromConfig(repoId: string, raw: any, info: HfInfo = {}): ModelVariant {
  // Multimodal models (Qwen3.5+, Llama 4, Gemma 3...) keep the language model under text_config; omni models
  // (Qwen3-Omni) one level deeper, in the "thinker" that reads the inputs.
  const cfg = { ...raw, ...(raw.thinker_config?.text_config ?? raw.text_config ?? raw.llm_config ?? raw.language_config ?? {}) };
  const [org, name = repoId] = repoId.split("/");

  const hidden = cfg.hidden_size || cfg.d_model || cfg.n_embd || cfg.dim || 4096;
  const layers = cfg.num_hidden_layers || cfg.n_layer || cfg.num_layers || 32;
  const heads = cfg.num_attention_heads || cfg.n_head || 32;
  const kvHeads = cfg.num_key_value_heads ?? cfg.num_attention_groups ?? cfg.multi_query_group_num ?? heads;
  const headDim = cfg.head_dim || Math.round(hidden / heads);
  const vocab = cfg.vocab_size || 32000;
  const ffn = cfg.intermediate_size || cfg.ffn_dim || Math.round(hidden * 3.5);

  // Mixture of experts.
  const experts = cfg.num_experts || cfg.num_local_experts || cfg.n_routed_experts || cfg.moe_num_experts || null;
  const perToken = cfg.num_experts_per_tok || cfg.num_selected_experts || cfg.moe_top_k || null;
  const shared = cfg.n_shared_experts ?? (cfg.shared_expert_intermediate_size || cfg.share_expert_dim ? 1 : 0);
  const moeFfn = cfg.moe_intermediate_size || ffn;
  const moeLayerList = typeof cfg.moe_layers_enum === "string" ? cfg.moe_layers_enum.split(",").length : null;
  const denseLayers =
    cfg.first_k_dense_replace ?? (moeLayerList != null ? layers - moeLayerList : (cfg.mlp_only_layers?.length ?? 0));

  // Which layers keep a KV cache that grows with context, a sliding window, or a fixed recurrent state.
  const types: string[] = Array.isArray(cfg.layer_types)
    ? cfg.layer_types
    : typeof cfg.hybrid_override_pattern === "string" // Nemotron-H: M = Mamba, * = attention, - = MLP only
      ? [...cfg.hybrid_override_pattern].map((c) => (c === "M" ? "mamba" : c === "*" ? "full_attention" : "mlp"))
      : [];
  const count = (re: RegExp) => types.filter((t) => re.test(t)).length;
  let stateLayers = count(/linear|conv|mamba|ssm|recurrent/);
  let slidingLayers = count(/sliding|local/);
  let kvLayers = types.length ? count(/attention/) - slidingLayers - count(/linear/) : layers;
  if (!types.length && cfg.full_attention_interval) {
    kvLayers = Math.round(layers / cfg.full_attention_interval);
    stateLayers = layers - kvLayers;
  }
  // Gemma 2/3: every Nth layer is global, the rest use a sliding window.
  const pattern = cfg.sliding_window_pattern ?? (cfg.model_type === "gemma2" ? 2 : null);
  if (!types.length && cfg.sliding_window && pattern) {
    kvLayers = Math.floor(layers / pattern);
    slidingLayers = layers - kvLayers;
  }
  if (!types.length && !pattern && cfg.sliding_window && cfg.use_sliding_window !== false && cfg.max_window_layers != null) {
    slidingLayers = Math.max(0, layers - cfg.max_window_layers);
    kvLayers = layers - slidingLayers;
  }

  const mla = cfg.kv_lora_rank != null;

  // Parameter count: exact from safetensors unless weights are packed into integers (GPTQ/AWQ/INT4 repos).
  const st = info.safetensors;
  const packed = Object.entries(st?.parameters ?? {})
    .filter(([dtype]) => !FLOAT_DTYPES.test(dtype))
    .reduce((s, [, n]) => s + n, 0);
  const exact = st?.total && packed < st.total * 0.01 ? st.total : null;
  // BERT-style encoders have plain two-matrix MLPs and no output head; decoders use gated MLPs.
  const encoder = /bert|roberta|deberta|electra|mpnet/i.test(cfg.model_type ?? "");
  const attn = hidden * heads * headDim * 2 + hidden * kvHeads * headDim * 2;
  const expertParams = 3 * hidden * moeFfn;
  const mlp = experts ? (experts + shared) * expertParams : (encoder ? 2 : 3) * hidden * ffn;
  const estimated = layers * (attn + mlp) + vocab * hidden * (encoder || cfg.tie_word_embeddings ? 1 : 2);
  const total = exact ?? estimated;
  // Active parameters per token, used for speed: everything except the experts a token skips, capped by a
  // bottom-up count (attention + used experts + output head) for designs with large lookup tables.
  const topDown = experts && perToken ? total - (experts - perToken) * expertParams * (layers - denseLayers) : null;
  const bottomUp = experts && perToken ? (layers * (attn + (perToken + shared) * expertParams) + vocab * hidden) * 1.1 : null;
  const active = topDown && bottomUp ? Math.min(topDown, bottomUp) : null;
  // Prefer the size in the name ("122B-A10B") when present; it is what people quote.
  const nameTotal = name.match(/(?:^|[-_])(\d+(?:\.\d+)?)[Bb](?:[-_]|$)/)?.[1];
  const nameActive = name.match(/[-_]A(\d+(?:\.\d+)?)B/i)?.[1];

  return {
    slug: name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    name,
    family: name.match(/^(.*?)(?:[-_ ]\d+(?:\.\d+)?[BbMm]\b|$)/)?.[1] || name,
    provider: PROVIDERS[org] ?? org,
    hf_repo_id: repoId,
    task: "generation",
    modality: raw.vision_config ? "multimodal" : "text",
    release_date: info.createdAt?.slice(0, 10),
    downloads: info.downloads,
    likes: info.likes,
    // Exact counts include vision towers and other weights that get loaded; the name is the fallback.
    num_of_params: Number(exact ? (exact / 1e9).toFixed(2) : (nameTotal ?? (total / 1e9).toFixed(2))),
    architecture: experts ? "moe" : "dense",
    context_length: cfg.max_position_embeddings || cfg.max_sequence_length || cfg.seq_length || 4096,
    num_of_layers: layers,
    hidden_size: hidden,
    num_attention_heads: heads,
    num_key_value_heads: kvHeads,
    attention_head_dim: headDim,
    vocab_size: vocab,
    ffn_intermediate_size: ffn,
    tie_word_embeddings: !!cfg.tie_word_embeddings,
    attention_structure: mla ? "MLA" : kvHeads < heads ? "GQA" : "MHA",
    kv_latent_dim: mla ? cfg.kv_lora_rank + (cfg.qk_rope_head_dim || 64) : undefined,
    kv_layers: Math.max(0, kvLayers),
    sliding_layers: slidingLayers || undefined,
    sliding_window_size: slidingLayers ? cfg.sliding_window : undefined,
    state_layers: stateLayers || undefined,
    state_bytes_per_layer: stateLayers ? stateBytes(cfg, hidden) : undefined,
    ...(experts && {
      num_experts: experts,
      num_active_experts: (perToken || 2) + shared,
      num_of_shared_experts: shared || undefined,
      moe_ffn_intermediate_size: moeFfn,
      active_params: Number(nameActive ?? (active! / 1e9).toFixed(2)),
    }),
  };
}

/** Fetch a model's config and API record from Hugging Face (works from the browser and from Node). */
export async function importFromHuggingFace(repoId: string, token?: string): Promise<ModelVariant> {
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(`https://huggingface.co/${repoId}/resolve/main/config.json`, { headers });
  if (!res.ok) {
    throw new Error(
      res.status === 401 || res.status === 403
        ? `${repoId} is gated: accept its license on Hugging Face and add an access token.`
        : `Couldn't read config.json for ${repoId} (${res.status}).`,
    );
  }
  const config = await res.json();
  const fields = ["safetensors", "createdAt", "downloads", "likes"].map((f) => `expand%5B%5D=${f}`).join("&");
  const info = await fetch(`https://huggingface.co/api/models/${repoId}?${fields}`, { headers })
    .then((r) => (r.ok ? r.json() : {}))
    .catch(() => ({}));
  return specFromConfig(repoId, config, info);
}
