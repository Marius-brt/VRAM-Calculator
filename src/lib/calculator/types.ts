export type CalcMode = "inference" | "finetuning";

export type FinetuningMethod = "full" | "lora" | "qlora";

export type AttentionStructure = "MHA" | "GQA" | "MLA";

export type KvCacheQuantization = "auto" | "fp16" | "fp8" | "int8" | "int4";

export type FrameworkPreset =
  | "generic"
  | "vllm"
  | "sglang"
  | "tensorrt_llm"
  | "llama_cpp"
  | "ollama"
  | "hf_transformers"
  | "tei"
  | "unsloth"
  | "hf_trainer"
  | "torchtune";

export type ModelTask = "generation" | "embedding" | "reranking";

/**
 * How draft tokens are proposed: a separate small model, an EAGLE-3 head, the model's own multi-token-prediction
 * layers, a DFlash block-diffusion drafter, or lookups in the text so far (n-gram, suffix tree).
 */
export type SpecMethod = "draft" | "eagle3" | "mtp" | "dflash" | "ngram" | "suffix";

/** Model architecture as read from its Hugging Face config.json (see hf.ts). */
export interface ModelVariant {
  slug?: string;
  name?: string;
  provider?: string;
  family?: string;
  hf_repo_id?: string;
  /** Embedding and reranking models run one forward pass per input; absent means text generation. */
  task?: ModelTask;
  modality?: string;
  release_date?: string;
  /** Hugging Face downloads over the last 30 days and likes, used to rank models. */
  downloads?: number;
  likes?: number;
  /** 1 for catalog entries, 0 (or absent) for imported and custom models. */
  is_preset?: number;

  /** Total parameters, billions. */
  num_of_params?: number;
  /** Mixture of experts: parameters used per token, billions. */
  active_params?: number;
  architecture?: "dense" | "moe";
  context_length?: number;
  num_of_layers?: number;
  hidden_size?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  attention_head_dim?: number;
  vocab_size?: number;
  ffn_intermediate_size?: number;
  tie_word_embeddings?: boolean;
  attention_structure?: AttentionStructure | Lowercase<AttentionStructure>;
  /** MLA: cached latent size per token and layer (kv_lora_rank + rope dim). */
  kv_latent_dim?: number;
  num_experts?: number;
  /** Experts used per token, including shared experts. */
  num_active_experts?: number;
  num_of_shared_experts?: number;
  moe_ffn_intermediate_size?: number;

  // Layers by cache type; anything not listed is a full-attention layer.
  /** Layers whose KV cache grows with the whole context. */
  kv_layers?: number;
  /** Sliding-window layers keep at most `sliding_window_size` tokens. */
  sliding_layers?: number;
  sliding_window_size?: number;
  /** Linear attention / SSM / convolution layers keep a fixed state per sequence. */
  state_layers?: number;
  state_bytes_per_layer?: number;
}

export interface GpuSpec {
  key: string;
  label: string;
  vendor: "nvidia" | "amd" | "apple" | "intel" | "custom";
  type: "discrete" | "apu" | "apple_silicon";
  category?: "consumer" | "workstation" | "datacenter";
  memory: number; // GB
  bandwidth_gbs: number; // GB/s
  /** Dense FP16/BF16 tensor TFLOPS (FP32 accumulate). */
  fp16_tflops: number;
  /** Native FP8 / FP4 tensor cores. */
  fp8?: boolean;
  fp4?: boolean;
  tdp_watts?: number;
  /** Typical rental price per GPU-hour, null when it isn't commonly rented. */
  hourly_price_usd?: number | null;
  released?: boolean;
}

export interface CalculatorInput {
  calc_mode: CalcMode;
  gpu_key: string;
  model_variant: ModelVariant;
  quantization?: string;
  kv_cache_quantization?: KvCacheQuantization;
  /** Sequences per request (usually 1). */
  batch_size: number;
  /** Context length in tokens (prompt + output). */
  seq_length: number;
  concurrent_users?: number;
  /** All GPUs in the setup; any beyond tp × pp serve extra copies of the model. */
  num_gpus?: number;
  framework_preset?: FrameworkPreset;

  // Parallelism
  num_nodes?: number;
  gpus_per_node?: number;
  tp_degree?: number;
  pp_degree?: number;
  interconnect_type?: string;
  inter_node_interconnect?: string;

  // Custom hardware overrides
  custom_vram?: number;
  custom_bandwidth?: number;
  custom_tflops?: number;

  // Fine-tuning
  finetuning_method?: FinetuningMethod;
  fine_tuning_quantization?: string;
  lora_rank?: number;
  gradient_accumulation_steps?: number;
  num_samples?: number;
  num_epochs?: number;

  // Optimizations
  enable_offloading?: boolean;
  percentage_offload?: number;
  offload_kv_cache?: boolean;
  enable_prefix_caching?: boolean;
  shared_prefix_ratio?: number; // 0.0 - 1.0
  enable_continuous_batching?: boolean;
  optimization_config?: {
    gradient_checkpointing?: boolean;
    flash_attention?: boolean;
    use_8bit_optimizer?: boolean;
    use_paged_optimizer?: boolean;
    zero_stage?: number; // 0, 1, 2, 3
  };

  // Speculative decoding
  enable_speculative_decoding?: boolean;
  speculative_method?: SpecMethod;
  /** Draft model size in billions of parameters (method "draft" only; the others are sized from the model). */
  speculative_draft_model_params?: number;
  speculative_acceptance_rate?: number;
  speculative_draft_tokens?: number;

  // Environment
  carbon_intensity?: number; // g CO2e per kWh
  electricity_price?: number; // USD per kWh
}

export interface MemoryBreakdownItem {
  label: string;
  value: number; // percentage of total
  size_gb: number;
}

export interface CalculatorOutput {
  /** Memory across all GPUs of the setup, GB. */
  vram_usage: number;
  total_gpu_memory: number;
  /** What the runtime lets the model use (total minus the runtime's reserve). */
  usable_gpu_memory: number;
  /** The fullest GPU; fit is judged on it. */
  per_gpu_usage: number;
  per_gpu_usable: number;
  gpu_count: number;
  /** Copies of the model serving users in parallel. */
  replicas: number;
  vram_percentage: number;
  actual_vram_percentage: number;
  memory_status: "okay" | "moderate" | "high" | "very_high" | "insufficient" | "error";
  memory_details: Record<string, number | undefined>;
  memory_breakdown: MemoryBreakdownItem[];
  offloaded_memory: number;

  // Generation speed (null when it doesn't fit or for non-generation models)
  estimated_latency_tps: number | null; // one request alone
  per_user_tps: number | null; // each user under the configured load
  estimated_throughput_tps: number | null; // all users combined
  ms_per_token: number | null;
  ttft_ms: number | null;
  /** How many users fit before the KV cache runs out, and what's left. */
  concurrency?: {
    maxUsers: number;
    kvPerUserGb: number;
    kvFreeGb: number;
    requestsPerSecond: number;
    preemptionRisk: "none" | "moderate" | "high";
  };

  // Fine-tuning
  training_tps?: number | null;
  samples_per_second?: number | null;
  steps_per_second?: number | null;
  total_training_time_hours?: number | null;

  // Embedding & reranking models
  task?: ModelTask;
  items_per_second?: number;
  tokens_per_second_processed?: number;
  batch_latency_ms?: number;
  single_latency_ms?: number;

  // Energy & cost
  estimated_power_draw: number;
  daily_kwh: number;
  monthly_co2_kg: number;
  electricity_cost_month: number;
  hourly_cost_usd: number | null;
  monthly_cost_usd: number | null;

  /** Caveats about the inputs, e.g. a context longer than the model supports. */
  warnings: string[];
}
