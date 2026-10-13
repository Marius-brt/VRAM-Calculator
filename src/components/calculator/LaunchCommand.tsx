"use client";

import React, { useState } from "react";
import { Check, Copy } from "lucide-react";
import { CalculatorInput, CalculatorOutput, GpuSpec, ModelVariant, SpecMethod } from "@/lib/calculator/types";
import { geometry, isEncoderTask, weightBytes } from "@/lib/calculator/engine";
import { Button } from "@/components/ui/button";
import { Section } from "./controls";

/*
 * Flags checked against vLLM 0.31, SGLang 0.5.21, TensorRT-LLM 1.2.1, llama.cpp master, Ollama, transformers serve,
 * the Unsloth CLI, TRL and torchtune 0.6 (October 2026). Runtimes rename flags often; re-check when they bump.
 */

/** One copy of a model, as its runtime should start it. */
export interface Launch {
  runtime: string;
  quant: string;
  kvQuant?: string;
  tp: number;
  pp: number;
  /** Machines one copy spans; llama.cpp reaches the others over RPC. */
  machines?: number;
  /** Share of each GPU's memory to claim; left to the runtime's default when the copy has its GPUs to itself. */
  memoryFraction?: number;
  /** Concurrent sequences (batch size for embedding servers). */
  users: number;
  contextLength: number;
  /** GPUs this copy may use (CUDA_VISIBLE_DEVICES), or null for all of them. */
  devices: number[] | null;
  /** Index of this copy when several run on one machine (each gets its own port), else null. */
  copy: number | null;
  spec?: { method: SpecMethod; tokens: number };
  /** false turns fused attention kernels off; anything else keeps the runtime's default. */
  flash?: boolean;
  /** Share of the weights kept in CPU memory, and GB of KV cache per copy moved there. */
  offload?: { fraction: number; kvGb: number };
}

const GiB = 1024 ** 3;
// Formats a runtime can't produce at load time (unlike FP8 in vLLM): they need a checkpoint quantized ahead of time.
const PREQUANTIZED: Record<string, string> = { nvfp4: "NVFP4", mxfp4: "MXFP4", awq: "AWQ", gptq: "GPTQ", exl2: "EXL2" };
const isGguf = (quant: string) => /^q\d/.test(quant);
const repoOf = (model: ModelVariant) => model.hf_repo_id || model.slug || model.name || "model";
const join = (...parts: (string | number | false | null | undefined)[]) => parts.filter((p) => p || p === 0).join(" ");

const NAMES: Record<string, string> = {
  vllm: "vLLM",
  sglang: "SGLang",
  tensorrt_llm: "TensorRT-LLM",
  llama_cpp: "llama.cpp",
  ollama: "Ollama",
  hf_transformers: "Transformers",
  tei: "TEI",
};
const PORTS: Record<string, number> = { sglang: 30000, llama_cpp: 8080, ollama: 11434, tei: 3000 };
/** Speculative methods each runtime runs. Ollama only turns MTP on by itself. */
const SPEC_SUPPORT: Record<string, SpecMethod[]> = {
  vllm: ["draft", "eagle3", "mtp", "dflash", "ngram", "suffix"],
  sglang: ["draft", "eagle3", "mtp", "dflash", "ngram"],
  tensorrt_llm: ["draft", "eagle3", "mtp", "ngram"],
  llama_cpp: ["draft", "eagle3", "mtp", "dflash", "ngram"],
};
const DRAFTER: Partial<Record<SpecMethod, string>> = {
  draft: "<draft-model>",
  eagle3: "<eagle3-head>",
  dflash: "<dflash-drafter>",
};
const SPEC_PHRASES: Record<SpecMethod, string> = {
  draft: "a separate draft model",
  eagle3: "EAGLE-3 heads",
  mtp: "MTP",
  dflash: "DFlash",
  ngram: "n-gram lookup",
  suffix: "suffix decoding",
};
const GGUF_KV: Record<string, string> = { int8: "q8_0", fp8: "q8_0", int4: "q4_0" };

/** Shell command for one copy (null for runtimes without a standard server), plus what the user has to fill in. */
export function launch(model: ModelVariant, l: Launch): { cmd: string | null; notes: string[] } {
  const name = repoOf(model);
  const g = geometry(model);
  const notes: string[] = [];
  const runtime = NAMES[l.runtime] ?? l.runtime;
  const gguf = `${name.split("/").pop()}-${l.quant.toUpperCase()}.gguf`;
  const env = l.devices ? `CUDA_VISIBLE_DEVICES=${l.devices.join(",")} ` : "";
  // llama.cpp moved its default port to 9931, so it always gets one.
  const portNum = l.copy != null ? (PORTS[l.runtime] ?? 8000) + l.copy : l.runtime === "llama_cpp" ? 8080 : null;
  const port = portNum && `--port ${portNum}`;

  const spec = l.spec && (SPEC_SUPPORT[l.runtime] ?? []).includes(l.spec.method) ? l.spec : null;
  const k = spec?.tokens ?? 0;
  const drafter = spec && DRAFTER[spec.method];
  // Generic gets no command at all, so no note about what the command leaves out.
  if (l.spec && !spec && l.runtime in NAMES && !(l.runtime === "ollama" && l.spec.method === "mtp"))
    notes.push(`${runtime} doesn't support ${SPEC_PHRASES[l.spec.method]}, so the command leaves speculative decoding out.`);
  if (drafter) {
    const file = l.runtime === "llama_cpp" ? " (as a GGUF file)" : "";
    notes.push(
      spec.method === "draft"
        ? `Replace ${drafter} with a small model that shares the tokenizer${file}, such as the smallest of the same family.`
        : spec.method === "eagle3"
          ? `Replace ${drafter} with an EAGLE-3 head trained for ${name}${file}; search Hugging Face for "${name.split("/").pop()} eagle3".`
          : `Replace ${drafter} with a DFlash drafter for ${name}${file}; z-lab publishes them on Hugging Face.`,
    );
  }
  if (l.spec?.method === "mtp")
    notes.push(
      l.runtime === "ollama"
        ? "Ollama turns MTP on by itself for Gemma 4 and Qwen3.5 on Apple Silicon; there is no setting for it."
        : `MTP only works with models that ship MTP layers (DeepSeek-V3, GLM-4.5 and later, Qwen3.5, MiMo)${l.runtime === "llama_cpp" ? " and a GGUF that keeps them" : ""}.`,
    );

  const off = l.offload;
  const offGb = off ? Math.ceil((weightBytes(g, l.quant) * off.fraction) / (l.tp * l.pp) / GiB) : 0;
  const kvOffGb = off?.kvGb ? Math.ceil(off.kvGb) : 0;
  const gpuLayers = off ? Math.round(g.layers * (1 - off.fraction)) : g.layers;
  if (l.kvQuant && ["int8", "int4"].includes(l.kvQuant) && ["vllm", "sglang", "tensorrt_llm"].includes(l.runtime))
    notes.push(`${runtime} has no ${l.kvQuant.toUpperCase()} KV cache; FP8 is the closest it offers.`);

  let cmd: string | null;
  switch (l.runtime) {
    case "vllm": {
      const method = { draft: "draft_model", eagle3: "eagle3", mtp: "mtp", dflash: "dflash", ngram: "ngram", suffix: "suffix" };
      const config = spec && {
        method: method[spec.method],
        ...(drafter && { model: drafter }),
        num_speculative_tokens: k,
        ...(spec.method === "ngram" && { prompt_lookup_min: 2, prompt_lookup_max: 5 }),
      };
      cmd = join(
        `${env}vllm serve ${isGguf(l.quant) ? `${gguf} --tokenizer ${name}` : name}`,
        `--tensor-parallel-size ${l.tp}`,
        l.pp > 1 && `--pipeline-parallel-size ${l.pp}`,
        l.memoryFraction && `--gpu-memory-utilization ${l.memoryFraction}`,
        `--max-model-len ${l.contextLength}`,
        `--max-num-seqs ${l.users}`,
        ["fp8", "awq", "gptq"].includes(l.quant) && `--quantization ${l.quant}`,
        ["int8", "int4"].includes(l.quant) && "--quantization bitsandbytes",
        l.quant === "fp32" && "--dtype float32",
        l.kvQuant === "fp8" && "--kv-cache-dtype fp8",
        offGb > 0 && `--cpu-offload-gb ${offGb}`,
        kvOffGb > 0 && `--kv-offloading-size ${kvOffGb} --kv-offloading-backend native`,
        config && `--speculative-config '${JSON.stringify(config)}'`,
        // DFlash needs FlashAttention; Triton is the fused-kernel-free fallback.
        spec?.method === "dflash" ? "--attention-backend FLASH_ATTN" : l.flash === false && "--attention-backend TRITON_ATTN",
        port,
      );
      break;
    }
    case "sglang": {
      const algorithm = { draft: "STANDALONE", eagle3: "EAGLE3", mtp: "EAGLE", dflash: "DFLASH", ngram: "NGRAM", suffix: "" };
      const specFlags =
        spec &&
        join(
          `--speculative-algorithm ${algorithm[spec.method]}`,
          drafter && `--speculative-draft-model-path ${drafter}`,
          // DFlash and n-gram draft a block at once; the others draft a chain of k steps.
          spec.method === "dflash" || spec.method === "ngram"
            ? `--speculative-num-draft-tokens ${k}`
            : `--speculative-num-steps ${k} --speculative-eagle-topk 1 --speculative-num-draft-tokens ${k + 1}`,
        );
      cmd = join(
        `${env}python -m sglang.launch_server --model-path ${name}`,
        `--tp-size ${l.tp}`,
        l.pp > 1 && `--pp-size ${l.pp}`,
        l.memoryFraction && `--mem-fraction-static ${l.memoryFraction}`,
        `--context-length ${l.contextLength}`,
        `--max-running-requests ${l.users}`,
        l.quant === "fp8" && "--quantization fp8",
        l.kvQuant === "fp8" && "--kv-cache-dtype fp8_e5m2",
        offGb > 0 && `--cpu-offload-gb ${offGb}`,
        kvOffGb > 0 && `--enable-hierarchical-cache --hicache-size ${kvOffGb}`,
        specFlags,
        l.flash === false && "--attention-backend triton",
        port,
      );
      break;
    }
    case "tensorrt_llm": {
      // Speculative decoding and KV cache options only go through a YAML file.
      const yaml: string[] = [];
      if (spec) {
        const type = { draft: "DraftTarget", eagle3: "Eagle", mtp: "MTP", ngram: "NGram", dflash: "", suffix: "" }[spec.method];
        yaml.push("speculative_config:", `  decoding_type: ${type}`);
        if (spec.method === "mtp") yaml.push(`  num_nextn_predict_layers: ${k}`);
        else yaml.push(`  max_draft_len: ${k}`);
        if (drafter) yaml.push(`  speculative_model: ${drafter}`);
        if (spec.method === "ngram") yaml.push("  max_matching_ngram_size: 4");
      }
      if (l.kvQuant === "fp8" || kvOffGb > 0) {
        yaml.push("kv_cache_config:");
        if (l.kvQuant === "fp8") yaml.push("  dtype: fp8");
        if (kvOffGb > 0) yaml.push(`  host_cache_size: ${kvOffGb * GiB}`);
      }
      if (offGb > 0) notes.push("TensorRT-LLM can't keep weights in CPU memory; the command leaves that out.");
      const serve = join(
        `${env}trtllm-serve ${name}`,
        `--tp_size ${l.tp}`,
        l.pp > 1 && `--pp_size ${l.pp}`,
        `--max_batch_size ${l.users}`,
        `--max_seq_len ${l.contextLength}`,
        yaml.length > 0 && "--config config.yaml",
        port,
      );
      cmd = yaml.length ? `cat > config.yaml <<'EOF'\n${yaml.join("\n")}\nEOF\n${serve}` : serve;
      break;
    }
    case "llama_cpp": {
      const kv = GGUF_KV[l.kvQuant ?? ""];
      const others = Array.from({ length: (l.machines ?? 1) - 1 }, (_, i) => `<machine-${i + 2}>:50052`);
      const specFlags =
        spec &&
        (spec.method === "ngram"
          ? "--spec-default"
          : join(
              drafter && `-md ${drafter}.gguf`,
              `--spec-type ${{ draft: "draft-simple", eagle3: "draft-eagle3", mtp: "draft-mtp", dflash: "draft-dflash" }[spec.method as string]}`,
              `--spec-draft-n-max ${k}`,
              spec.method === "draft" && "-ngld all",
            ));
      if (kv && l.flash === false) notes.push("A quantized KV cache needs flash attention, so the command keeps it on.");
      if (others.length)
        notes.push(
          `Start ggml-rpc-server -p 50052 on the other ${others.length === 1 ? "machine" : `${others.length} machines`} first.`,
        );
      cmd = join(
        `${env}llama-server -m ${gguf}`,
        // One shared context, split evenly between the parallel slots.
        `-c ${l.contextLength * l.users} -np ${l.users}`,
        // MoE: keep attention on the GPU and move whole layers of experts to the CPU, which is much faster.
        off && g.moe ? `-ngl all -ncmoe ${Math.ceil(g.layers * off.fraction)}` : `-ngl ${off ? gpuLayers : "all"}`,
        kvOffGb > 0 && "-nkvo",
        kv ? `-fa on -ctk ${kv} -ctv ${kv}` : l.flash === false && "-fa off",
        specFlags,
        others.length > 0 && `--rpc ${others.join(",")}`,
        port,
      );
      break;
    }
    case "ollama": {
      const kv = GGUF_KV[l.kvQuant ?? ""];
      if (off)
        notes.push(`Ollama picks GPU layers itself; to match this setup, add PARAMETER num_gpu ${gpuLayers} to a Modelfile.`);
      cmd = join(
        env + (portNum ? `OLLAMA_HOST=127.0.0.1:${portNum}` : ""),
        `OLLAMA_NUM_PARALLEL=${l.users}`,
        `OLLAMA_CONTEXT_LENGTH=${l.contextLength}`,
        kv ? "OLLAMA_FLASH_ATTENTION=1" : l.flash === false && "OLLAMA_FLASH_ATTENTION=0",
        kv && `OLLAMA_KV_CACHE_TYPE=${kv}`,
        "ollama serve",
      );
      break;
    }
    case "hf_transformers": {
      const bnb = ({ int8: "bnb-8bit", int4: "bnb-4bit" } as Record<string, string>)[l.quant];
      if (l.flash !== false) notes.push("FlashAttention 2 needs the flash-attn package (pip install flash-attn).");
      if (off) notes.push("--device auto puts what doesn't fit on the GPUs into CPU memory.");
      cmd = join(
        `${env}transformers serve ${name}`,
        (l.tp * l.pp > 1 || off) && "--device auto",
        `--dtype ${l.quant === "fp32" ? "float32" : "bfloat16"}`,
        `--attn-implementation ${l.flash === false ? "eager" : "flash_attention_2"}`,
        bnb && `--quantization ${bnb}`,
        l.users > 1 && "--continuous-batching",
        port,
      );
      break;
    }
    case "tei":
      cmd = join(
        `${env}text-embeddings-router --model-id ${name}`,
        `--max-batch-tokens ${l.contextLength * l.users}`,
        `--max-client-batch-size ${l.users}`,
        port,
      );
      break;
    default:
      cmd = null;
      notes.push(
        "Generic has no launch command; pick vLLM, SGLang, TensorRT-LLM, llama.cpp, Ollama or Transformers as the runtime.",
      );
  }

  const q = l.quant.toUpperCase();
  if (isGguf(l.quant))
    notes.push(
      ["llama_cpp", "ollama", "vllm"].includes(l.runtime)
        ? `Needs a ${q} GGUF file of ${name}; most popular models have one on Hugging Face${l.runtime === "ollama" ? `. Then run ollama run hf.co/<user>/<repo>:${q}` : ""}.`
        : `${runtime} doesn't serve GGUF files; use llama.cpp or Ollama for them.`,
    );
  else if (PREQUANTIZED[l.quant] || (l.runtime === "tensorrt_llm" && l.quant !== "fp16"))
    notes.push(
      `${q} weights come pre-quantized: point the command at a checkpoint of ${name} quantized to ${q} (search Hugging Face).`,
    );
  return { cmd, notes };
}

/** Fine-tuning command for the chosen framework, plus what the user has to fill in. */
export function trainCommand(input: CalculatorInput): { cmd: string | null; notes: string[] } {
  const name = repoOf(input.model_variant);
  const method = input.finetuning_method ?? "lora";
  const gpus = input.num_gpus ?? 1;
  const opt = input.optimization_config ?? {};
  const rank = input.lora_rank ?? 16;
  const accum = input.gradient_accumulation_steps ?? 1;
  const epochs = input.num_epochs ?? 1;
  const dataset = "Replace <your-dataset> with a Hugging Face dataset or a local file.";
  switch (input.framework_preset) {
    case "unsloth":
      return {
        cmd: join(
          `unsloth train --model ${name} --dataset <your-dataset>`,
          `--training-type ${method === "full" ? "full" : "lora"}`,
          method === "qlora" ? "--load-in-4bit" : "--no-load-in-4bit",
          method !== "full" && `--lora-r ${rank} --lora-alpha ${rank}`,
          `--max-seq-length ${input.seq_length}`,
          `--batch-size ${input.batch_size}`,
          `--gradient-accumulation-steps ${accum}`,
          `--num-epochs ${epochs}`,
          opt.gradient_checkpointing !== false && "--gradient-checkpointing unsloth",
          "--output-dir ./outputs",
        ),
        notes: [
          dataset,
          ...(gpus > 1 ? ["This trains on one GPU; pick Hugging Face Trainer as the framework for a multi-GPU command."] : []),
        ],
      };
    case "hf_trainer": {
      const zero = opt.zero_stage ?? 0;
      return {
        cmd: join(
          "trl sft",
          `--model_name_or_path ${name}`,
          "--dataset_name <your-dataset>",
          method !== "full" && `--use_peft --lora_r ${rank} --lora_alpha ${rank * 2}`,
          method === "qlora" && "--load_in_4bit",
          `--attn_implementation ${opt.flash_attention === false ? "eager" : "flash_attention_2"}`,
          `--per_device_train_batch_size ${input.batch_size}`,
          `--gradient_accumulation_steps ${accum}`,
          // TRL checkpoints gradients by default.
          opt.gradient_checkpointing === false && "--gradient_checkpointing False",
          `--max_length ${input.seq_length}`,
          `--num_train_epochs ${epochs}`,
          opt.use_paged_optimizer ? "--optim paged_adamw_8bit" : opt.use_8bit_optimizer && "--optim adamw_8bit",
          gpus > 1 && `--accelerate_config ${zero > 0 ? `zero${zero}` : "multi_gpu"} --num_processes ${gpus}`,
          "--output_dir ./outputs",
        ),
        notes: [
          dataset,
          ...((input.num_nodes ?? 1) > 1 ? ["Across nodes, launch it on each one with accelerate's multi-node settings."] : []),
        ],
      };
    }
    case "torchtune": {
      // Configs exist only for the models torchtune shipped with, named like llama3_1/8B_lora_single_device.
      const llama = /llama-?3(?:\.(\d))?/i.exec(name);
      const family = llama ? `llama3${llama[1] ? `_${llama[1]}` : ""}` : /qwen2\.5/i.test(name) ? "qwen2_5" : null;
      const size = /(\d+)B\b/i.exec(input.model_variant.name ?? name)?.[1];
      const single = gpus === 1;
      return {
        cmd: join(
          "tune run",
          !single && `--nproc_per_node ${gpus}`,
          `${method === "full" ? "full" : "lora"}_finetune_${single ? "single_device" : "distributed"}`,
          `--config ${family && size ? `${family}/${size}B_${method}${single ? "_single_device" : ""}` : "<config>"}`,
          `batch_size=${input.batch_size}`,
          `gradient_accumulation_steps=${accum}`,
          `epochs=${epochs}`,
          method !== "full" && `model.lora_rank=${rank} model.lora_alpha=${rank * 2}`,
          `enable_activation_checkpointing=${opt.gradient_checkpointing !== false ? "True" : "False"}`,
          `tokenizer.max_seq_len=${input.seq_length}`,
        ),
        notes: [
          "torchtune is no longer maintained (last release April 2025) and has configs only for the models it shipped with; tune ls lists them.",
          "It trains on the config's dataset; change dataset in the config to use yours.",
        ],
      };
    }
    default:
      return {
        cmd: null,
        notes: ["Generic PyTorch has no standard command; pick Unsloth, Hugging Face Trainer or torchtune as the framework."],
      };
  }
}

export function Command({ cmd }: { cmd: string }) {
  return (
    <div className="flex items-start gap-2 rounded-md bg-background/60 py-1.5 pl-3 pr-1.5">
      <code className="min-w-0 flex-1 overflow-x-auto py-0.5 font-mono text-xs leading-relaxed text-foreground">
        {cmd.split("\n").map((line, l) =>
          // Indented lines (YAML) keep their spaces; command lines wrap between flags, never inside one.
          /^\s/.test(line) ? (
            <span key={l} className="block whitespace-pre">
              {line}
            </span>
          ) : (
            <span key={l} className="block">
              {line.split(/ (?=-)/).map((part, i) => (
                <React.Fragment key={i}>
                  {i > 0 && " "}
                  <span className="inline-block max-w-full [overflow-wrap:anywhere]">{part}</span>
                </React.Fragment>
              ))}
            </span>
          ),
        )}
      </code>
      <CopyButton text={cmd} />
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-7 w-7 shrink-0"
      aria-label={copied ? "Copied" : "Copy command"}
      title={copied ? "Copied" : "Copy command"}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {}
      }}
    >
      {copied ? <Check /> : <Copy />}
    </Button>
  );
}

function Notes({ notes }: { notes: (string | false | null | undefined)[] }) {
  return (
    <>
      {[...new Set(notes.filter(Boolean))].map((n) => (
        <p key={n as string} className="text-xs text-muted-foreground">
          {n}
        </p>
      ))}
    </>
  );
}

/** Simple view: commands for one machine of the setup, and what to do on the others. */
export function RunSection({ input, gpu, output }: { input: CalculatorInput; gpu: GpuSpec; output: CalculatorOutput }) {
  if (output.memory_status === "insufficient" || output.memory_status === "error") return null;

  if (input.calc_mode === "finetuning") {
    const { cmd, notes } = trainCommand(input);
    return (
      <Section title="Run it">
        {cmd && <Command cmd={cmd} />}
        <Notes notes={notes} />
      </Section>
    );
  }

  const model = input.model_variant;
  const encoder = isEncoderTask(model);
  const runtime = encoder ? "tei" : (input.framework_preset ?? "vllm");
  const quant = input.quantization ?? "fp16";
  const apple = gpu.type === "apple_silicon";
  // A Mac is a machine of its own; other GPUs sit gpus_per_node to a node.
  const perMachine = apple ? 1 : (input.gpus_per_node ?? output.gpu_count);
  const machines = Math.ceil(output.gpu_count / perMachine);
  // Embedding servers run one copy per GPU.
  const tp = encoder ? 1 : (input.tp_degree ?? 1);
  const pp = encoder ? 1 : (input.pp_degree ?? 1);
  const perCopy = tp * pp;
  const span = Math.ceil(perCopy / perMachine);
  const here = Math.max(1, Math.floor(perMachine / perCopy));
  const users = encoder ? input.batch_size : Math.ceil((input.concurrent_users ?? 1) / output.replicas);
  const fraction = input.enable_offloading ? (input.percentage_offload ?? 0.3) : 0;
  // offloaded_memory is per copy: weights first, the rest is KV cache.
  const kvGb = fraction ? Math.max(0, output.offloaded_memory - (weightBytes(geometry(model), quant) * fraction) / GiB) : 0;

  const launches = Array.from({ length: here }, (_, c) =>
    launch(model, {
      runtime,
      quant,
      kvQuant: input.kv_cache_quantization,
      tp,
      pp,
      machines: span,
      users,
      contextLength: input.seq_length,
      devices: here > 1 && !apple ? Array.from({ length: perCopy }, (_, i) => c * perCopy + i) : null,
      copy: here > 1 ? c : null,
      spec:
        input.enable_speculative_decoding && !encoder
          ? { method: input.speculative_method ?? "draft", tokens: input.speculative_draft_tokens ?? 4 }
          : undefined,
      flash: input.optimization_config?.flash_attention,
      offload: fraction ? { fraction, kvGb } : undefined,
    }),
  );

  const where = apple ? "Macs" : "nodes";
  return (
    <Section title="Run it">
      {launches.map(({ cmd }, i) => cmd && <Command key={i} cmd={cmd} />)}
      <Notes
        notes={[
          ...launches[0].notes,
          span > 1
            ? runtime === "vllm"
              ? `Each copy spans ${span} ${where}: start Ray on them first (ray start --head on one, ray start --address=<head-ip>:6379 on the others), then run this on the head.`
              : runtime !== "llama_cpp" && `Each copy spans ${span} ${where}; follow your runtime's multi-node guide.`
            : machines > 1 && `Run the same on each of the ${machines} ${where}.`,
          output.replicas > 1 && `That makes ${output.replicas} copies: put them behind a load balancer (e.g. LiteLLM or nginx).`,
        ]}
      />
    </Section>
  );
}
