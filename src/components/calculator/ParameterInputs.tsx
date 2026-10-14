"use client";

import React from "react";
import { ChevronDown } from "lucide-react";
import { Input } from "@/components/ui/input";
import type { ModelTask, SpecMethod } from "@/lib/calculator/types";
import { Button } from "@/components/ui/button";
import { Field, Section, Segmented, Select, StepSlider, ToggleRow, formatTokens } from "./controls";

/** Powers of two from `from` up to `to`, each followed by the halfway size (2K, 3K, 4K, 6K, 8K...). */
const doubling = (from: number, to: number) => {
  const steps: number[] = [];
  for (let s = from; s <= to; s *= 2) steps.push(s, s * 1.5);
  return steps.filter((s) => s <= to);
};
/** 512 to 128K, then on up to the model's own context window (1M, 10M, ...), which is always a stop. */
export const contextSteps = (modelContext: number) => {
  const steps = [512, ...doubling(1024, Math.max(131072, modelContext))].filter((s) => s <= 131072 || s < modelContext);
  return modelContext > 131072 ? [...steps, modelContext] : steps;
};
/** Embedding/reranking inputs: 64 tokens up to the model's maximum (they can't go beyond it). */
export const encoderSteps = (max: number) => [...doubling(64, max).filter((s) => s < max), max];

/** 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32... up to `max`: each power of two, then 1.25× and 1.5× of it. */
export const countSteps = (max: number) => {
  const steps = [1, 2, 3, 4, 5, 6];
  for (let s = 8; s <= max; s *= 2) steps.push(s, s * 1.25, s * 1.5);
  return steps.filter((n) => n <= max);
};
/** Tick labels for dense steps: only the powers of two. */
const pow2 = (steps: number[]) => steps.filter((n) => Number.isInteger(Math.log2(n)));

const BATCH_STEPS = countSteps(128);
const ENCODER_BATCH_STEPS = countSteps(1024);
const USER_STEPS = countSteps(2048);
const ACCUM_STEPS = countSteps(64);
const PERCENT_STEPS = Array.from({ length: 19 }, (_, i) => (i + 1) * 5);
const PERCENT_MARKS = [10, 30, 50, 70, 90];
const ACCEPT_STEPS = PERCENT_STEPS.filter((n) => n >= 30);
const DRAFT_TOKEN_STEPS = countSteps(16);

/** Typical settings for each method; acceptance depends a lot on the workload (code and RAG repeat more). */
export const SPEC_METHODS: Record<SpecMethod, { label: string; tokens: number; acceptance: number; note: string }> = {
  draft: {
    label: "Draft model",
    tokens: 4,
    acceptance: 0.7,
    note: "A small model of the same family (same tokenizer) proposes tokens.",
  },
  eagle3: {
    label: "EAGLE-3 head",
    tokens: 4,
    acceptance: 0.75,
    note: "A trained one-layer head reads the model's hidden states.",
  },
  mtp: {
    label: "MTP (built-in)",
    tokens: 1,
    acceptance: 0.85,
    note: "Uses the model's own multi-token-prediction layers (DeepSeek-V3, GLM-4.5+, Qwen3.5, MiMo).",
  },
  dflash: { label: "DFlash", tokens: 16, acceptance: 0.8, note: "A block-diffusion drafter proposes a whole block in one pass." },
  ngram: {
    label: "N-gram lookup",
    tokens: 4,
    acceptance: 0.4,
    note: "Copies continuations from the prompt. No extra memory; best for code edits, RAG and summaries.",
  },
  suffix: {
    label: "Suffix decoding",
    tokens: 8,
    acceptance: 0.5,
    note: "Matches earlier prompts and answers. No extra memory; best for agent loops.",
  },
};
const pct = (n: number) => `${n}%`;

export const WEIGHT_QUANTS: [string, [string, string][]][] = [
  [
    "16 and 32-bit",
    [
      ["fp16", "FP16 / BF16"],
      ["fp32", "FP32"],
    ],
  ],
  [
    "8-bit",
    [
      ["fp8", "FP8"],
      ["int8", "INT8 (bitsandbytes)"],
      ["q8_0", "GGUF Q8_0 (8.5 bpw)"],
    ],
  ],
  [
    "5 and 6-bit",
    [
      ["q6_k", "GGUF Q6_K (6.6 bpw)"],
      ["q5_k_m", "GGUF Q5_K_M (5.7 bpw)"],
      ["q5_k_s", "GGUF Q5_K_S (5.5 bpw)"],
    ],
  ],
  [
    "4-bit",
    [
      ["nvfp4", "NVFP4 (Blackwell)"],
      ["mxfp4", "MXFP4"],
      ["awq", "AWQ"],
      ["gptq", "GPTQ"],
      ["q4_k_m", "GGUF Q4_K_M (4.8 bpw)"],
      ["q4_k_s", "GGUF Q4_K_S (4.6 bpw)"],
      ["exl2", "EXL2 (4.5 bpw)"],
      ["int4", "INT4 (bitsandbytes)"],
    ],
  ],
  [
    "2 and 3-bit",
    [
      ["q3_k_m", "GGUF Q3_K_M (3.9 bpw)"],
      ["q2_k", "GGUF Q2_K (2.9 bpw)"],
    ],
  ],
];
export const KV_QUANTS: [string, string][] = [
  ["auto", "Runtime default"],
  ["fp16", "FP16"],
  ["fp8", "FP8 (half the memory)"],
  ["int8", "INT8"],
  ["int4", "INT4 (quarter the memory)"],
];
export const SERVING_RUNTIMES: [string, string][] = [
  ["vllm", "vLLM"],
  ["sglang", "SGLang"],
  ["tensorrt_llm", "TensorRT-LLM"],
  ["llama_cpp", "llama.cpp"],
  ["ollama", "Ollama"],
  ["hf_transformers", "Hugging Face Transformers"],
  ["generic", "Generic"],
];
export const options = (list: [string, string][]) =>
  list.map(([v, l]) => (
    <option key={v} value={v}>
      {l}
    </option>
  ));
export const WeightOptions = () =>
  WEIGHT_QUANTS.map(([group, list]) => (
    <optgroup key={group} label={group}>
      {options(list)}
    </optgroup>
  ));

interface ParameterInputsProps {
  /** Opens the quantization guide (memory saved and quality cost of each format). */
  onOpenScorecard: () => void;
  calcMode: "inference" | "finetuning";
  modelContext: number;
  task?: ModelTask;
  quantization: string;
  setQuantization: (q: string) => void;
  kvCacheQuant: string;
  setKvCacheQuant: (kq: string) => void;
  seqLength: number;
  setSeqLength: (seq: number) => void;
  batchSize: number;
  setBatchSize: (bs: number) => void;
  concurrentUsers: number;
  setConcurrentUsers: (users: number) => void;
  frameworkPreset: string;
  setFrameworkPreset: (fp: string) => void;
  enablePrefixCaching: boolean;
  setEnablePrefixCaching: (pc: boolean) => void;
  sharedPrefixRatio: number;
  setSharedPrefixRatio: (ratio: number) => void;
  enableOffloading: boolean;
  setEnableOffloading: (off: boolean) => void;
  offloadPercentage: number;
  setOffloadPercentage: (pct: number) => void;
  offloadKvCache: boolean;
  setOffloadKvCache: (okv: boolean) => void;
  enableSpeculative: boolean;
  setEnableSpeculative: (spec: boolean) => void;
  specMethod: SpecMethod;
  setSpecMethod: (m: SpecMethod) => void;
  specDraftTokens: number;
  setSpecDraftTokens: (k: number) => void;
  specDraftParams: number;
  setSpecDraftParams: (p: number) => void;
  specAcceptanceRate: number;
  setSpecAcceptanceRate: (r: number) => void;

  // Fine-tuning
  ftMethod: "full" | "lora" | "qlora";
  setFtMethod: (m: "full" | "lora" | "qlora") => void;
  ftQuant: string;
  setFtQuant: (q: string) => void;
  loraRank: number;
  setLoraRank: (r: number) => void;
  gradAccumSteps: number;
  setGradAccumSteps: (ga: number) => void;
  optimizerType: "adamw" | "adamw_8bit" | "paged_adamw";
  setOptimizerType: (opt: "adamw" | "adamw_8bit" | "paged_adamw") => void;
  zeroStage: number;
  setZeroStage: (stage: number) => void;
  gradCheckpointing: boolean;
  setGradCheckpointing: (gc: boolean) => void;
  flashAttention: boolean;
  setFlashAttention: (fa: boolean) => void;
  numSamples: number;
  setNumSamples: (s: number) => void;
  epochs: number;
  setEpochs: (ep: number) => void;
}

export function ParameterInputs(p: ParameterInputsProps) {
  const inference = p.calcMode === "inference";
  const encoder = p.task === "embedding" || p.task === "reranking";
  const advancedOn = inference
    ? [p.enablePrefixCaching, p.enableSpeculative, p.enableOffloading, !p.flashAttention].filter(Boolean).length
    : [p.enableOffloading, p.gradCheckpointing, p.flashAttention].filter(Boolean).length;

  return (
    <>
      {inference ? (
        <Section
          title="Precision"
          action={
            <Button variant="ghost" size="xs" onClick={p.onOpenScorecard}>
              Compare formats
            </Button>
          }
        >
          <div className={encoder ? "grid gap-3" : "grid gap-3 sm:grid-cols-[1.4fr_1fr]"}>
            <Field label="Model weights">
              <Select value={p.quantization} onChange={(e) => p.setQuantization(e.target.value)}>
                <WeightOptions />
              </Select>
            </Field>
            {!encoder && (
              <Field label="KV cache">
                <Select value={p.kvCacheQuant} onChange={(e) => p.setKvCacheQuant(e.target.value)}>
                  {options(KV_QUANTS)}
                </Select>
              </Field>
            )}
          </div>
        </Section>
      ) : (
        <Section title="Fine-tuning method">
          <Segmented
            label="Fine-tuning method"
            value={p.ftMethod}
            onChange={p.setFtMethod}
            options={[
              { value: "qlora", label: "QLoRA" },
              { value: "lora", label: "LoRA" },
              { value: "full", label: "Full" },
            ]}
          />
          <div className="grid gap-3 sm:grid-cols-2">
            {p.ftMethod === "full" ? (
              <>
                <Field label="Training precision">
                  <Select value={p.ftQuant} onChange={(e) => p.setFtQuant(e.target.value)}>
                    <option value="fp16">FP16 / BF16</option>
                    <option value="fp32">FP32</option>
                    <option value="fp8">FP8 (Transformer Engine)</option>
                  </Select>
                </Field>
                <Field label="DeepSpeed ZeRO">
                  <Select value={p.zeroStage} onChange={(e) => p.setZeroStage(Number(e.target.value))}>
                    <option value={0}>Off (plain DDP)</option>
                    <option value={1}>Stage 1, shard optimizer</option>
                    <option value={2}>Stage 2, plus gradients</option>
                    <option value={3}>Stage 3, plus weights</option>
                  </Select>
                </Field>
              </>
            ) : (
              <Field label="LoRA rank">
                <Select value={p.loraRank} onChange={(e) => p.setLoraRank(Number(e.target.value))}>
                  {[8, 16, 32, 64, 128].map((r) => (
                    <option key={r} value={r}>
                      r = {r}
                      {r === 16 ? " (common default)" : ""}
                    </option>
                  ))}
                </Select>
              </Field>
            )}
            <Field label="Optimizer">
              <Select
                value={p.optimizerType}
                onChange={(e) => p.setOptimizerType(e.target.value as ParameterInputsProps["optimizerType"])}
              >
                <option value="adamw">AdamW (8 bytes/param)</option>
                <option value="adamw_8bit">8-bit AdamW (2 bytes/param)</option>
                <option value="paged_adamw">Paged 8-bit AdamW</option>
              </Select>
            </Field>
          </div>
        </Section>
      )}

      <Section title="Workload">
        {encoder ? (
          <>
            <StepSlider
              label={p.task === "reranking" ? "Query + document length" : "Input length"}
              unit="tokens"
              steps={encoderSteps(p.modelContext)}
              marks={pow2(encoderSteps(p.modelContext))}
              value={p.seqLength}
              onChange={p.setSeqLength}
              format={formatTokens}
            />
            <StepSlider
              label="Batch size"
              unit={p.task === "reranking" ? "pairs" : "inputs"}
              steps={ENCODER_BATCH_STEPS}
              marks={pow2(ENCODER_BATCH_STEPS)}
              value={p.batchSize}
              onChange={p.setBatchSize}
            />
          </>
        ) : (
          <>
            <StepSlider
              label="Context length"
              unit="tokens"
              steps={contextSteps(p.modelContext)}
              marks={pow2(contextSteps(p.modelContext))}
              value={p.seqLength}
              onChange={p.setSeqLength}
              format={formatTokens}
              note={
                p.seqLength > p.modelContext && (
                  <p className="text-xs text-warning">
                    Longer than this model&apos;s {formatTokens(p.modelContext)} context window.
                  </p>
                )
              }
            />
            <StepSlider
              label="Batch size"
              steps={BATCH_STEPS}
              marks={pow2(BATCH_STEPS)}
              value={p.batchSize}
              onChange={p.setBatchSize}
            />
            {inference ? (
              <StepSlider
                label="Concurrent users"
                steps={USER_STEPS}
                marks={pow2(USER_STEPS)}
                value={p.concurrentUsers}
                onChange={p.setConcurrentUsers}
              />
            ) : (
              <>
                <StepSlider
                  label="Gradient accumulation"
                  unit="steps"
                  steps={ACCUM_STEPS}
                  marks={pow2(ACCUM_STEPS)}
                  value={p.gradAccumSteps}
                  onChange={p.setGradAccumSteps}
                />
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Training samples">
                    <Input
                      type="number"
                      min={1}
                      value={p.numSamples}
                      onChange={(e) => p.setNumSamples(Math.max(1, Number(e.target.value)))}
                    />
                  </Field>
                  <Field label="Epochs">
                    <Input
                      type="number"
                      min={1}
                      value={p.epochs}
                      onChange={(e) => p.setEpochs(Math.max(1, Number(e.target.value)))}
                    />
                  </Field>
                </div>
              </>
            )}
          </>
        )}
        {!encoder && (
          <Field label="Runtime">
            <Select value={p.frameworkPreset} onChange={(e) => p.setFrameworkPreset(e.target.value)}>
              {inference ? (
                options(SERVING_RUNTIMES)
              ) : (
                <>
                  <option value="unsloth">Unsloth</option>
                  <option value="hf_trainer">Hugging Face Trainer</option>
                  <option value="torchtune">torchtune</option>
                  <option value="generic">Generic PyTorch</option>
                </>
              )}
            </Select>
          </Field>
        )}
      </Section>

      {!encoder && (
        <details className="group border-t">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-4 sm:px-5 text-sm font-semibold text-heading [&::-webkit-details-marker]:hidden">
            <span>
              Advanced
              {advancedOn > 0 && <span className="ml-2 text-xs font-normal text-link">{advancedOn} on</span>}
            </span>
            <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
          </summary>
          <div className="space-y-5 px-4 pb-5 sm:px-5">
            {!inference && (
              <ToggleRow
                title="Gradient checkpointing"
                description="Recompute activations in the backward pass; cuts activation memory by about 80%."
                checked={p.gradCheckpointing}
                onCheckedChange={p.setGradCheckpointing}
              />
            )}
            <OptimizationToggles {...p} inference={inference} />
          </div>
        </details>
      )}
    </>
  );
}

type OptimizationProps = Pick<
  ParameterInputsProps,
  | "enablePrefixCaching"
  | "setEnablePrefixCaching"
  | "sharedPrefixRatio"
  | "setSharedPrefixRatio"
  | "enableSpeculative"
  | "setEnableSpeculative"
  | "specMethod"
  | "setSpecMethod"
  | "specDraftTokens"
  | "setSpecDraftTokens"
  | "specDraftParams"
  | "setSpecDraftParams"
  | "specAcceptanceRate"
  | "setSpecAcceptanceRate"
  | "enableOffloading"
  | "setEnableOffloading"
  | "offloadPercentage"
  | "setOffloadPercentage"
  | "offloadKvCache"
  | "setOffloadKvCache"
  | "flashAttention"
  | "setFlashAttention"
> & { inference: boolean };

/** Prefix caching, speculative decoding, offloading and flash attention; also used per model in the Advanced view. */
export function OptimizationToggles(p: OptimizationProps) {
  const inference = p.inference;
  return (
    <>
      {inference && (
        <>
          <ToggleRow
            title="Prefix caching"
            description="Reuse the KV cache of a shared system prompt across users."
            checked={p.enablePrefixCaching}
            onCheckedChange={p.setEnablePrefixCaching}
          >
            <StepSlider
              label="Shared prefix"
              steps={PERCENT_STEPS}
              marks={PERCENT_MARKS}
              value={Math.round(p.sharedPrefixRatio * 100)}
              onChange={(v) => p.setSharedPrefixRatio(v / 100)}
              format={pct}
              unit="%"
            />
          </ToggleRow>
          <ToggleRow
            title="Speculative decoding"
            description="Guess several tokens cheaply, check them in one pass; typically 1.5 to 3× faster for few users."
            checked={p.enableSpeculative}
            onCheckedChange={p.setEnableSpeculative}
          >
            <Field label="Method" hint={SPEC_METHODS[p.specMethod].note}>
              <Select
                value={p.specMethod}
                onChange={(e) => {
                  const m = e.target.value as SpecMethod;
                  p.setSpecMethod(m);
                  p.setSpecDraftTokens(SPEC_METHODS[m].tokens);
                  p.setSpecAcceptanceRate(SPEC_METHODS[m].acceptance);
                }}
              >
                {Object.entries(SPEC_METHODS).map(([v, m]) => (
                  <option key={v} value={v}>
                    {m.label}
                  </option>
                ))}
              </Select>
            </Field>
            {p.specMethod === "draft" && (
              <Field label="Draft model size (billions of parameters)">
                <Input
                  type="number"
                  min={0.1}
                  step={0.5}
                  value={p.specDraftParams}
                  onChange={(e) => p.setSpecDraftParams(Number(e.target.value))}
                />
              </Field>
            )}
            <StepSlider
              label="Tokens drafted per step"
              steps={DRAFT_TOKEN_STEPS}
              marks={pow2(DRAFT_TOKEN_STEPS)}
              value={p.specDraftTokens}
              onChange={p.setSpecDraftTokens}
            />
            <StepSlider
              label="Acceptance rate"
              steps={ACCEPT_STEPS}
              marks={PERCENT_MARKS.filter((n) => n >= 30)}
              value={Math.round(p.specAcceptanceRate * 100)}
              onChange={(v) => p.setSpecAcceptanceRate(v / 100)}
              format={pct}
              unit="%"
            />
          </ToggleRow>
        </>
      )}
      <ToggleRow
        title="Offload to CPU memory"
        description="Keep some layers in system RAM to run models larger than VRAM, at a speed cost."
        checked={p.enableOffloading}
        onCheckedChange={p.setEnableOffloading}
      >
        <StepSlider
          label="Layers offloaded"
          steps={PERCENT_STEPS}
          marks={PERCENT_MARKS}
          value={p.offloadPercentage}
          onChange={p.setOffloadPercentage}
          format={pct}
          unit="%"
        />
        <ToggleRow
          title="Offload KV cache too"
          description="Moves most of the KV cache to system RAM."
          checked={p.offloadKvCache}
          onCheckedChange={p.setOffloadKvCache}
        />
      </ToggleRow>
      <ToggleRow
        title="Flash attention"
        description={
          inference
            ? "Fused attention kernels. Without them llama.cpp, Ollama and Transformers build the full attention matrix."
            : "Memory-efficient attention kernels (FlashAttention 2 or 3)."
        }
        checked={p.flashAttention}
        onCheckedChange={p.setFlashAttention}
      />
    </>
  );
}
