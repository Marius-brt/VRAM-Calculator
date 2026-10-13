"use client";

import React from "react";
import { ChevronDown } from "lucide-react";
import { Input } from "@/components/ui/input";
import type { ModelTask, SpecMethod } from "@/lib/calculator/types";
import { Field, Section, Segmented, Select, StepSlider, ToggleRow, formatTokens } from "./controls";

/** 512 to 128K, then doubling up to the model's own context window (1M, 10M, ...). */
export const contextSteps = (modelContext: number) => {
  const steps = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
  for (let s = 262144; s < modelContext; s *= 2) steps.push(s);
  if (modelContext > 131072) steps.push(modelContext);
  return steps;
};
/** Embedding/reranking inputs: 64 tokens up to the model's maximum (they can't go beyond it). */
export const encoderSteps = (max: number) => {
  const steps: number[] = [];
  for (let s = 64; s < max; s *= 2) steps.push(s);
  return [...steps, max];
};

const BATCH_STEPS = [1, 2, 4, 8, 16, 32, 64, 128];
const ENCODER_BATCH_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
const USER_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const ACCUM_STEPS = [1, 2, 4, 8, 16, 32, 64];
const PERCENT_STEPS = [10, 20, 30, 40, 50, 60, 70, 80, 90];
const ACCEPT_STEPS = [30, 40, 50, 60, 70, 80, 85, 90];
const DRAFT_TOKEN_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16];

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

interface ParameterInputsProps {
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
        <Section title="Precision">
          <div className={encoder ? "grid gap-3" : "grid gap-3 sm:grid-cols-[1.4fr_1fr]"}>
            <Field label="Model weights">
              <Select value={p.quantization} onChange={(e) => p.setQuantization(e.target.value)}>
                <optgroup label="16 and 32-bit">
                  <option value="fp16">FP16 / BF16</option>
                  <option value="fp32">FP32</option>
                </optgroup>
                <optgroup label="8-bit">
                  <option value="fp8">FP8</option>
                  <option value="int8">INT8 (bitsandbytes)</option>
                  <option value="q8_0">GGUF Q8_0 (8.5 bpw)</option>
                </optgroup>
                <optgroup label="5 and 6-bit">
                  <option value="q6_k">GGUF Q6_K (6.6 bpw)</option>
                  <option value="q5_k_m">GGUF Q5_K_M (5.7 bpw)</option>
                  <option value="q5_k_s">GGUF Q5_K_S (5.5 bpw)</option>
                </optgroup>
                <optgroup label="4-bit">
                  <option value="nvfp4">NVFP4 (Blackwell)</option>
                  <option value="mxfp4">MXFP4</option>
                  <option value="awq">AWQ</option>
                  <option value="gptq">GPTQ</option>
                  <option value="q4_k_m">GGUF Q4_K_M (4.8 bpw)</option>
                  <option value="q4_k_s">GGUF Q4_K_S (4.6 bpw)</option>
                  <option value="exl2">EXL2 (4.5 bpw)</option>
                  <option value="int4">INT4 (bitsandbytes)</option>
                </optgroup>
                <optgroup label="2 and 3-bit">
                  <option value="q3_k_m">GGUF Q3_K_M (3.9 bpw)</option>
                  <option value="q2_k">GGUF Q2_K (2.9 bpw)</option>
                </optgroup>
              </Select>
            </Field>
            {!encoder && (
              <Field label="KV cache">
                <Select value={p.kvCacheQuant} onChange={(e) => p.setKvCacheQuant(e.target.value)}>
                  <option value="auto">Runtime default</option>
                  <option value="fp16">FP16</option>
                  <option value="fp8">FP8 (half the memory)</option>
                  <option value="int8">INT8</option>
                  <option value="int4">INT4 (quarter the memory)</option>
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
              value={p.seqLength}
              onChange={p.setSeqLength}
              format={formatTokens}
            />
            <StepSlider
              label="Batch size"
              unit={p.task === "reranking" ? "pairs" : "inputs"}
              steps={ENCODER_BATCH_STEPS}
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
            <StepSlider label="Batch size" steps={BATCH_STEPS} value={p.batchSize} onChange={p.setBatchSize} />
            {inference ? (
              <StepSlider label="Concurrent users" steps={USER_STEPS} value={p.concurrentUsers} onChange={p.setConcurrentUsers} />
            ) : (
              <>
                <StepSlider
                  label="Gradient accumulation"
                  unit="steps"
                  steps={ACCUM_STEPS}
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
                <>
                  <option value="vllm">vLLM</option>
                  <option value="sglang">SGLang</option>
                  <option value="tensorrt_llm">TensorRT-LLM</option>
                  <option value="llama_cpp">llama.cpp</option>
                  <option value="ollama">Ollama</option>
                  <option value="hf_transformers">Hugging Face Transformers</option>
                  <option value="generic">Generic</option>
                </>
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
                    value={p.specDraftTokens}
                    onChange={p.setSpecDraftTokens}
                  />
                  <StepSlider
                    label="Acceptance rate"
                    steps={ACCEPT_STEPS}
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
            {!inference && (
              <ToggleRow
                title="Gradient checkpointing"
                description="Recompute activations in the backward pass; cuts activation memory by about 80%."
                checked={p.gradCheckpointing}
                onCheckedChange={p.setGradCheckpointing}
              />
            )}
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
          </div>
        </details>
      )}
    </>
  );
}
