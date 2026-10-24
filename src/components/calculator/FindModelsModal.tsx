"use client";

import React, { useMemo, useState } from "react";
import { calculate, isEncoderTask } from "@/lib/calculator/engine";
import { CalculatorInput, GpuSpec, ModelVariant } from "@/lib/calculator/types";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Field, Segmented, Select, formatTokens } from "./controls";
import { formatParams } from "./ModelSelector";

type Sort = "popular" | "newest" | "largest" | "fastest";

const GGUF_RUNTIMES = new Set(["llama_cpp", "ollama", "unsloth"]);
const LABELS: Record<string, string> = { fp16: "BF16", fp8: "FP8", awq: "4-bit", q8_0: "Q8_0", q6_k: "Q6_K", q4_k_m: "Q4_K_M" };

interface Props {
  open: boolean;
  onClose: () => void;
  baseInput: CalculatorInput;
  gpu: GpuSpec;
  hardwareLabel: string;
  models: ModelVariant[];
  onUse: (model: ModelVariant, quantization: string) => void;
}

/** Every chat model in the catalog at the best precision that fits the current hardware and workload. */
export function FindModelsModal({ open, onClose, baseInput, gpu, hardwareLabel, models, onUse }: Props) {
  const [sort, setSort] = useState<Sort>("popular");
  const [minSpeed, setMinSpeed] = useState(10);

  const fits = useMemo(() => {
    if (!open) return [];
    const ladder = GGUF_RUNTIMES.has(baseInput.framework_preset ?? "") ? ["q8_0", "q6_k", "q4_k_m"] : ["fp16", "fp8", "awq"];
    return models
      .filter((m) => !isEncoderTask(m) && m.is_preset === 1)
      .flatMap((m) => {
        for (const quant of ladder) {
          const out = calculate({ ...baseInput, calc_mode: "inference", model_variant: m, quantization: quant }, gpu);
          if (out.memory_status !== "insufficient") return [{ model: m, quant, tps: out.per_user_tps ?? 0, ttft: out.ttft_ms }];
        }
        return [];
      });
  }, [open, models, baseInput, gpu]);

  const shown = useMemo(() => {
    const by: Record<Sort, (a: (typeof fits)[number], b: (typeof fits)[number]) => number> = {
      popular: (a, b) => (b.model.downloads ?? 0) - (a.model.downloads ?? 0),
      newest: (a, b) => (b.model.release_date ?? "").localeCompare(a.model.release_date ?? ""),
      largest: (a, b) => (b.model.num_of_params ?? 0) - (a.model.num_of_params ?? 0),
      fastest: (a, b) => b.tps - a.tps,
    };
    return fits.filter((f) => f.tps >= minSpeed).sort(by[sort]).slice(0, 40);
  }, [fits, sort, minSpeed]);

  const users = baseInput.concurrent_users ?? 1;

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Models that run on {hardwareLabel}</DialogTitle>
          <DialogDescription>
            {fits.length} of {models.filter((m) => !isEncoderTask(m) && m.is_preset === 1).length} catalog models fit with{" "}
            {formatTokens(baseInput.seq_length)} tokens of context and {users} {users === 1 ? "user" : "users"}, each at the highest
            precision that fits. Popularity is Hugging Face downloads over the last 30 days.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap items-end justify-between gap-3">
          <Segmented
            label="Sort by"
            value={sort}
            onChange={setSort}
            options={[
              { value: "popular", label: "Popular" },
              { value: "newest", label: "Newest" },
              { value: "largest", label: "Largest" },
              { value: "fastest", label: "Fastest" },
            ]}
          />
          <Field label="Speed per user" className="w-40">
            <Select value={minSpeed} onChange={(e) => setMinSpeed(Number(e.target.value))}>
              <option value={0}>Any speed</option>
              <option value={10}>10+ tok/s</option>
              <option value={20}>20+ tok/s</option>
              <option value={50}>50+ tok/s</option>
            </Select>
          </Field>
        </div>

        {shown.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            Nothing fits at this speed. Lower the context or users, add GPUs, or allow slower speeds.
          </p>
        ) : (
          <ul className="-mx-2 divide-y divide-border/60">
            {shown.map(({ model, quant, tps }) => (
              <li key={model.slug} className="flex items-center justify-between gap-4 px-2 py-2.5">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-heading">
                    {model.name}
                    <span className="ml-2 font-normal text-muted-foreground">{LABELS[quant] ?? quant}</span>
                  </div>
                  <div className="mt-0.5 text-xs tabular-nums text-muted-foreground">
                    {model.provider}, {formatParams(model.num_of_params)}
                    {model.architecture === "moe" && model.active_params ? ` (${formatParams(model.active_params)} active)` : ""}, about{" "}
                    {Math.round(tps)} tok/s per user
                    {model.release_date ? `, released ${model.release_date.slice(0, 7)}` : ""}
                    {model.downloads ? `, ${(model.downloads / 1000).toFixed(0)}k downloads` : ""}
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  className="shrink-0"
                  onClick={() => {
                    onUse(model, quant);
                    onClose();
                  }}
                >
                  Use this
                </Button>
              </li>
            ))}
          </ul>
        )}
      </DialogContent>
    </Dialog>
  );
}
