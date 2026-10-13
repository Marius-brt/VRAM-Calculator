"use client";

import React, { useState, useMemo } from "react";
import { CalculatorInput, GpuSpec } from "@/lib/calculator/types";
import { calculate } from "@/lib/calculator/engine";
import { QUANT_LADDER, runtimeFor } from "@/lib/calculator/planner";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Segmented } from "./controls";

type Objective = "cost" | "throughput" | "single_gpu";

const QUANT_LABELS: Record<string, string> = { fp16: "FP16", fp8: "FP8", awq: "AWQ 4-bit", q8_0: "Q8_0", q4_k_m: "Q4_K_M" };

interface SetupOptimizerModalProps {
  open: boolean;
  onClose: () => void;
  baseInput: CalculatorInput;
  gpus: GpuSpec[];
  onApplyConfig: (config: { gpuKey: string; numGpus: number; quantization: string; tpDegree: number }) => void;
}

export function SetupOptimizerModal({ open, onClose, baseInput, gpus, onApplyConfig }: SetupOptimizerModalProps) {
  const [objective, setObjective] = useState<Objective>("cost");

  // Every GPU in the catalog at the runtime's usual precisions, on 1 to 8 GPUs.
  const recommendations = useMemo(() => {
    if (!open || !baseInput.model_variant) return [];
    const results: Array<{
      gpu: GpuSpec;
      gpuCount: number;
      quant: string;
      vramPerGpu: number;
      tps: number;
      hourlyCost: number | null;
    }> = [];

    for (const gpu of gpus) {
      if (gpu.released === false || gpu.memory <= 0) continue;
      const runtime = runtimeFor(gpu, baseInput.model_variant);
      for (const quant of QUANT_LADDER[runtime] ?? ["fp16"]) {
        // Smallest GPU count that fits for this GPU/precision pair (a Mac is a single GPU).
        for (const count of gpu.type === "apple_silicon" ? [1] : [1, 2, 4, 8]) {
          const res = calculate(
            {
              ...baseInput,
              gpu_key: gpu.key,
              quantization: quant,
              num_gpus: count,
              tp_degree: count,
              pp_degree: 1,
              framework_preset: runtime as CalculatorInput["framework_preset"],
            },
            gpu,
          );
          if (res.memory_status === "insufficient") continue;
          results.push({
            gpu,
            gpuCount: count,
            quant,
            vramPerGpu: res.per_gpu_usage,
            tps: res.per_user_tps ?? 0,
            hourlyCost: gpu.hourly_price_usd != null ? gpu.hourly_price_usd * count : null,
          });
          break;
        }
      }
    }

    const byCost = (a: (typeof results)[number], b: (typeof results)[number]) => a.hourlyCost! - b.hourlyCost! || b.tps - a.tps;
    if (objective === "cost")
      return results
        .filter((r) => r.hourlyCost != null)
        .sort(byCost)
        .slice(0, 8);
    if (objective === "single_gpu")
      return results
        .filter((r) => r.gpuCount === 1)
        .sort((a, b) => b.tps - a.tps)
        .slice(0, 8);
    return results.sort((a, b) => b.tps - a.tps).slice(0, 8);
  }, [open, baseInput, gpus, objective]);

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Find hardware for {baseInput.model_variant?.name || "this model"}</DialogTitle>
          <DialogDescription>
            Every GPU in the catalog at full, 8-bit and 4-bit precision, with {baseInput.seq_length.toLocaleString()} tokens of
            context and {baseInput.concurrent_users ?? 1} {(baseInput.concurrent_users ?? 1) === 1 ? "user" : "users"}. Speeds are
            per user.
          </DialogDescription>
        </DialogHeader>

        <Segmented
          label="Optimize for"
          value={objective}
          onChange={setObjective}
          options={[
            { value: "cost", label: "Lowest cost" },
            { value: "throughput", label: "Fastest" },
            { value: "single_gpu", label: "Single GPU" },
          ]}
        />

        {recommendations.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            Nothing fits on up to 8 GPUs. Shorten the context or lower the batch size.
          </p>
        ) : (
          <ul className="-mx-2 divide-y divide-border/60">
            {recommendations.map((rec) => (
              <li key={`${rec.gpu.key}-${rec.quant}`} className="flex items-center justify-between gap-4 px-2 py-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-heading">
                    {rec.gpuCount > 1 && `${rec.gpuCount}× `}
                    {rec.gpu.label}
                    <span className="ml-2 font-normal text-muted-foreground">{QUANT_LABELS[rec.quant]}</span>
                  </div>
                  <div className="mt-0.5 text-xs tabular-nums text-muted-foreground">
                    {rec.vramPerGpu} of {rec.gpu.memory} GB per GPU, about {rec.tps} tok/s
                    {rec.hourlyCost != null ? `, $${rec.hourlyCost.toFixed(2)} per hour` : ", not commonly rented"}
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    onApplyConfig({
                      gpuKey: rec.gpu.key,
                      numGpus: rec.gpuCount,
                      quantization: rec.quant,
                      tpDegree: rec.gpuCount,
                    });
                    onClose();
                  }}
                  className="shrink-0"
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
