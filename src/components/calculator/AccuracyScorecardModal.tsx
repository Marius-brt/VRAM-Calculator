"use client";

import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface AccuracyScorecardModalProps {
  open: boolean;
  onClose: () => void;
  onSelectQuant: (quant: string) => void;
  current: string;
}

export function AccuracyScorecardModal({ open, onClose, onSelectQuant, current }: AccuracyScorecardModalProps) {
  const scorecardData = [
    {
      format: "FP16 / BF16",
      key: "fp16",
      bpw: 16.0,
      saving: "0%",
      pplDelta: "0.00 (Baseline)",
      quality: "Lossless",
      qualityColor: "text-success",
      recommendation: "Maximum quality, reference training precision",
    },
    {
      format: "GGUF Q8_0",
      key: "q8_0",
      bpw: 8.5,
      saving: "47%",
      pplDelta: "+0.01",
      quality: "Imperceptible",
      qualityColor: "text-success",
      recommendation: "Near-lossless quantization for CPU/Metal",
    },
    {
      format: "FP8 (E4M3)",
      key: "fp8",
      bpw: 8.0,
      saving: "50%",
      pplDelta: "+0.02",
      quality: "Imperceptible",
      qualityColor: "text-success",
      recommendation: "Native Tensor Core FP8 on Ada / Hopper / Blackwell",
    },
    {
      format: "GGUF Q5_K_M",
      key: "q5_k_m",
      bpw: 5.5,
      saving: "65%",
      pplDelta: "+0.04",
      quality: "Excellent",
      qualityColor: "text-success",
      recommendation: "Sweet spot for high reasoning & coding fidelity",
    },
    {
      format: "GGUF Q4_K_M",
      key: "q4_k_m",
      bpw: 4.8,
      saving: "70%",
      pplDelta: "+0.11",
      quality: "Very Good",
      qualityColor: "text-heading",
      recommendation: "Default community standard, best memory-to-quality ratio",
    },
    {
      format: "AWQ (4-bit)",
      key: "awq",
      bpw: 4.6,
      saving: "71%",
      pplDelta: "+0.10",
      quality: "Very Good",
      qualityColor: "text-heading",
      recommendation: "Optimal 4-bit precision for vLLM & TensorRT-LLM serving",
    },
    {
      format: "GPTQ (4-bit)",
      key: "gptq",
      bpw: 4.6,
      saving: "71%",
      pplDelta: "+0.13",
      quality: "Good",
      qualityColor: "text-heading",
      recommendation: "Standard 4-bit GPU inference format",
    },
    {
      format: "GGUF Q3_K_M",
      key: "q3_k_m",
      bpw: 3.9,
      saving: "76%",
      pplDelta: "+0.35",
      quality: "Noticeable Drop",
      qualityColor: "text-warning",
      recommendation: "Used when memory is strictly constrained",
    },
    {
      format: "GGUF Q2_K",
      key: "q2_k",
      bpw: 2.9,
      saving: "82%",
      pplDelta: "+1.20",
      quality: "High Degradation",
      qualityColor: "text-danger",
      recommendation: "Not recommended for reasoning or math tasks",
    },
  ];

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Quantization guide</DialogTitle>
          <DialogDescription>
            How much memory each format saves and what it costs in quality, measured as perplexity increase on Wikitext-2 and C4.
            Lower is better.
          </DialogDescription>
        </DialogHeader>

        <div className="-mx-2 overflow-x-auto">
          <table className="w-full text-left text-[13px]">
            <thead>
              <tr className="border-b text-xs text-muted-foreground">
                <th className="px-2 py-2 font-normal">Format</th>
                <th className="px-2 py-2 text-right font-normal">Bits per weight</th>
                <th className="px-2 py-2 text-right font-normal">Memory saved</th>
                <th className="px-2 py-2 text-right font-normal">Perplexity</th>
                <th className="px-2 py-2 font-normal">Quality</th>
                <th className="px-2 py-2" />
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {scorecardData.map((row) => (
                <tr key={row.key} title={row.recommendation}>
                  <td className="px-2 py-2.5 font-medium text-heading">{row.format}</td>
                  <td className="px-2 py-2.5 text-right tabular-nums">{row.bpw}</td>
                  <td className="px-2 py-2.5 text-right tabular-nums">{row.saving}</td>
                  <td className="px-2 py-2.5 text-right tabular-nums">{row.pplDelta.replace(" (Baseline)", "")}</td>
                  <td className={`px-2 py-2.5 ${row.qualityColor}`}>{row.quality}</td>
                  <td className="px-2 py-2.5 text-right">
                    {row.key === current ? (
                      <span className="text-xs text-muted-foreground">In use</span>
                    ) : (
                      <Button
                        variant="subtle"
                        size="xs"
                        onClick={() => {
                          onSelectQuant(row.key);
                          onClose();
                        }}
                      >
                        Use
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DialogContent>
    </Dialog>
  );
}
