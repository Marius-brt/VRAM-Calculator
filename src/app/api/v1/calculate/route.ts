import { NextRequest, NextResponse } from "next/server";
import { calculate } from "@/lib/calculator/engine";
import type { CalculatorInput, GpuSpec } from "@/lib/calculator/types";
import { GPUS } from "@/lib/data";

export async function POST(req: NextRequest) {
  const body = (await req.json().catch(() => null)) as CalculatorInput | null;
  if (!body?.model_variant || !body.gpu_key) {
    return NextResponse.json({ error: "Expected { gpu_key, model_variant, seq_length, ... }" }, { status: 400 });
  }
  // Unknown keys work as custom GPUs when custom_vram / custom_bandwidth / custom_tflops are given.
  const gpu: GpuSpec = GPUS.find((g) => g.key === body.gpu_key) ?? {
    key: body.gpu_key,
    label: "Custom GPU",
    vendor: "custom",
    type: "discrete",
    memory: body.custom_vram ?? 24,
    bandwidth_gbs: body.custom_bandwidth ?? 1000,
    fp16_tflops: body.custom_tflops ?? 150,
  };
  return NextResponse.json(calculate(body, gpu));
}
