"use client";

import React, { useMemo, useState } from "react";
import { CalculatorInput, GpuSpec } from "@/lib/calculator/types";
import { calculate } from "@/lib/calculator/engine";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from "recharts";
import { Section, Segmented, formatTokens } from "./controls";
import { encoderSteps } from "./ParameterInputs";

type Axis = "seq" | "batch" | "users";
type Metric = "memory" | "speed";

/** Doubling from 1K to the model's window (at least 128K), thinned to 8 points. */
const contextPoints = (ctx: number) => {
  let pts: number[] = [];
  for (let s = 1024; s <= Math.max(131072, ctx); s *= 2) pts.push(s);
  if (ctx > pts[pts.length - 1]) pts.push(ctx);
  while (pts.length > 8) pts = pts.filter((_, i) => (pts.length - 1 - i) % 2 === 0);
  return pts;
};

const PRIMARY = "#845ef7";
const SECONDARY = "#22b8cf";

interface Point {
  x: string;
  vram: number;
  kvCache: number;
  perUser: number | null;
  throughput: number | null;
  ttft: number | null;
}

interface ScalingChartsProps {
  baseInput: CalculatorInput;
  gpu: GpuSpec;
  capacity: number;
}

export function ScalingCharts({ baseInput, gpu, capacity }: ScalingChartsProps) {
  const [picked, setPicked] = useState<Axis>("seq");
  const [metric, setMetric] = useState<Metric>("memory");
  const inference = baseInput.calc_mode === "inference";
  const task = baseInput.model_variant.task;
  const encoder = task === "embedding" || task === "reranking";
  // Embedding/reranking load is just a batch: no concurrent users axis.
  const hasUsers = inference && !encoder;
  const mode = picked === "users" && !hasUsers ? "seq" : picked;
  const speed = metric === "speed" && inference;
  const speedUnit = task === "reranking" ? "pairs per second" : encoder ? "embeddings per second" : "tok/s per user";
  const ctx = baseInput.model_variant.context_length || 4096;

  const axis = useMemo(() => {
    if (mode === "seq")
      return {
        label: encoder ? "Input length" : "Context length",
        field: "seq_length" as const,
        points: encoder
          ? encoderSteps(ctx).filter((s, i, all) => (all.length - 1 - i) % Math.ceil(all.length / 8) === 0)
          : contextPoints(ctx),
        format: formatTokens,
      };
    if (mode === "batch")
      return { label: "Batch size", field: "batch_size" as const, points: [1, 2, 4, 8, 16, 32, 64], format: String };
    return { label: "Concurrent users", field: "concurrent_users" as const, points: [1, 2, 4, 8, 16, 32, 64], format: String };
  }, [mode, ctx, encoder]);

  // Same engine as the summary above, so the curve passes through the headline numbers.
  const data: Point[] = useMemo(
    () =>
      axis.points.map((v) => {
        const r = calculate({ ...baseInput, [axis.field]: v }, gpu);
        return {
          x: axis.format(v),
          vram: r.vram_usage,
          kvCache: r.memory_details["KV Cache"] ?? 0,
          perUser: (encoder ? r.items_per_second : r.per_user_tps) ?? null,
          throughput: r.estimated_throughput_tps,
          ttft: r.ttft_ms,
        };
      }),
    [baseInput, gpu, axis, encoder],
  );

  const tick = { fontSize: 11, fill: "hsl(var(--muted-foreground))" };

  return (
    <Section
      title={speed ? "Speed as you scale" : "Memory as you scale"}
      action={
        <>
          {inference && (
            <Segmented
              label="Metric"
              value={metric}
              onChange={setMetric}
              options={[
                { value: "memory", label: "Memory" },
                { value: "speed", label: "Speed" },
              ]}
            />
          )}
          <Segmented
            label="Scale by"
            value={mode}
            onChange={setPicked}
            options={[
              { value: "seq", label: encoder ? "Input" : "Context" },
              { value: "batch", label: "Batch" },
              ...(hasUsers ? [{ value: "users" as const, label: "Users" }] : []),
            ]}
          />
        </>
      }
    >
      <div className="h-60 w-full text-xs">
        {data.length > 0 && (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 16, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="x" tick={tick} tickLine={false} axisLine={{ stroke: "hsl(var(--border))" }} />
              <YAxis
                tick={tick}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => (speed ? `${v}` : `${v} GB`)}
                width={64}
                domain={[0, "auto"]}
              />
              <Tooltip
                cursor={{ stroke: "hsl(var(--input))" }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const p = payload[0].payload as Point;
                  return (
                    <div className="rounded-md border bg-popover px-3 py-2 text-xs tabular-nums shadow-lg">
                      <div className="mb-1 font-medium text-heading">
                        {axis.label}: {p.x}
                      </div>
                      {speed ? (
                        p.perUser == null ? (
                          <div className="text-muted-foreground">Doesn&apos;t fit in memory</div>
                        ) : (
                          <>
                            <div>
                              {p.perUser} {speedUnit}
                            </div>
                            {!encoder && <div className="text-muted-foreground">{p.throughput} tok/s combined</div>}
                            {!encoder && <div className="text-muted-foreground">{p.ttft} ms to first token</div>}
                          </>
                        )
                      ) : (
                        <>
                          <div>Total {p.vram} GB</div>
                          {!encoder && <div className="text-muted-foreground">KV cache {p.kvCache} GB</div>}
                        </>
                      )}
                    </div>
                  );
                }}
              />
              {/* Recharts only sees direct children, so no fragments here. */}
              {speed && (
                <Line
                  type="monotone"
                  dataKey="perUser"
                  stroke={PRIMARY}
                  strokeWidth={2}
                  dot={{ r: 2.5, fill: PRIMARY }}
                  isAnimationActive={false}
                />
              )}
              {!speed && (
                <ReferenceLine
                  y={capacity}
                  stroke="hsl(var(--danger))"
                  strokeDasharray="4 4"
                  ifOverflow="extendDomain"
                  label={{
                    value: `${Number(capacity.toFixed(1))} GB usable`,
                    fill: "hsl(var(--danger))",
                    fontSize: 11,
                    position: "insideTopLeft",
                  }}
                />
              )}
              {!speed && (
                <Line
                  type="monotone"
                  dataKey="vram"
                  stroke={PRIMARY}
                  strokeWidth={2}
                  dot={{ r: 2.5, fill: PRIMARY }}
                  isAnimationActive={false}
                />
              )}
              {!speed && !encoder && (
                <Line
                  type="monotone"
                  dataKey="kvCache"
                  stroke={SECONDARY}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
        {speed ? (
          <span className="flex items-center gap-2">
            <span className="h-0.5 w-4" style={{ background: PRIMARY }} /> {speedUnit[0].toUpperCase() + speedUnit.slice(1)}{" "}
            (gaps: doesn&apos;t fit)
          </span>
        ) : (
          <>
            <span className="flex items-center gap-2">
              <span className="h-0.5 w-4" style={{ background: PRIMARY }} /> Total memory
            </span>
            {!encoder && (
              <span className="flex items-center gap-2">
                <span className="h-0.5 w-4" style={{ background: SECONDARY }} /> KV cache
              </span>
            )}
            <span className="flex items-center gap-2">
              <span className="w-4 border-t border-dashed border-danger" /> Usable memory
            </span>
          </>
        )}
      </div>
    </Section>
  );
}
