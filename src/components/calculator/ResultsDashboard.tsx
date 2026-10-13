"use client";

import React from "react";
import Link from "next/link";
import { REPO_URL } from "@/components/SiteFooter";
import { CalculatorOutput } from "@/lib/calculator/types";
import { cn } from "@/lib/utils";
import { Num, Section } from "./controls";

type Status = CalculatorOutput["memory_status"];

const fits = (hw: string) => `Fits on ${hw}`;
const STATUS: Record<Status, { title: (hw: string) => string; note: string; tone: string }> = {
  okay: { title: fits, note: "Plenty of headroom for longer prompts or more users.", tone: "bg-success/15 text-success" },
  moderate: { title: fits, note: "Comfortable headroom for production use.", tone: "bg-success/15 text-success" },
  high: { title: fits, note: "Tight. Long prompts or traffic spikes may queue.", tone: "bg-warning/15 text-warning" },
  very_high: {
    title: (hw) => `Barely fits on ${hw}`,
    note: "Close to the limit. Expect out-of-memory errors under load.",
    tone: "bg-caution/15 text-caution",
  },
  insufficient: {
    title: (hw) => `Doesn't fit on ${hw}`,
    note: "Add GPUs, shorten the context, or pick a smaller quantization.",
    tone: "bg-danger/15 text-danger",
  },
  error: {
    title: () => "Couldn't calculate this setup",
    note: "Check the model and hardware settings.",
    tone: "bg-danger/15 text-danger",
  },
};

const ENCODER_NOTES: Partial<Record<Status, string>> = {
  okay: "Room for longer inputs or bigger batches.",
  moderate: "Comfortable headroom.",
  high: "Tight. Bigger batches or longer inputs may run out of memory.",
  insufficient: "Lower the batch size or input length, or add GPUs.",
};

const TRAINING_NOTES: Partial<Record<Status, string>> = {
  okay: "Room for longer sequences or bigger batches.",
  moderate: "Comfortable headroom.",
  high: "Tight. Longer sequences or bigger batches may run out of memory.",
  insufficient: "Add GPUs, switch to QLoRA, or shorten the sequence length.",
};

// Fixed hues for the common parts so they read the same across setups.
const SEGMENT_COLORS: Record<string, string> = {
  "Base Model Weights": "#845ef7",
  "Model Weights": "#845ef7",
  "KV Cache": "#22b8cf",
  Activations: "#f06595",
};
const EXTRA_COLORS = ["#94d82d", "#748ffc", "#ffa94d", "#cc5de8", "#20c997"];
const segmentColor = (label: string, i: number) =>
  SEGMENT_COLORS[label] ?? (/overhead/i.test(label) ? "#868e96" : EXTRA_COLORS[i % EXTRA_COLORS.length]);

const fmt = (n: number | null | undefined, digits = 1) =>
  n == null || !Number.isFinite(n) ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: digits });

const formatHours = (h?: number | null) => {
  if (h == null || !Number.isFinite(h)) return { value: null };
  if (h < 1) return { value: h * 60, unit: "min", digits: 0 };
  if (h < 48) return { value: h, unit: "h" };
  return { value: h / 24, unit: "days" };
};

function Stat({
  label,
  value,
  digits = 1,
  unit,
  hint,
}: {
  label: string;
  value: number | null | undefined;
  digits?: number;
  unit?: string;
  hint?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[13px] text-muted-foreground">{label}</div>
      <div className="mt-1 flex items-baseline gap-1">
        <span className="text-2xl font-semibold tabular-nums tracking-tight text-heading">
          <Num value={value} digits={digits} />
        </span>
        {unit && value != null && <span className="text-sm text-muted-foreground">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

function Rows({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="divide-y divide-border/60 text-[13px]">
      {rows.map(([k, v]) => (
        <div key={k} className="flex items-baseline justify-between gap-4 py-2">
          <dt className="text-muted-foreground">{k}</dt>
          <dd className="text-right tabular-nums text-heading">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

interface ResultsDashboardProps {
  output: CalculatorOutput;
  calcMode: "inference" | "finetuning";
  hardwareLabel: string;
  users: number;
  chart: React.ReactNode;
  /** Launch commands for this setup. */
  run?: React.ReactNode;
}

export function ResultsDashboard({ output, calcMode, hardwareLabel, users, chart, run }: ResultsDashboardProps) {
  const used = output.vram_usage;
  const capacity = output.total_gpu_memory;
  // The runtime keeps part of each GPU in reserve; fit is judged against what's left.
  const limit = output.usable_gpu_memory;
  const status = STATUS[output.memory_status] ?? STATUS.error;
  const scale = Math.max(capacity, used) || 1;
  const over = used > limit;
  const free = Math.max(0, limit - used);
  const pos = (gb: number) => `${(gb / scale) * 100}%`;
  const breakdown = (output.memory_breakdown ?? []).filter((b) => b.size_gb > 0);
  const kv = output.concurrency;
  const encoder = output.task === "embedding" || output.task === "reranking";

  return (
    <>
      <Section>
        <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-2">
          <div className="min-w-0">
            <h2 className="text-2xl font-semibold tracking-tight text-heading sm:text-[28px]">{status.title(hardwareLabel)}</h2>
            <p className="mt-1.5 text-sm">
              <span className="tabular-nums text-heading">
                <Num value={used} digits={2} /> GB
              </span>{" "}
              of <span className="tabular-nums">{fmt(limit, 2)} GB</span>
              {limit < capacity * 0.99 ? ` usable (${fmt(capacity, 0)} GB total)` : ""}
              {output.gpu_count > 1 ? `, ${fmt(output.per_gpu_usage, 1)} GB on each GPU` : ""}.{" "}
              {(encoder && ENCODER_NOTES[output.memory_status]) ||
                (calcMode === "finetuning" && TRAINING_NOTES[output.memory_status]) ||
                status.note}
            </p>
            {output.warnings.map((w) => (
              <p key={w} className="mt-1.5 text-xs text-warning">
                {w}
              </p>
            ))}
          </div>
          <span className={cn("shrink-0 rounded-full px-2.5 py-1 text-sm font-medium tabular-nums", status.tone)}>
            <Num value={output.actual_vram_percentage} digits={0} />% used
          </span>
        </div>

        <div className="pt-2">
          <div
            className="relative flex h-8 w-full gap-px overflow-hidden rounded-md bg-secondary"
            role="img"
            aria-label={`${fmt(used, 2)} of ${fmt(capacity, 2)} GB used`}
          >
            {breakdown.map((b, i) => (
              <div
                key={b.label}
                title={`${b.label}: ${b.size_gb} GB`}
                className="h-full shrink-0 transition-[width] duration-300"
                style={{ width: pos(b.size_gb), background: segmentColor(b.label, i) }}
              />
            ))}
            {over && (
              <div
                className="absolute inset-y-0"
                style={{
                  left: pos(limit),
                  width: `calc(${pos(used)} - ${pos(limit)})`,
                  background: "repeating-linear-gradient(135deg, transparent 0 5px, hsl(var(--background) / 0.6) 5px 10px)",
                }}
              />
            )}
            {(over || limit < capacity * 0.99) && (
              <div className="absolute inset-y-0 w-0.5 -translate-x-1/2 bg-heading" style={{ left: pos(limit) }} />
            )}
          </div>
          {(over || limit < capacity * 0.99) && (
            <div className="relative mt-1 h-4 text-xs text-muted-foreground">
              <span
                className={cn(
                  "absolute whitespace-nowrap tabular-nums",
                  limit / scale > 0.85 ? "-translate-x-full" : "-translate-x-1/2",
                )}
                style={{ left: pos(limit) }}
              >
                {fmt(limit, 1)} GB usable
              </span>
            </div>
          )}
        </div>

        <ul className="grid gap-x-8 gap-y-1.5 text-[13px] sm:grid-cols-2">
          {breakdown.map((b, i) => (
            <li key={b.label} className="flex items-center gap-2.5">
              <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: segmentColor(b.label, i) }} />
              <span className="min-w-0 flex-1 truncate">{b.label}</span>
              <span className="tabular-nums text-heading">{fmt(b.size_gb, 2)} GB</span>
              <span className="w-10 text-right tabular-nums text-muted-foreground">{fmt(b.value, 0)}%</span>
            </li>
          ))}
          {!over && (
            <li className="flex items-center gap-2.5">
              <span className="h-2.5 w-2.5 shrink-0 rounded-sm border border-input bg-secondary" />
              <span className="min-w-0 flex-1 truncate">Free</span>
              <span className="tabular-nums text-heading">{fmt(free, 2)} GB</span>
              <span className="w-10 text-right tabular-nums text-muted-foreground">
                {fmt(limit ? (free / limit) * 100 : 0, 0)}%
              </span>
            </li>
          )}
        </ul>
      </Section>

      <Section title={encoder ? "Throughput" : calcMode === "inference" ? "Speed" : "Training"}>
        {encoder ? (
          <div className="grid grid-cols-2 gap-x-6 gap-y-5 md:grid-cols-4">
            <Stat
              label={output.task === "reranking" ? "Pairs scored" : "Embeddings"}
              value={output.items_per_second}
              unit="per s"
              hint="With full batches"
            />
            <Stat
              label="Tokens processed"
              value={output.tokens_per_second_processed}
              digits={0}
              unit="tok/s"
              hint="Input tokens"
            />
            <Stat label="Batch latency" value={output.batch_latency_ms} digits={0} unit="ms" hint="One full batch" />
            <Stat label="Single request" value={output.single_latency_ms} unit="ms" hint="One input alone" />
          </div>
        ) : calcMode === "inference" ? (
          <div className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3 xl:grid-cols-5">
            <Stat
              label="Per user"
              value={output.per_user_tps}
              unit="tok/s"
              hint={`Each of ${users} ${users === 1 ? "user" : "users"}`}
            />
            <Stat
              label="Throughput"
              value={output.estimated_throughput_tps}
              unit="tok/s"
              hint={output.replicas > 1 ? `${output.replicas} copies combined` : "All users combined"}
            />
            <Stat label="Single request" value={output.estimated_latency_tps} unit="tok/s" hint="When running alone" />
            <Stat label="First token" value={output.ttft_ms} digits={0} unit="ms" hint="Prompt processing" />
            <Stat label="Token latency" value={output.ms_per_token} unit="ms" hint="Between tokens" />
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-x-6 gap-y-5 md:grid-cols-4">
            <Stat label="Training speed" value={output.training_tps} digits={0} unit="tok/s" />
            <Stat label="Samples" value={output.samples_per_second} digits={2} unit="/s" />
            <Stat label="Steps" value={output.steps_per_second} digits={3} unit="/s" />
            <Stat label="Total time" {...formatHours(output.total_training_time_hours)} hint="All epochs" />
          </div>
        )}
        {!encoder && over && (calcMode === "inference" ? output.estimated_latency_tps : output.training_tps) == null && (
          <p className="text-xs text-muted-foreground">Speed is only estimated for setups that fit in memory.</p>
        )}
        {!encoder && calcMode === "inference" && output.estimated_latency_tps != null && (
          <p className="text-xs text-muted-foreground">
            Estimates{" "}
            <Link href="/methodology#calibration" className="text-link underline-offset-2 hover:underline">
              fitted to published benchmarks
            </Link>
            . Measured your own setup?{" "}
            <a href={`${REPO_URL}/issues/new?template=benchmark.yml`} className="text-link underline-offset-2 hover:underline">
              Share the numbers
            </a>{" "}
            to make them more accurate.
          </p>
        )}
      </Section>

      {run}

      {chart}

      <div className="grid border-t md:grid-cols-2 md:divide-x">
        {calcMode === "inference" && kv && (
          <Section title="Serving capacity" className="border-t-0">
            <Rows
              rows={[
                ["Users that fit at this context", kv.maxUsers >= 9999 ? "Not limited by KV cache" : fmt(kv.maxUsers, 0)],
                ["KV cache per user", `${fmt(kv.kvPerUserGb, 2)} GB`],
                ["KV cache space still free", `${fmt(kv.kvFreeGb, 1)} GB`],
                ["Requests per second", `${fmt(kv.requestsPerSecond, 2)} at 512 output tokens`],
                [
                  "Risk of queueing users",
                  <span key="risk" className={kv.preemptionRisk === "none" ? "text-success" : "text-warning"}>
                    {kv.preemptionRisk === "none" ? "None" : kv.preemptionRisk === "high" ? "High" : "Moderate"}
                  </span>,
                ],
              ]}
            />
          </Section>
        )}
        <Section title="Cost and energy" className="border-t-0 max-md:border-t first:max-md:border-t-0">
          <Rows
            rows={[
              [
                "Cloud rental",
                output.hourly_cost_usd != null ? (
                  <>
                    $<Num key="h" value={output.hourly_cost_usd} digits={2} /> per hour, ${fmt(output.monthly_cost_usd, 0)} per
                    month
                  </>
                ) : (
                  <span className="text-muted-foreground">Not commonly rented</span>
                ),
              ],
              [
                "Electricity, always on",
                <>
                  $<Num key="e" value={output.electricity_cost_month} digits={0} /> per month at $0.15/kWh
                </>,
              ],
              ["Power draw", `${fmt(output.estimated_power_draw, 0)} W`],
              ["Energy", `${fmt(output.daily_kwh)} kWh per day`],
              ["Emissions", `${fmt(output.monthly_co2_kg)} kg CO₂e per month`],
            ]}
          />
        </Section>
      </div>
    </>
  );
}
