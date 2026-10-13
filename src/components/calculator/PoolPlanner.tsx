"use client";

import React, { useMemo, useState } from "react";
import { CopyPlus, Plus, Trash2 } from "lucide-react";
import { GpuSpec, ModelVariant } from "@/lib/calculator/types";
import { MAX_COPIES, planPool, type Placement, type PoolConfig, type PoolPlan } from "@/lib/calculator/planner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Field, Num, Section, Select, Stepper, formatTokens } from "./controls";
import { GpuPicker } from "./GpuSelector";
import { Command, launch } from "./LaunchCommand";
import { ModelPicker } from "./ModelSelector";
import { contextSteps, encoderSteps } from "./ParameterInputs";
import { GPU_LINK_OPTIONS, NETWORK_OPTIONS, defaultGpuLink, isEncoderTask } from "@/lib/calculator/engine";

const MODEL_COLORS = ["#845ef7", "#22b8cf", "#f06595", "#94d82d", "#ffa94d", "#748ffc", "#cc5de8", "#20c997"];

const QUANTS: [string, string][] = [
  ["auto", "Auto"],
  ["fp16", "FP16 / BF16"],
  ["fp8", "FP8"],
  ["nvfp4", "NVFP4 (Blackwell)"],
  ["awq", "AWQ (4-bit)"],
  ["q8_0", "GGUF Q8_0"],
  ["q4_k_m", "GGUF Q4_K_M"],
];
const QUANT_LABEL = Object.fromEntries(QUANTS);
const USER_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

const newId = () => Math.random().toString(36).slice(2, 9);
const round1 = (n: number) => Number(n.toFixed(1));

/** Starting point: two models sharing a 2× RTX 4090 box. Fixed ids keep server and client renders identical. */
export function defaultPool(models: any[]): PoolConfig {
  const find = (slug: string) => models.find((m) => m.slug === slug);
  const picks = [find("llama-3-1-8b-instruct") ?? models[0], find("qwen3-14b")].filter(Boolean);
  return {
    network: "ethernet100g",
    nodes: [{ id: "node-1", gpuKey: "4090_24", count: 2, link: defaultGpuLink({ key: "4090_24" }) }],
    models: picks.map((model, i) => ({ id: `model-${i + 1}`, model, contextLength: 8192, users: 4, quant: "auto" })),
  };
}

/** Context choices for a model: its own window and the standard steps below it. */
const contextOptions = (model: ModelVariant) => {
  const max = model.context_length || 4096;
  if (isEncoderTask(model)) return encoderSteps(max);
  return [...new Set([...contextSteps(max).filter((s) => s <= max), max])];
};

function Block({
  title,
  swatch,
  onRemove,
  onDuplicate,
  children,
}: {
  title: string;
  swatch?: string;
  onRemove?: () => void;
  onDuplicate?: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3 rounded-lg border bg-background/40 p-3">
      <div className="flex min-h-7 items-center justify-between gap-2">
        <span className="mr-auto flex items-center gap-2 text-[13px] font-medium text-heading">
          {swatch && <span className="h-2.5 w-2.5 rounded-sm" style={{ background: swatch }} />}
          {title}
        </span>
        {onDuplicate && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            aria-label={`Duplicate ${title}`}
            title="Duplicate"
            onClick={onDuplicate}
          >
            <CopyPlus />
          </Button>
        )}
        {onRemove && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 hover:text-danger"
            aria-label={`Remove ${title}`}
            title="Remove"
            onClick={onRemove}
          >
            <Trash2 />
          </Button>
        )}
      </div>
      {children}
    </div>
  );
}

type PoolModel = PoolConfig["models"][number];

/** Node, tensor parallel and GPU choice for one model. Shows what Auto picked; clicking a GPU pins it. */
function PinControls({
  model: m,
  placement,
  nodes,
  setPin,
}: {
  model: PoolModel;
  placement?: Placement;
  nodes: PoolConfig["nodes"];
  setPin: (pin: PoolModel["pin"]) => void;
}) {
  const pin = m.pin ?? {};
  const label = (id: string) => `Node ${nodes.findIndex((n) => n.id === id) + 1}`;
  const autoNodes = placement ? [...new Set(placement.gpus.map((g) => g.nodeId))] : [];
  const node = nodes.find((n) => n.id === (pin.nodeId ?? (autoNodes.length === 1 ? autoNodes[0] : undefined)));
  const pinned = node && pin.nodeId && pin.gpus?.length ? pin.gpus.filter((i) => i < node.count) : null;
  const auto = !pinned && placement && node ? placement.gpus.filter((g) => g.nodeId === node.id).map((g) => g.index) : [];

  // Change some choices, keep the others; nothing left means fully automatic.
  const update = (patch: NonNullable<PoolModel["pin"]>) => {
    const next = Object.fromEntries(Object.entries({ ...pin, ...patch }).filter(([, v]) => v !== undefined));
    setPin(Object.keys(next).length ? next : undefined);
  };

  const toggle = (i: number) => {
    const base = pinned ?? auto;
    const next = base.includes(i) ? base.filter((x) => x !== i) : [...base, i].sort((a, b) => a - b);
    setPin(next.length ? { nodeId: node!.id, gpus: next } : { nodeId: node!.id });
  };

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-3 gap-2">
        <Field label="Node">
          <Select
            value={pin.nodeId ?? "auto"}
            onChange={(e) => update({ nodeId: e.target.value === "auto" ? undefined : e.target.value, gpus: undefined })}
          >
            <option value="auto">
              Auto
              {autoNodes.length ? ` (${autoNodes.map(label).join(" + ")})` : ""}
            </option>
            {nodes.map((n, i) => (
              <option key={n.id} value={n.id}>
                Node {i + 1}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Tensor parallel">
          <Select
            value={pinned ? "gpus" : (pin.tp ?? "auto")}
            onChange={(e) => update({ tp: e.target.value === "auto" ? undefined : Number(e.target.value), gpus: undefined })}
          >
            <option value="auto">Auto{placement ? ` (${placement.tp})` : ""}</option>
            {pinned && <option value="gpus">{pinned.length} (your GPUs)</option>}
            {[1, 2, 4, 8].map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </Select>
        </Field>
        <Field label="Copies">
          <Select
            value={pinned ? 1 : (pin.copies ?? "auto")}
            disabled={!!pinned}
            title={pinned ? "GPUs you pick hold one copy" : undefined}
            onChange={(e) => update({ copies: e.target.value === "auto" ? undefined : Number(e.target.value) })}
          >
            <option value="auto">Auto{placement ? ` (${placement.copies.length})` : ""}</option>
            {Array.from({ length: MAX_COPIES }, (_, i) => (
              <option key={i} value={i + 1}>
                {i + 1}
              </option>
            ))}
          </Select>
        </Field>
      </div>
      {node && node.count > 1 ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-xs text-muted-foreground">{label(node.id)} GPUs</span>
          {Array.from({ length: node.count }, (_, i) => {
            const on = pinned?.includes(i);
            const picked = auto.includes(i);
            return (
              <button
                key={i}
                type="button"
                aria-pressed={!!on}
                title={
                  on
                    ? "You picked this GPU"
                    : picked
                      ? "Picked automatically; click to choose GPUs yourself"
                      : "Click to use this GPU"
                }
                onClick={() => toggle(i)}
                className={cn(
                  "h-7 min-w-7 rounded-md border px-2 text-xs tabular-nums transition-colors",
                  on
                    ? "border-primary bg-primary text-primary-foreground"
                    : picked
                      ? "border-primary/80 text-heading"
                      : "border-input text-muted-foreground hover:text-heading",
                )}
              >
                {i}
              </button>
            );
          })}
          {m.pin && (
            <button
              type="button"
              className="ml-auto text-xs text-link underline-offset-2 hover:underline"
              onClick={() => setPin(undefined)}
            >
              Back to auto
            </button>
          )}
        </div>
      ) : autoNodes.length > 1 && !pin.nodeId ? (
        <p className="text-xs text-muted-foreground">Auto splits it across {autoNodes.map(label).join(" and ")}.</p>
      ) : null}
    </div>
  );
}

interface PoolPlannerProps {
  gpus: GpuSpec[];
  models: any[];
  pool: PoolConfig;
  setPool: React.Dispatch<React.SetStateAction<PoolConfig>>;
  onOpenHfImport: () => void;
}

export function PoolPlanner({ gpus, models, pool, setPool, onOpenHfImport }: PoolPlannerProps) {
  // The planner runs the engine a few hundred times; that's milliseconds, so it runs on every change.
  const plan: PoolPlan = useMemo(() => planPool(pool, gpus), [pool, gpus]);

  const gpuOf = (key: string) => gpus.find((g) => g.key === key);
  const nodeNumber = (id: string) => pool.nodes.findIndex((n) => n.id === id) + 1;
  const colorOf = (modelId: string) => MODEL_COLORS[pool.models.findIndex((m) => m.id === modelId) % MODEL_COLORS.length];

  const updateNode = (id: string, patch: Partial<PoolConfig["nodes"][number]>) =>
    setPool((p) => ({ ...p, nodes: p.nodes.map((n) => (n.id === id ? { ...n, ...patch } : n)) }));
  const updateModel = (id: string, patch: Partial<PoolConfig["models"][number]>) =>
    setPool((p) => ({ ...p, models: p.models.map((m) => (m.id === id ? { ...m, ...patch } : m)) }));

  const addNode = () =>
    setPool((p) => {
      // A new node copies the last one: most pools are built from identical boxes.
      const last = p.nodes[p.nodes.length - 1];
      return { ...p, nodes: [...p.nodes, { ...(last ?? { gpuKey: "4090_24", count: 1, link: "pcie4" }), id: newId() }] };
    });
  const addModel = () =>
    setPool((p) => {
      const model = models.find((m) => m.slug === "llama-3-8b") ?? models[0];
      return {
        ...p,
        models: [
          ...p.models,
          { id: newId(), model, contextLength: Math.min(8192, model.context_length || 4096), users: 1, quant: "auto" },
        ],
      };
    });

  const placed = plan?.placements.length ?? 0;
  const total = pool.models.length;
  const usedGpus = new Set(plan?.placements.flatMap((p) => p.gpus.map((g) => `${g.nodeId}:${g.index}`))).size;
  const allocated = plan?.gpus.reduce((s, g) => s + g.used, 0) ?? 0;
  const usable = plan?.gpus.reduce((s, g) => s + g.usable, 0) ?? 0;
  const tone =
    !plan || total === 0
      ? "bg-secondary text-muted-foreground"
      : placed === total
        ? "bg-success/15 text-success"
        : placed > 0
          ? "bg-warning/15 text-warning"
          : "bg-danger/15 text-danger";
  const headline =
    total === 0
      ? "Add a model to plan"
      : !plan
        ? "Planning…"
        : placed === total
          ? total === 1
            ? "The model fits"
            : total === 2
              ? "Both models fit"
              : `All ${total} models fit`
          : `${placed} of ${total} models fit`;

  return (
    <>
      <div className="rounded-xl border bg-card">
        <Section
          title="Nodes"
          action={
            <Button variant="ghost" size="xs" onClick={addNode}>
              <Plus /> Add node
            </Button>
          }
        >
          {pool.nodes.map((n, i) => {
            const gpu = gpuOf(n.gpuKey);
            const isMac = gpu?.type === "apple_silicon";
            return (
              <Block
                key={n.id}
                title={`Node ${i + 1}`}
                onDuplicate={() =>
                  setPool((p) => {
                    const nodes = [...p.nodes];
                    nodes.splice(nodes.findIndex((x) => x.id === n.id) + 1, 0, { ...n, id: newId() });
                    return { ...p, nodes };
                  })
                }
                onRemove={
                  pool.nodes.length > 1
                    ? () =>
                        setPool((p) => ({
                          ...p,
                          nodes: p.nodes.filter((x) => x.id !== n.id),
                          // Pins to a removed node fall back to auto.
                          models: p.models.map((m) => (m.pin?.nodeId === n.id ? { ...m, pin: undefined } : m)),
                        }))
                    : undefined
                }
              >
                {gpu && (
                  <GpuPicker
                    gpus={gpus}
                    currentGpu={gpu}
                    allowCustom={false}
                    onSelect={(key) => updateNode(n.id, { gpuKey: key, link: defaultGpuLink(gpuOf(key)) })}
                  />
                )}
                {isMac ? (
                  <p className="text-xs text-muted-foreground">One Mac per node. Add nodes to pool several Macs.</p>
                ) : (
                  <div className="flex flex-wrap items-end gap-3">
                    <Field group label="GPUs">
                      <Stepper
                        label={`GPUs in node ${i + 1}`}
                        value={n.count}
                        onChange={(count) => updateNode(n.id, { count })}
                        max={16}
                      />
                    </Field>
                    {n.count > 1 && (
                      <Field label="GPUs linked by" className="min-w-[10rem] flex-1">
                        <Select value={n.link} onChange={(e) => updateNode(n.id, { link: e.target.value })}>
                          {GPU_LINK_OPTIONS.map(([v, l]) => (
                            <option key={v} value={v}>
                              {l}
                            </option>
                          ))}
                        </Select>
                      </Field>
                    )}
                  </div>
                )}
              </Block>
            );
          })}
          {pool.nodes.length > 1 && (
            <Field label="Nodes linked by" hint="Only used when a model is too big for one node and is split across several.">
              <Select value={pool.network} onChange={(e) => setPool((p) => ({ ...p, network: e.target.value }))}>
                {NETWORK_OPTIONS.map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </Select>
            </Field>
          )}
        </Section>

        <Section
          title="Models"
          action={
            <>
              <Button variant="ghost" size="xs" onClick={onOpenHfImport}>
                Import from Hugging Face
              </Button>
              <Button variant="ghost" size="xs" onClick={addModel}>
                <Plus /> Add model
              </Button>
            </>
          }
        >
          {pool.models.length === 0 && (
            <p className="text-sm text-muted-foreground">Add the models you want to serve on these nodes.</p>
          )}
          {pool.models.map((m, i) => (
            <Block
              key={m.id}
              title={`Model ${i + 1}`}
              swatch={MODEL_COLORS[i % MODEL_COLORS.length]}
              onRemove={() => setPool((p) => ({ ...p, models: p.models.filter((x) => x.id !== m.id) }))}
            >
              <ModelPicker
                models={models}
                onOpenHfImport={onOpenHfImport}
                selected={m.model}
                onSelect={(model) =>
                  updateModel(m.id, { model, contextLength: Math.min(m.contextLength, model.context_length || m.contextLength) })
                }
              />
              <div className="grid grid-cols-3 gap-2">
                <Field label={isEncoderTask(m.model) ? "Input length" : "Context"}>
                  <Select value={m.contextLength} onChange={(e) => updateModel(m.id, { contextLength: Number(e.target.value) })}>
                    {contextOptions(m.model).map((s) => (
                      <option key={s} value={s}>
                        {formatTokens(s)}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label={isEncoderTask(m.model) ? "Batch" : "Users"}>
                  <Select value={m.users} onChange={(e) => updateModel(m.id, { users: Number(e.target.value) })}>
                    {USER_STEPS.map((u) => (
                      <option key={u} value={u}>
                        {u}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Precision">
                  <Select value={m.quant} onChange={(e) => updateModel(m.id, { quant: e.target.value })}>
                    {QUANTS.map(([v, l]) => (
                      <option key={v} value={v}>
                        {l}
                      </option>
                    ))}
                  </Select>
                </Field>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Field label={isEncoderTask(m.model) ? "Min items per second" : "Min tok/s per user"}>
                  <Input
                    type="number"
                    min={0}
                    placeholder="Any"
                    value={m.target?.minSpeed ?? ""}
                    onChange={(e) =>
                      updateModel(m.id, { target: { ...m.target, minSpeed: Number(e.target.value) || undefined } })
                    }
                  />
                </Field>
                {!isEncoderTask(m.model) && (
                  <Field label="Max first token (ms)">
                    <Input
                      type="number"
                      min={0}
                      placeholder="Any"
                      value={m.target?.maxTtftMs ?? ""}
                      onChange={(e) =>
                        updateModel(m.id, { target: { ...m.target, maxTtftMs: Number(e.target.value) || undefined } })
                      }
                    />
                  </Field>
                )}
              </div>
              <PinControls
                model={m}
                placement={plan?.placements.find((p) => p.modelId === m.id)}
                nodes={pool.nodes}
                setPin={(pin) => updateModel(m.id, { pin })}
              />
            </Block>
          ))}
        </Section>
      </div>

      <div className="min-w-0 rounded-xl border bg-card">
        <Section>
          <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-2">
            <div className="min-w-0">
              <h2 className="text-2xl font-semibold tracking-tight text-heading sm:text-[28px]">{headline}</h2>
              {plan && (
                <p className="mt-1.5 text-sm">
                  Using <span className="tabular-nums text-heading">{usedGpus}</span> of {plan.gpus.length} GPUs,{" "}
                  <span className="tabular-nums text-heading">
                    <Num value={allocated} /> GB
                  </span>{" "}
                  of {Math.round(usable)} GB usable memory allocated.
                  {plan.totalTps > 0 && (
                    <>
                      {" "}
                      Chat models generate{" "}
                      <span className="tabular-nums text-heading">
                        <Num value={plan.totalTps} digits={0} /> tok/s
                      </span>{" "}
                      in total.
                    </>
                  )}
                  {pool.models.some((m) => m.quant === "auto") && " Auto picks the highest precision that still fits everything."}
                </p>
              )}
            </div>
            {total > 0 && (
              <span className={cn("shrink-0 rounded-full px-2.5 py-1 text-sm font-medium tabular-nums", tone)}>
                {placed}/{total} placed
              </span>
            )}
          </div>
        </Section>

        {plan && plan.gpus.length > 0 && (
          <Section title="Memory per GPU">
            {pool.nodes.map((n, i) => {
              const rows = plan.gpus.filter((g) => g.nodeId === n.id);
              const gpu = gpuOf(n.gpuKey);
              if (!rows.length || !gpu) return null;
              return (
                <div key={n.id} className="space-y-1.5">
                  <div className="text-[13px]">
                    <span className="text-heading">Node {i + 1}</span>{" "}
                    <span className="text-muted-foreground">
                      {rows.length}× {gpu.label}
                    </span>
                  </div>
                  {rows.map((g) => (
                    <div key={g.index} className="flex items-center gap-3 text-xs">
                      <span className="w-11 shrink-0 text-muted-foreground">GPU {g.index}</span>
                      <div className="relative flex h-5 min-w-0 flex-1 gap-px overflow-hidden rounded bg-secondary">
                        {plan.placements
                          .filter((p) => p.gpus.some((r) => r.nodeId === g.nodeId && r.index === g.index))
                          .flatMap((p) => {
                            const name = pool.models.find((m) => m.id === p.modelId)?.model.name;
                            // Weights and runtime in the model's colour, its KV cache in a lighter shade.
                            return [
                              <div
                                key={p.modelId}
                                className="h-full shrink-0 transition-[width] duration-300"
                                title={`${name}: ${round1(p.perGpuGb - p.kvPerGpuGb)} GB weights and runtime`}
                                style={{
                                  width: `${((p.perGpuGb - p.kvPerGpuGb) / g.memory) * 100}%`,
                                  background: colorOf(p.modelId),
                                }}
                              />,
                              p.kvPerGpuGb > 0 && (
                                <div
                                  key={`${p.modelId}-kv`}
                                  className="h-full shrink-0 opacity-50 transition-[width] duration-300"
                                  title={`${name}: ${round1(p.kvPerGpuGb)} GB KV cache`}
                                  style={{ width: `${(p.kvPerGpuGb / g.memory) * 100}%`, background: colorOf(p.modelId) }}
                                />
                              ),
                            ];
                          })}
                        {/* Past this line is the runtime's reserve. */}
                        <div
                          className="absolute inset-y-0 w-px bg-heading/50"
                          style={{ left: `${(g.usable / g.memory) * 100}%` }}
                        />
                      </div>
                      <span className="w-24 shrink-0 text-right tabular-nums text-muted-foreground">
                        <span className="text-heading">
                          <Num value={g.used} />
                        </span>{" "}
                        / {g.memory} GB
                      </span>
                    </div>
                  ))}
                </div>
              );
            })}
            <p className="text-xs text-muted-foreground">
              Solid: weights and runtime. Lighter shade: KV cache. The thin line marks the runtime&apos;s reserve.
            </p>
          </Section>
        )}

        {plan && total > 0 && (
          <Section title="Placement">
            {pool.models.map((m) => {
              const p = plan.placements.find((x) => x.modelId === m.id);
              const miss = plan.unplaced.find((x) => x.modelId === m.id);
              if (!p) {
                return (
                  <div key={m.id} className="rounded-lg border border-danger/40 p-4">
                    <div className="flex items-center gap-2 font-medium text-heading">
                      <span className="h-2.5 w-2.5 rounded-sm" style={{ background: colorOf(m.id) }} />
                      {m.model.name}
                      <span className="text-[13px] font-normal text-danger">Doesn&apos;t fit</span>
                    </div>
                    <p className="mt-1.5 text-[13px] text-muted-foreground">{miss?.reason}</p>
                  </div>
                );
              }
              const describe = (gpus: Placement["gpus"]) => {
                const byNode = new Map<string, number[]>();
                for (const g of gpus) byNode.set(g.nodeId, [...(byNode.get(g.nodeId) ?? []), g.index]);
                return [...byNode]
                  .map(([id, idx]) => `Node ${nodeNumber(id)}, ${idx.length > 1 ? "GPUs" : "GPU"} ${idx.join(", ")}`)
                  .join(" and ");
              };
              const where =
                p.copies.length > 1 ? `${p.copies.length} copies: ${p.copies.map(describe).join("; ")}` : describe(p.gpus);
              const target = m.target?.minSpeed || m.target?.maxTtftMs ? m.target : null;
              // KV cache headroom: a tensor-parallel group grows evenly, so its tightest GPU sets the limit.
              const kvGb = p.kvPerGpuGb * p.gpus.length;
              const freeGb =
                Math.min(
                  ...p.gpus.map((r) => {
                    const g = plan.gpus.find((x) => x.nodeId === r.nodeId && x.index === r.index);
                    return g ? Math.max(0, g.usable - g.used) : 0;
                  }),
                ) * p.gpus.length;
              const gbPerUser = kvGb / m.users || 0;
              const moreUsers = gbPerUser > 0 ? Math.floor(freeGb / gbPerUser) : null;
              return (
                <div key={m.id} className="space-y-3 rounded-lg border p-4">
                  <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                    <span className="flex items-center gap-2 font-medium text-heading">
                      <span className="h-2.5 w-2.5 rounded-sm" style={{ background: colorOf(m.id) }} />
                      {m.model.name}
                    </span>
                    <span className="text-[13px] text-muted-foreground">{where}</span>
                  </div>
                  <dl className="grid grid-cols-2 gap-3 text-[13px] sm:grid-cols-4">
                    <div>
                      <dt className="text-xs text-muted-foreground">Precision</dt>
                      <dd className="mt-0.5 text-heading">{QUANT_LABEL[p.quant] ?? p.quant.toUpperCase()}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-muted-foreground">Split</dt>
                      <dd className="mt-0.5 text-heading">
                        {p.pp > 1
                          ? `${p.pp} stages × ${p.tp} GPU${p.tp > 1 ? "s" : ""}`
                          : p.tp > 1
                            ? `Tensor parallel ${p.tp}`
                            : "One GPU"}
                        {p.copies.length > 1 && `, ${p.copies.length} copies`}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-xs text-muted-foreground">Memory per GPU</dt>
                      <dd className="mt-0.5 tabular-nums text-heading">
                        <Num value={p.perGpuGb} /> GB{" "}
                        <span className="text-muted-foreground">({Math.round(p.memoryFraction * 100)}%)</span>
                      </dd>
                    </div>
                    {p.itemsPerSecond != null ? (
                      <div>
                        <dt className="text-xs text-muted-foreground">Throughput</dt>
                        <dd className="mt-0.5 tabular-nums text-heading">
                          <Num value={p.itemsPerSecond} /> {p.task === "reranking" ? "pairs/s" : "embeddings/s"}
                        </dd>
                      </div>
                    ) : (
                      <div>
                        <dt className="text-xs text-muted-foreground">
                          Per user
                          {p.copies.length > 1 ? `, ${p.usersPerCopy} per copy` : ""}
                        </dt>
                        <dd className="mt-0.5 tabular-nums text-heading">
                          <Num value={p.perUserTps} /> tok/s
                          {p.ttftMs != null && (
                            <span className="text-muted-foreground">, {p.ttftMs.toLocaleString()} ms first</span>
                          )}
                        </dd>
                      </div>
                    )}
                  </dl>
                  {p.itemsPerSecond == null && kvGb > 0 && (
                    <p className="text-[13px] text-muted-foreground">
                      KV cache{" "}
                      <span className="tabular-nums text-heading">
                        <Num value={kvGb} digits={2} /> GB
                      </span>{" "}
                      for {m.users} {m.users === 1 ? "user" : "users"} at {formatTokens(m.contextLength)}.{" "}
                      <span className="tabular-nums text-heading">
                        <Num value={freeGb} /> GB
                      </span>{" "}
                      still free on its GPUs
                      {moreUsers != null && moreUsers > 0
                        ? `, room for about ${moreUsers.toLocaleString()} more ${moreUsers === 1 ? "user" : "users"} at this context`
                        : ", not enough for another user at this context"}
                      {p.runtime === "vllm" && moreUsers ? " (raise --gpu-memory-utilization to use it)." : "."}
                    </p>
                  )}
                  {target && (
                    <p className={cn("text-[13px]", p.meetsTarget ? "text-success" : "text-warning")}>
                      {p.meetsTarget
                        ? "Meets your target."
                        : `Misses your target${target.minSpeed ? `: ${p.perUserTps ?? p.itemsPerSecond ?? 0} per ${p.itemsPerSecond != null ? "second" : "user"} (wanted ${target.minSpeed})` : ""}${target.maxTtftMs ? `, first token ${p.ttftMs ?? "?"} ms (wanted ${target.maxTtftMs})` : ""}. Add GPUs or lower the users or context.`}
                    </p>
                  )}
                  {p.copies.map((gpus, copy) => {
                    const { cmd } = launch(m.model, {
                      runtime: p.runtime,
                      quant: p.quant,
                      tp: p.tp,
                      pp: p.pp,
                      machines: p.pp,
                      memoryFraction: p.memoryFraction,
                      users: p.usersPerCopy,
                      contextLength: m.contextLength,
                      // Macs (llama.cpp) are one GPU each, and a copy spread over nodes can't be pinned with one variable.
                      devices:
                        p.runtime !== "llama_cpp" && gpus.every((g) => g.nodeId === gpus[0].nodeId)
                          ? gpus.map((g) => g.index)
                          : null,
                      copy: p.copies.length > 1 ? copy : null,
                    });
                    return cmd && <Command key={copy} cmd={cmd} />;
                  })}
                  {p.copies.length > 1 && (
                    <p className="text-xs text-muted-foreground">
                      Put the copies behind a load balancer (e.g. LiteLLM or nginx).
                    </p>
                  )}
                  {launch(m.model, {
                    runtime: p.runtime,
                    quant: p.quant,
                    tp: p.tp,
                    pp: p.pp,
                    machines: p.pp,
                    users: 1,
                    contextLength: 1,
                    devices: null,
                    copy: null,
                  }).notes.map((n) => (
                    <p key={n} className="text-xs text-muted-foreground">
                      {n}
                    </p>
                  ))}
                </div>
              );
            })}
          </Section>
        )}
      </div>
    </>
  );
}
