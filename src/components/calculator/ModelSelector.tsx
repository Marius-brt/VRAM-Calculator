"use client";

import React, { useMemo, useState } from "react";
import { ModelVariant } from "@/lib/calculator/types";
import { Button } from "@/components/ui/button";
import { Chips, Picker, Section, Specs, formatTokens, matchesQuery } from "./controls";

const MINE = "Imported and custom";

export const formatParams = (b?: number | null) =>
  b == null ? "?" : b < 1 ? `${Math.round(b * 1000)}M` : `${Number(b.toFixed(b >= 100 ? 0 : 1))}B`;

const providerOf = (m: any) => (m.is_preset === 1 ? m.provider || m.family || "Other" : MINE);

type TaskFilter = "all" | "generation" | "embedding" | "reranking";
export const TASK_LABELS = { generation: "Chat", embedding: "Embedding", reranking: "Reranking" } as const;
const taskOf = (m: ModelVariant) => m.task ?? "generation";

/** Searchable model list grouped by provider; imported and custom models first. */
export function ModelPicker({
  models,
  selected: m,
  onSelect: onSelectModel,
  onOpenHfImport,
}: {
  models: any[];
  selected: ModelVariant;
  onSelect: (model: ModelVariant) => void;
  onOpenHfImport?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [task, setTask] = useState<TaskFilter>("all");

  const groups = useMemo(() => {
    const byProvider = new Map<string, any[]>();
    for (const m of models) {
      if (task !== "all" && taskOf(m) !== task) continue;
      const text = `${m.name} ${m.slug} ${m.provider ?? ""} ${m.family ?? ""} ${formatParams(m.num_of_params)} ${m.architecture ?? ""} ${
        m.task ? `${m.task} ${m.task === "embedding" ? "embed" : "rerank"}` : ""
      }`;
      if (query && !matchesQuery(text, query)) continue;
      const p = providerOf(m);
      byProvider.set(p, [...(byProvider.get(p) ?? []), m]);
    }
    // Groups alphabetically, imported and custom first; newest models first inside each group.
    const newest = (a: ModelVariant, b: ModelVariant) => (b.release_date ?? "").localeCompare(a.release_date ?? "");
    return [...byProvider]
      .sort(([a], [b]) => (a === MINE ? -1 : b === MINE ? 1 : a.localeCompare(b)))
      .map(([label, items]) => ({ label, items: items.sort(newest) }));
  }, [models, query, task]);

  return (
    <Picker
      toolbar={
        <Chips
          value={task}
          onChange={setTask}
          options={[
            { value: "all", label: "All" },
            ...Object.entries(TASK_LABELS).map(([value, label]) => ({ value: value as TaskFilter, label })),
          ]}
        />
      }
      groups={groups}
      getKey={(x) => x.slug ?? x.name ?? ""}
      selectedKey={m.slug ?? m.name}
      onSelect={onSelectModel}
      query={query}
      onQueryChange={setQuery}
      placeholder={`Search ${models.length} models, e.g. Llama 70B, Qwen, bge`}
      footer={
        onOpenHfImport && (
          <>
            Not listed?{" "}
            <button type="button" onClick={onOpenHfImport} className="text-link underline-offset-2 hover:underline">
              Import from Hugging Face
            </button>
          </>
        )
      }
      emptyState={
        <>
          No model matches.{" "}
          {onOpenHfImport && (
            <button type="button" onClick={onOpenHfImport} className="text-link underline-offset-2 hover:underline">
              Import it from Hugging Face
            </button>
          )}
        </>
      }
      renderItem={(x) => (
        <div className="flex items-baseline justify-between gap-3">
          <span className="truncate text-heading">
            {x.name}
            {x.architecture === "moe" && <span className="ml-2 text-xs text-muted-foreground">MoE</span>}
            {x.task && x.task !== "generation" && (
              <span className="ml-2 text-xs text-link">{TASK_LABELS[x.task as keyof typeof TASK_LABELS]}</span>
            )}
          </span>
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{formatParams(x.num_of_params)}</span>
        </div>
      )}
      trigger={
        <>
          <span className="block text-xs text-muted-foreground">{providerOf(m)}</span>
          <span className="block truncate text-[15px] font-medium text-heading">{m.name || "Choose a model"}</span>
        </>
      }
    />
  );
}

interface ModelSelectorProps {
  models: any[];
  selectedModel: ModelVariant;
  onSelectModel: (model: ModelVariant) => void;
  onOpenCustomize: () => void;
  onOpenHfImport: () => void;
  onOpenFindModels: () => void;
}

export function ModelSelector({
  models,
  selectedModel,
  onSelectModel,
  onOpenCustomize,
  onOpenHfImport,
  onOpenFindModels,
}: ModelSelectorProps) {
  const m = selectedModel;
  const kvHeads = m.num_key_value_heads || m.num_attention_heads || 32;
  const attention = (m.attention_structure || (kvHeads < (m.num_attention_heads || 32) ? "GQA" : "MHA")).toUpperCase();

  return (
    <Section
      title="Model"
      action={
        <>
          <Button variant="ghost" size="xs" onClick={onOpenFindModels}>
            Find models
          </Button>
          <Button variant="ghost" size="xs" onClick={onOpenCustomize}>
            Edit specs
          </Button>
        </>
      }
    >
      <ModelPicker models={models} selected={m} onSelect={onSelectModel} onOpenHfImport={onOpenHfImport} />

      <Specs
        className="grid-cols-2 sm:grid-cols-4"
        items={
          m.task === "embedding" || m.task === "reranking"
            ? [
                ["Parameters", formatParams(m.num_of_params)],
                ["Type", TASK_LABELS[m.task]],
                ["Layers", `${m.num_of_layers ?? "?"} × ${m.hidden_size ?? "?"}`],
                ["Max input", `${formatTokens(m.context_length || 512)} tokens`],
              ]
            : [
                ["Parameters", formatParams(m.num_of_params)],
                [
                  "Architecture",
                  m.architecture === "moe" ? `MoE, ${m.num_active_experts || 2} of ${m.num_experts || 8}` : "Dense",
                ],
                ["Attention", `${attention}, ${kvHeads} KV`],
                ["Context", formatTokens(m.context_length || 4096)],
              ]
        }
      />
    </Section>
  );
}
