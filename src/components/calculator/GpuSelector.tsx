"use client";

import React, { useMemo, useState } from "react";
import { GpuSpec } from "@/lib/calculator/types";
import { GPU_LINK_OPTIONS, NETWORK_OPTIONS } from "@/lib/calculator/engine";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Chips, Field, Picker, Section, Segmented, Select, Specs, Stepper, matchesQuery } from "./controls";

const VENDORS = [
  { value: "nvidia", label: "NVIDIA" },
  { value: "amd", label: "AMD" },
  { value: "apple", label: "Apple" },
  { value: "intel", label: "Intel" },
] as const;

type VendorFilter = "all" | (typeof VENDORS)[number]["value"];

const CUSTOM_GPUS: GpuSpec[] = [
  {
    key: "custom_discrete",
    label: "Custom GPU",
    vendor: "custom",
    type: "discrete",
    memory: 0,
    bandwidth_gbs: 0,
    fp16_tflops: 0,
  },
  {
    key: "custom_apple_silicon",
    label: "Custom Apple Silicon",
    vendor: "custom",
    type: "apple_silicon",
    memory: 0,
    bandwidth_gbs: 0,
    fp16_tflops: 0,
  },
];

export const vendorName = (vendor: string) =>
  vendor === "custom" ? "Custom" : (VENDORS.find((v) => v.value === vendor)?.label ?? vendor);

const searchText = (g: GpuSpec) =>
  [
    vendorName(g.vendor),
    g.label,
    g.key,
    `${g.memory}gb ${g.memory} gb`,
    g.category === "datacenter" ? "datacenter data center server" : "",
    g.type === "apple_silicon" ? "mac macbook studio mini" : "",
  ].join(" ");

/** Searchable GPU list grouped by vendor, with vendor filter chips. */
export function GpuPicker({
  gpus,
  currentGpu,
  onSelect,
  allowCustom = true,
}: {
  gpus: GpuSpec[];
  currentGpu: GpuSpec;
  onSelect: (key: string) => void;
  allowCustom?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [vendor, setVendor] = useState<VendorFilter>("all");

  const groups = useMemo(() => {
    const listed = gpus.filter((g) => !g.key.startsWith("custom_"));
    const keep = (g: GpuSpec) => !query || matchesQuery(searchText(g), query);
    const byVendor = VENDORS.filter((v) => vendor === "all" || vendor === v.value).map((v) => ({
      label: v.label,
      items: listed.filter((g) => g.vendor === v.value && keep(g)),
    }));
    return vendor === "all" && allowCustom ? [...byVendor, { label: "Custom", items: CUSTOM_GPUS.filter(keep) }] : byVendor;
  }, [gpus, query, vendor, allowCustom]);

  return (
    <Picker
      groups={groups}
      getKey={(g) => g.key}
      selectedKey={currentGpu.key}
      onSelect={(g) => onSelect(g.key)}
      query={query}
      onQueryChange={setQuery}
      placeholder="Search GPUs, e.g. 4090, H100, M4 Max"
      emptyState="No GPU matches. Try fewer words, or pick Custom GPU."
      toolbar={
        <Chips
          value={vendor}
          onChange={setVendor}
          options={[{ value: "all", label: "All" }, ...VENDORS.map((v) => ({ value: v.value, label: v.label }))]}
        />
      }
      renderItem={(g) => (
        <div className="flex items-baseline justify-between gap-3">
          <span className="truncate text-heading">
            {g.label}
            {g.released === false && <span className="ml-2 text-xs text-warning">Upcoming</span>}
          </span>
          {g.memory > 0 && (
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{g.bandwidth_gbs.toLocaleString()} GB/s</span>
          )}
        </div>
      )}
      trigger={
        <>
          <span className="block text-xs text-muted-foreground">{vendorName(currentGpu.vendor)}</span>
          <span className="block truncate text-[15px] font-medium text-heading">{currentGpu.label}</span>
        </>
      }
    />
  );
}

const MAX_RIG_GPUS = 16;

interface GpuSelectorProps {
  /** Opens the search over every GPU in the catalog for the current model. */
  onOpenFindHardware: () => void;
  gpus: GpuSpec[];
  currentGpu: GpuSpec;
  selectedGpuKey: string;
  onSelectGpuKey: (key: string) => void;
  numGpus: number;
  setNumGpus: (count: number) => void;
  /** Copies of the model; left out where copies don't apply (fine-tuning, embedding models). */
  copies?: number;
  setCopies: (copies: number) => void;
  isCluster: boolean;
  setIsCluster: (cluster: boolean) => void;
  numNodes: number;
  setNumNodes: (nodes: number) => void;
  gpusPerNode: number;
  setGpusPerNode: (gpus: number) => void;
  tpDegree: number;
  setTpDegree: (tp: number) => void;
  ppDegree: number;
  setPpDegree: (pp: number) => void;
  interconnect: string;
  setInterconnect: (ic: string) => void;
  interNodeInterconnect: string;
  setInterNodeInterconnect: (inic: string) => void;
  customVram: number;
  setCustomVram: (v: number) => void;
  customBandwidth: number;
  setCustomBandwidth: (b: number) => void;
  customTflops: number;
  setCustomTflops: (t: number) => void;
}

export function GpuSelector(props: GpuSelectorProps) {
  const {
    gpus,
    currentGpu,
    selectedGpuKey,
    onSelectGpuKey,
    numGpus,
    setNumGpus,
    copies,
    setCopies,
    isCluster,
    setIsCluster,
    numNodes,
    setNumNodes,
    gpusPerNode,
    setGpusPerNode,
    tpDegree,
    setTpDegree,
    ppDegree,
    setPpDegree,
  } = props;
  const isCustom = selectedGpuKey.startsWith("custom_");
  const totalGpus = isCluster ? numNodes * gpusPerNode : numGpus;
  const totalMemory = Math.round(totalGpus * currentGpu.memory);
  const clusterCopies = Math.floor(totalGpus / Math.max(1, tpDegree * ppDegree));
  const divisors = Array.from({ length: numGpus }, (_, i) => i + 1).filter((d) => numGpus % d === 0);

  const switchTopology = (cluster: boolean) => {
    if (cluster === isCluster) return;
    // Carry the GPU count across so results don't jump when switching.
    if (cluster) {
      setNumNodes(1);
      setGpusPerNode(numGpus);
      setTpDegree(numGpus);
      setPpDegree(1);
    } else {
      setNumGpus(Math.min(MAX_RIG_GPUS, totalGpus));
    }
    setIsCluster(cluster);
  };

  return (
    <Section
      title="Hardware"
      action={
        <>
          <Button variant="ghost" size="xs" onClick={props.onOpenFindHardware}>
            Find hardware
          </Button>
          <Segmented
            label="Topology"
            value={isCluster ? "cluster" : "single"}
            onChange={(v) => switchTopology(v === "cluster")}
            options={[
              { value: "single", label: "One machine" },
              { value: "cluster", label: "Cluster" },
            ]}
          />
        </>
      }
    >
      <GpuPicker gpus={gpus} currentGpu={currentGpu} onSelect={onSelectGpuKey} />

      {isCustom ? (
        <div className="grid grid-cols-3 gap-3">
          <Field label="Memory (GB)">
            <Input type="number" min={1} value={props.customVram} onChange={(e) => props.setCustomVram(Number(e.target.value))} />
          </Field>
          <Field label="Bandwidth (GB/s)">
            <Input
              type="number"
              min={1}
              value={props.customBandwidth}
              onChange={(e) => props.setCustomBandwidth(Number(e.target.value))}
            />
          </Field>
          <Field label="FP16 TFLOPS">
            <Input
              type="number"
              min={1}
              value={props.customTflops}
              onChange={(e) => props.setCustomTflops(Number(e.target.value))}
            />
          </Field>
        </div>
      ) : (
        <Specs
          items={[
            [currentGpu.type === "apple_silicon" ? "Unified memory" : "Memory", `${currentGpu.memory} GB`],
            ["Bandwidth", `${currentGpu.bandwidth_gbs.toLocaleString()} GB/s`],
            ["FP16 compute", `${currentGpu.fp16_tflops.toLocaleString()} TFLOPS`],
          ]}
        />
      )}

      {isCluster ? (
        <p className="text-[13px] text-muted-foreground">
          <span className="tabular-nums text-heading">{totalGpus}</span> GPUs,{" "}
          <span className="tabular-nums text-heading">{totalMemory.toLocaleString()} GB</span> total
          {copies != null && clusterCopies > 1 && `, ${clusterCopies} copies of the model`}
        </p>
      ) : (
        <div className="flex items-center gap-3">
          <Stepper label="GPUs" value={numGpus} onChange={setNumGpus} max={MAX_RIG_GPUS} />
          <span className="text-[13px] text-muted-foreground">
            {numGpus === 1 ? "GPU" : "GPUs"}, <span className="tabular-nums text-heading">{totalMemory.toLocaleString()} GB</span>{" "}
            total
            {numGpus > 1 && ((copies ?? 1) > 1 ? `, ${copies} copies of the model` : ", split with tensor parallelism")}
          </span>
        </div>
      )}

      {!isCluster && copies != null && numGpus > 1 && (
        <Field
          label="Copies of the model"
          hint="More copies serve more users at once; fewer, wider copies answer each user faster."
        >
          <Select value={copies} onChange={(e) => setCopies(Number(e.target.value))}>
            {divisors.map((d) => (
              <option key={d} value={d}>
                {d === 1 ? `1, split over all ${numGpus} GPUs` : `${d}, on ${numGpus / d} GPU${numGpus / d > 1 ? "s" : ""} each`}
              </option>
            ))}
          </Select>
        </Field>
      )}

      {isCluster && (
        <div className="space-y-4 rounded-lg border bg-background/40 p-4">
          <div className="grid grid-cols-2 gap-x-4 gap-y-3">
            <Field group label="Nodes">
              <Stepper label="Nodes" value={numNodes} onChange={setNumNodes} max={64} />
            </Field>
            <Field group label="GPUs per node">
              <Stepper label="GPUs per node" value={gpusPerNode} onChange={setGpusPerNode} max={16} />
            </Field>
            <Field group label="Tensor parallel">
              <Stepper label="Tensor parallel degree" value={tpDegree} onChange={setTpDegree} max={64} />
            </Field>
            <Field group label="Pipeline parallel">
              <Stepper label="Pipeline parallel degree" value={ppDegree} onChange={setPpDegree} max={64} />
            </Field>
          </div>
          {(tpDegree * ppDegree > totalGpus || tpDegree > gpusPerNode) && (
            <p className="text-xs text-warning">
              {tpDegree * ppDegree > totalGpus
                ? `Tensor × pipeline parallel needs ${tpDegree * ppDegree} GPUs, but the cluster has ${totalGpus}.`
                : "Tensor parallel across nodes sends every layer over the network, which is slow."}{" "}
              <button
                type="button"
                className="text-link underline-offset-2 hover:underline"
                onClick={() => {
                  setTpDegree(gpusPerNode);
                  setPpDegree(numNodes);
                }}
              >
                Use tensor parallel {gpusPerNode} inside nodes, pipeline parallel {numNodes} across them
              </button>
            </p>
          )}
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Within a node">
              <Select value={props.interconnect} onChange={(e) => props.setInterconnect(e.target.value)}>
                {GPU_LINK_OPTIONS.map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Between nodes">
              <Select value={props.interNodeInterconnect} onChange={(e) => props.setInterNodeInterconnect(e.target.value)}>
                {NETWORK_OPTIONS.map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
        </div>
      )}
    </Section>
  );
}
