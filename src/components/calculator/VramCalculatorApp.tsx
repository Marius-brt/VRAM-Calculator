"use client";

import React, { useState, useEffect, useMemo } from "react";
import { CalculatorInput, CalculatorOutput, GpuSpec, ModelVariant, SpecMethod } from "@/lib/calculator/types";
import { calculate, isEncoderTask } from "@/lib/calculator/engine";
import { CalculatorHeader } from "./CalculatorHeader";
import { ModelSelector, TASK_LABELS } from "./ModelSelector";
import { GpuSelector } from "./GpuSelector";
import { ParameterInputs } from "./ParameterInputs";
import { ResultsDashboard } from "./ResultsDashboard";
import { RunSection } from "./LaunchCommand";
import { ScalingCharts } from "./ScalingCharts";
import { SetupOptimizerModal } from "./SetupOptimizerModal";
import { FindModelsModal } from "./FindModelsModal";
import { HuggingFaceImportModal } from "./HuggingFaceImportModal";
import { AccuracyScorecardModal } from "./AccuracyScorecardModal";
import { CustomizeModelModal } from "./CustomizeModelModal";
import { SavedConfigsModal, SavedSetup } from "./SavedConfigsModal";
import { PoolPlanner, defaultPool } from "./PoolPlanner";
import type { PoolConfig } from "@/lib/calculator/planner";
import { Section, Segmented } from "./controls";

export type View = "simple" | "advanced";
const POOL_KEY = "vram-calc:pool";
/** Hugging Face imports and edited models, kept in this browser. */
const MY_MODELS_KEY = "vram-calc:my-models";
const DEFAULT_MODEL = "llama-3-1-8b-instruct";

const DEFAULT_GPU = "4090_24";

interface VramCalculatorAppProps {
  initialModels: any[];
  initialGpus: GpuSpec[];
}

export function VramCalculatorApp({ initialModels, initialGpus }: VramCalculatorAppProps) {
  const [models, setModels] = useState<any[]>(initialModels);
  const [gpus] = useState<GpuSpec[]>(initialGpus);
  const [isDark, setIsDark] = useState(true);
  const [view, setView] = useState<View>("simple");
  const [pool, setPool] = useState<PoolConfig>(() => defaultPool(initialModels));

  // Core Calculator State
  const [pickedMode, setCalcMode] = useState<"inference" | "finetuning">("inference");
  const [selectedModel, setSelectedModel] = useState<ModelVariant>(() => {
    return models.find((m) => m.slug === DEFAULT_MODEL) ?? models[0];
  });

  // Embedding and reranking models only have an inference calculation.
  const encoder = isEncoderTask(selectedModel);
  const calcMode = encoder ? "inference" : pickedMode;
  const selectModel = (m: ModelVariant) => {
    setSelectedModel(m);
    if (isEncoderTask(m)) setSeqLength((s) => Math.min(s, m.context_length || 512));
  };

  const [selectedGpuKey, setSelectedGpuKey] = useState<string>(DEFAULT_GPU);
  const [quantization, setQuantization] = useState<string>("q4_k_m");
  const [kvCacheQuant, setKvCacheQuant] = useState<string>("fp16");
  const [seqLength, setSeqLength] = useState<number>(2048);
  const [batchSize, setBatchSize] = useState<number>(1);
  const [concurrentUsers, setConcurrentUsers] = useState<number>(1);
  const [numGpus, setNumGpus] = useState<number>(1);
  const [copies, setCopies] = useState<number>(1);
  // The engine only accepts serving runtimes for inference and training runtimes for fine-tuning.
  const [servingRuntime, setServingRuntime] = useState<string>("vllm");
  const [trainingRuntime, setTrainingRuntime] = useState<string>("unsloth");
  const [frameworkPreset, setFrameworkPreset] =
    calcMode === "inference" ? [servingRuntime, setServingRuntime] : [trainingRuntime, setTrainingRuntime];

  // Cluster & Parallelism
  const [isCluster, setIsCluster] = useState<boolean>(false);
  const [numNodes, setNumNodes] = useState<number>(1);
  const [gpusPerNode, setGpusPerNode] = useState<number>(1);
  const [tpDegree, setTpDegree] = useState<number>(1);
  const [ppDegree, setPpDegree] = useState<number>(1);
  const [interconnect, setInterconnect] = useState<string>("nvlink_gen4");
  const [interNodeInterconnect, setInterNodeInterconnect] = useState<string>("infiniband_hdr");

  // Custom hardware overrides
  const [customVram, setCustomVram] = useState<number>(24);
  const [customBandwidth, setCustomBandwidth] = useState<number>(1008);
  const [customTflops, setCustomTflops] = useState<number>(165);

  // Advanced toggles
  const [enablePrefixCaching, setEnablePrefixCaching] = useState<boolean>(false);
  const [sharedPrefixRatio, setSharedPrefixRatio] = useState<number>(0.3);
  const [enableOffloading, setEnableOffloading] = useState<boolean>(false);
  const [offloadPercentage, setOffloadPercentage] = useState<number>(30);
  const [offloadKvCache, setOffloadKvCache] = useState<boolean>(false);
  const [enableSpeculative, setEnableSpeculative] = useState<boolean>(false);
  const [specMethod, setSpecMethod] = useState<SpecMethod>("draft");
  const [specDraftTokens, setSpecDraftTokens] = useState<number>(4);
  const [specDraftParams, setSpecDraftParams] = useState<number>(1.5);
  const [specAcceptanceRate, setSpecAcceptanceRate] = useState<number>(0.7);

  // Fine-tuning states
  const [ftMethod, setFtMethod] = useState<"full" | "lora" | "qlora">("lora");
  const [ftQuant, setFtQuant] = useState<string>("fp16");
  const [loraRank, setLoraRank] = useState<number>(16);
  const [gradAccumSteps, setGradAccumSteps] = useState<number>(1);
  const [optimizerType, setOptimizerType] = useState<"adamw" | "adamw_8bit" | "paged_adamw">("adamw");
  const [zeroStage, setZeroStage] = useState<number>(0);
  const [gradCheckpointing, setGradCheckpointing] = useState<boolean>(true);
  const [flashAttention, setFlashAttention] = useState<boolean>(true);
  const [numSamples, setNumSamples] = useState<number>(10000);
  const [epochs, setEpochs] = useState<number>(3);

  // Modals state
  const [optimizerOpen, setOptimizerOpen] = useState(false);
  const [findModelsOpen, setFindModelsOpen] = useState(false);
  const [scorecardOpen, setScorecardOpen] = useState(false);
  const [savedConfigsOpen, setSavedConfigsOpen] = useState(false);
  const [hfImportOpen, setHfImportOpen] = useState(false);
  const [customizeOpen, setCustomizeOpen] = useState(false);

  // Find current GPU spec
  const currentGpu = useMemo((): GpuSpec => {
    if (selectedGpuKey === "custom_discrete") {
      return {
        key: "custom_discrete",
        label: "Custom GPU",
        vendor: "custom",
        type: "discrete",
        memory: customVram || 24,
        bandwidth_gbs: customBandwidth || 1008,
        fp16_tflops: customTflops || 165,
        hourly_price_usd: 0.5,
      };
    }
    if (selectedGpuKey === "custom_apple_silicon") {
      return {
        key: "custom_apple_silicon",
        label: "Custom Apple Silicon",
        vendor: "custom",
        type: "apple_silicon",
        memory: customVram || 64,
        bandwidth_gbs: customBandwidth || 400,
        fp16_tflops: customTflops || 40,
        hourly_price_usd: 0.3,
      };
    }
    return gpus.find((g) => g.key === selectedGpuKey) || gpus.find((g) => g.key === DEFAULT_GPU) || gpus[0];
  }, [gpus, selectedGpuKey, customVram, customBandwidth, customTflops]);

  // In cluster mode the GPU count comes from the topology.
  const totalGpus = isCluster ? numNodes * gpusPerNode : numGpus;
  // Copies split one machine's GPUs evenly (a count that no longer divides them means one copy).
  // In a cluster they follow from tensor × pipeline parallel.
  const showCopies = calcMode === "inference" && !encoder;
  const replicas = !isCluster && showCopies && numGpus % copies === 0 ? copies : 1;

  // Construct current calculator input
  const currentInput = useMemo((): CalculatorInput => {
    return {
      calc_mode: calcMode,
      gpu_key: selectedGpuKey,
      model_variant: selectedModel,
      quantization,
      kv_cache_quantization: (kvCacheQuant as any) || "fp16",
      batch_size: batchSize,
      seq_length: seqLength,
      concurrent_users: concurrentUsers,
      num_gpus: totalGpus,
      framework_preset: frameworkPreset as any,
      num_nodes: isCluster ? numNodes : undefined,
      gpus_per_node: isCluster ? gpusPerNode : undefined,
      // Each copy splits its share of the machine's GPUs with tensor parallelism (the vLLM default).
      tp_degree: isCluster ? tpDegree : totalGpus / replicas,
      pp_degree: isCluster ? ppDegree : 1,
      interconnect_type: interconnect,
      inter_node_interconnect: interNodeInterconnect,
      custom_vram: selectedGpuKey.startsWith("custom_") ? customVram : undefined,
      custom_bandwidth: selectedGpuKey.startsWith("custom_") ? customBandwidth : undefined,
      custom_tflops: selectedGpuKey.startsWith("custom_") ? customTflops : undefined,
      enable_prefix_caching: enablePrefixCaching,
      shared_prefix_ratio: sharedPrefixRatio,
      enable_continuous_batching: true,
      enable_offloading: enableOffloading,
      percentage_offload: offloadPercentage / 100,
      offload_kv_cache: offloadKvCache,
      enable_speculative_decoding: enableSpeculative,
      speculative_method: specMethod,
      speculative_draft_tokens: specDraftTokens,
      speculative_draft_model_params: specDraftParams,
      speculative_acceptance_rate: specAcceptanceRate,
      finetuning_method: ftMethod,
      fine_tuning_quantization: ftQuant,
      lora_rank: loraRank,
      gradient_accumulation_steps: gradAccumSteps,
      num_samples: numSamples,
      num_epochs: epochs,
      optimization_config: {
        gradient_checkpointing: gradCheckpointing,
        flash_attention: flashAttention,
        use_8bit_optimizer: optimizerType === "adamw_8bit",
        use_paged_optimizer: optimizerType === "paged_adamw",
        zero_stage: zeroStage,
      },
    };
  }, [
    calcMode,
    selectedGpuKey,
    selectedModel,
    quantization,
    kvCacheQuant,
    batchSize,
    seqLength,
    concurrentUsers,
    totalGpus,
    replicas,
    frameworkPreset,
    isCluster,
    numNodes,
    gpusPerNode,
    tpDegree,
    ppDegree,
    interconnect,
    interNodeInterconnect,
    customVram,
    customBandwidth,
    customTflops,
    enablePrefixCaching,
    sharedPrefixRatio,
    enableOffloading,
    offloadPercentage,
    offloadKvCache,
    enableSpeculative,
    specMethod,
    specDraftTokens,
    specDraftParams,
    specAcceptanceRate,
    ftMethod,
    ftQuant,
    loraRank,
    gradAccumSteps,
    numSamples,
    epochs,
    gradCheckpointing,
    flashAttention,
    optimizerType,
    zeroStage,
  ]);

  // The engine is pure and fast, so results update synchronously as inputs change.
  const output: CalculatorOutput = useMemo(() => calculate(currentInput, currentGpu), [currentInput, currentGpu]);

  // Imported and edited models live in this browser; merge them in on load.
  useEffect(() => {
    try {
      const mine: ModelVariant[] = JSON.parse(localStorage.getItem(MY_MODELS_KEY) || "[]");
      if (mine.length) setModels((prev) => [...mine, ...prev.filter((m) => !mine.some((x) => x.slug === m.slug))]);
    } catch {}
  }, []);
  const addMyModel = (m: ModelVariant) => {
    setModels((prev) => [m, ...prev.filter((x) => x.slug !== m.slug)]);
    try {
      const mine: ModelVariant[] = JSON.parse(localStorage.getItem(MY_MODELS_KEY) || "[]");
      localStorage.setItem(MY_MODELS_KEY, JSON.stringify([m, ...mine.filter((x) => x.slug !== m.slug)]));
    } catch {}
  };

  // Theme: dark by default, remembered per browser.
  useEffect(() => {
    try {
      if (localStorage.getItem("theme") === "light") setIsDark(false);
    } catch {}
  }, []);
  useEffect(() => {
    document.documentElement.classList.toggle("light", !isDark);
    try {
      localStorage.setItem("theme", isDark ? "dark" : "light");
    } catch {}
  }, [isDark]);

  // Stored pools and setups hold a copy of each model; prefer the current catalog entry.
  const latest = (m: ModelVariant) => (m?.slug && models.find((x) => x.slug === m.slug)) || m;
  const withLatestModels = (p: PoolConfig): PoolConfig => ({
    ...p,
    models: p.models.map((m) => ({ ...m, model: latest(m.model) })),
  });

  // The pool is remembered per browser. Must stay above the pool-writing effect.
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(POOL_KEY) || "null");
      if (saved?.nodes && saved?.models) setPool(withLatestModels(saved));
    } catch {}
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(POOL_KEY, JSON.stringify(pool));
    } catch {}
  }, [pool]);

  // Read URL query params once on mount. Must stay above the URL-writing effect.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("view") === "advanced") setView("advanced");
    const found = models.find((m) => m.slug === params.get("model"));
    if (found) setSelectedModel(found);
    const gKey = params.get("gpu");
    if (gKey && (gKey.startsWith("custom_") || gpus.some((g) => g.key === gKey))) setSelectedGpuKey(gKey);
    const q = params.get("quant");
    if (q) setQuantization(q);
    const s = Number(params.get("seq"));
    if (s > 0) setSeqLength(s);
    const n = Number(params.get("gpus"));
    if (n > 0) setNumGpus(Math.min(16, n));
    const c = Number(params.get("copies"));
    if (c > 1) setCopies(c);
    const m = params.get("mode");
    if (m === "inference" || m === "finetuning") setCalcMode(m);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep the address bar in sync so "Copy link" shares the current setup.
  useEffect(() => {
    const params =
      view === "advanced"
        ? new URLSearchParams({ view })
        : new URLSearchParams({
            mode: calcMode,
            gpu: selectedGpuKey,
            gpus: String(numGpus),
            quant: quantization,
            seq: String(seqLength),
          });
    if (view === "simple" && selectedModel.slug) params.set("model", selectedModel.slug);
    if (view === "simple" && replicas > 1) params.set("copies", String(replicas));
    window.history.replaceState(null, "", `?${params}`);
  }, [view, calcMode, selectedGpuKey, numGpus, replicas, quantization, seqLength, selectedModel.slug]);

  const loadSetup = (setup: SavedSetup) => {
    if (setup.kind === "pool") {
      setPool(withLatestModels(setup.config));
      setView("advanced");
      return;
    }
    const loaded = setup.config;
    setView("simple");
    if (loaded.model_variant) setSelectedModel(latest(loaded.model_variant));
    if (loaded.gpu_key) setSelectedGpuKey(loaded.gpu_key);
    if (loaded.quantization) setQuantization(loaded.quantization);
    if (loaded.kv_cache_quantization) setKvCacheQuant(loaded.kv_cache_quantization);
    if (loaded.seq_length) setSeqLength(loaded.seq_length);
    if (loaded.batch_size) setBatchSize(loaded.batch_size);
    if (loaded.concurrent_users) setConcurrentUsers(loaded.concurrent_users);
    if (loaded.framework_preset)
      (loaded.calc_mode === "finetuning" ? setTrainingRuntime : setServingRuntime)(loaded.framework_preset);
    if (loaded.calc_mode) setCalcMode(loaded.calc_mode);
    if (loaded.num_nodes) {
      setIsCluster(true);
      setNumNodes(loaded.num_nodes);
      setGpusPerNode(loaded.gpus_per_node || 1);
      setTpDegree(loaded.tp_degree || 1);
      setPpDegree(loaded.pp_degree || 1);
    } else {
      setIsCluster(false);
      if (loaded.num_gpus) setNumGpus(Math.min(16, loaded.num_gpus));
      setCopies(loaded.num_gpus && loaded.tp_degree ? Math.max(1, Math.round(loaded.num_gpus / loaded.tp_degree)) : 1);
    }
  };

  const downloadJson = () => {
    if (!output) return;
    const blob = new Blob([JSON.stringify({ input: currentInput, output }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `vram-calc-${selectedModel.name}-${currentGpu.label}.json`;
    a.click();
  };

  return (
    <div className="flex flex-1 flex-col">
      <CalculatorHeader
        onOpenSavedConfigs={() => setSavedConfigsOpen(true)}
        isDark={isDark}
        setIsDark={setIsDark}
        view={view}
        setView={setView}
      />

      <main className="mx-auto grid w-full max-w-[1320px] flex-1 items-start gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[400px_minmax(0,1fr)] xl:grid-cols-[440px_minmax(0,1fr)]">
        {view === "advanced" ? (
          <PoolPlanner gpus={gpus} models={models} pool={pool} setPool={setPool} onOpenHfImport={() => setHfImportOpen(true)} />
        ) : (
          <>
            <div className="rounded-xl border bg-card">
              <Section>
                {encoder ? (
                  <p className="text-[13px] text-muted-foreground">
                    {TASK_LABELS[selectedModel.task as "embedding" | "reranking"]} model: sized for batched inference, one forward
                    pass per {selectedModel.task === "reranking" ? "query and document pair" : "input"}.
                  </p>
                ) : (
                  <Segmented
                    label="Calculation"
                    value={calcMode}
                    onChange={setCalcMode}
                    options={[
                      { value: "inference", label: "Inference" },
                      { value: "finetuning", label: "Fine-tuning" },
                    ]}
                  />
                )}
              </Section>

              <ModelSelector
                models={models}
                selectedModel={selectedModel}
                onSelectModel={selectModel}
                onOpenCustomize={() => setCustomizeOpen(true)}
                onOpenHfImport={() => setHfImportOpen(true)}
                onOpenFindModels={() => setFindModelsOpen(true)}
              />

              <GpuSelector
                onOpenFindHardware={() => setOptimizerOpen(true)}
                gpus={gpus}
                currentGpu={currentGpu}
                selectedGpuKey={selectedGpuKey}
                onSelectGpuKey={setSelectedGpuKey}
                numGpus={numGpus}
                setNumGpus={setNumGpus}
                copies={showCopies ? replicas : undefined}
                setCopies={setCopies}
                isCluster={isCluster}
                setIsCluster={setIsCluster}
                numNodes={numNodes}
                setNumNodes={setNumNodes}
                gpusPerNode={gpusPerNode}
                setGpusPerNode={setGpusPerNode}
                tpDegree={tpDegree}
                setTpDegree={setTpDegree}
                ppDegree={ppDegree}
                setPpDegree={setPpDegree}
                interconnect={interconnect}
                setInterconnect={setInterconnect}
                interNodeInterconnect={interNodeInterconnect}
                setInterNodeInterconnect={setInterNodeInterconnect}
                customVram={customVram}
                setCustomVram={setCustomVram}
                customBandwidth={customBandwidth}
                setCustomBandwidth={setCustomBandwidth}
                customTflops={customTflops}
                setCustomTflops={setCustomTflops}
              />

              <ParameterInputs
                onOpenScorecard={() => setScorecardOpen(true)}
                calcMode={calcMode}
                modelContext={selectedModel.context_length || 4096}
                task={selectedModel.task}
                quantization={quantization}
                setQuantization={setQuantization}
                kvCacheQuant={kvCacheQuant}
                setKvCacheQuant={setKvCacheQuant}
                seqLength={seqLength}
                setSeqLength={setSeqLength}
                batchSize={batchSize}
                setBatchSize={setBatchSize}
                concurrentUsers={concurrentUsers}
                setConcurrentUsers={setConcurrentUsers}
                frameworkPreset={frameworkPreset}
                setFrameworkPreset={setFrameworkPreset}
                enablePrefixCaching={enablePrefixCaching}
                setEnablePrefixCaching={setEnablePrefixCaching}
                sharedPrefixRatio={sharedPrefixRatio}
                setSharedPrefixRatio={setSharedPrefixRatio}
                enableOffloading={enableOffloading}
                setEnableOffloading={setEnableOffloading}
                offloadPercentage={offloadPercentage}
                setOffloadPercentage={setOffloadPercentage}
                offloadKvCache={offloadKvCache}
                setOffloadKvCache={setOffloadKvCache}
                enableSpeculative={enableSpeculative}
                setEnableSpeculative={setEnableSpeculative}
                specMethod={specMethod}
                setSpecMethod={setSpecMethod}
                specDraftTokens={specDraftTokens}
                setSpecDraftTokens={setSpecDraftTokens}
                specDraftParams={specDraftParams}
                setSpecDraftParams={setSpecDraftParams}
                specAcceptanceRate={specAcceptanceRate}
                setSpecAcceptanceRate={setSpecAcceptanceRate}
                ftMethod={ftMethod}
                setFtMethod={setFtMethod}
                ftQuant={ftQuant}
                setFtQuant={setFtQuant}
                loraRank={loraRank}
                setLoraRank={setLoraRank}
                gradAccumSteps={gradAccumSteps}
                setGradAccumSteps={setGradAccumSteps}
                optimizerType={optimizerType}
                setOptimizerType={setOptimizerType}
                zeroStage={zeroStage}
                setZeroStage={setZeroStage}
                gradCheckpointing={gradCheckpointing}
                setGradCheckpointing={setGradCheckpointing}
                flashAttention={flashAttention}
                setFlashAttention={setFlashAttention}
                numSamples={numSamples}
                setNumSamples={setNumSamples}
                epochs={epochs}
                setEpochs={setEpochs}
              />
            </div>

            <div className="min-w-0 rounded-xl border bg-card">
              <ResultsDashboard
                output={output}
                calcMode={calcMode}
                hardwareLabel={`${totalGpus}× ${currentGpu.label}`}
                users={concurrentUsers}
                chart={<ScalingCharts baseInput={currentInput} gpu={currentGpu} capacity={output.usable_gpu_memory} />}
                run={<RunSection input={currentInput} gpu={currentGpu} output={output} />}
              />
            </div>
          </>
        )}
      </main>

      {/* Modals */}
      <FindModelsModal
        open={findModelsOpen}
        onClose={() => setFindModelsOpen(false)}
        baseInput={currentInput}
        gpu={currentGpu}
        hardwareLabel={`${totalGpus}× ${currentGpu.label}`}
        models={models}
        onUse={(m, quant) => {
          setCalcMode("inference");
          selectModel(m);
          setQuantization(quant);
        }}
      />

      <SetupOptimizerModal
        open={optimizerOpen}
        onClose={() => setOptimizerOpen(false)}
        baseInput={currentInput}
        gpus={gpus}
        onApplyConfig={({ gpuKey, numGpus, quantization }) => {
          setSelectedGpuKey(gpuKey);
          setIsCluster(false);
          setNumGpus(numGpus);
          setQuantization(quantization);
        }}
      />

      <HuggingFaceImportModal
        open={hfImportOpen}
        onClose={() => setHfImportOpen(false)}
        onModelImported={(newModel) => {
          addMyModel(newModel);
          if (view === "advanced") {
            // Add it to the pool so it gets placed straight away.
            const contextLength = Math.min(8192, newModel.context_length || 4096);
            setPool((p) => ({
              ...p,
              models: [...p.models, { id: crypto.randomUUID(), model: newModel, contextLength, users: 1, quant: "auto" }],
            }));
          } else {
            selectModel(newModel);
          }
        }}
      />

      <AccuracyScorecardModal
        open={scorecardOpen}
        onClose={() => setScorecardOpen(false)}
        current={quantization}
        onSelectQuant={(q) => setQuantization(q)}
      />

      <CustomizeModelModal
        open={customizeOpen}
        onClose={() => setCustomizeOpen(false)}
        initialModel={selectedModel}
        onSaveModel={(edited) => {
          // Save as a separate custom entry; editing a custom model again replaces it.
          const isCustom = edited.slug?.startsWith("custom:");
          const m = {
            ...edited,
            is_preset: 0,
            slug: isCustom ? edited.slug : `custom:${edited.slug ?? Date.now()}`,
            name: !isCustom && edited.name === selectedModel.name ? `${edited.name} (custom)` : edited.name,
          };
          addMyModel(m);
          setSelectedModel(m);
        }}
      />

      <SavedConfigsModal
        open={savedConfigsOpen}
        onClose={() => setSavedConfigsOpen(false)}
        current={view === "advanced" ? { kind: "pool", config: pool } : { kind: calcMode, config: currentInput }}
        onLoad={loadSetup}
        onDownload={view === "simple" ? downloadJson : undefined}
      />
    </div>
  );
}
