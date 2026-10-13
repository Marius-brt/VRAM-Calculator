import type { Metadata } from "next";
import Link from "next/link";
import { Github } from "lucide-react";
import { CPU_GBPS, FITTED, KV_BYTES, LINKS, RUNTIMES, TRAINERS, WEIGHT_BITS } from "@/lib/calculator/engine";
import { BENCHMARKS, predict, typicalError, type Benchmark } from "@/lib/calculator/benchmarks";
import { GPUS, MODELS } from "@/lib/data";
import { AUTHOR, REPO_URL } from "@/components/SiteFooter";

export const metadata: Metadata = {
  title: "Methodology | VRAM Calculator",
  description:
    "Every formula behind the VRAM Calculator's memory, speed, fine-tuning and cost estimates, and the data they are fitted to.",
};

// Tables below are built from the engine's own constants and the benchmark file, so they can't drift from the code.

const SECTIONS = [
  ["overview", "Overview"],
  ["model", "Model shape"],
  ["memory", "Memory"],
  ["decode", "Decode speed"],
  ["prefill", "Prompt processing"],
  ["calibration", "Calibration"],
  ["finetuning", "Fine-tuning"],
  ["encoders", "Embedding and reranking"],
  ["planner", "Several models"],
  ["cost", "Cost and energy"],
  ["commands", "Launch commands"],
  ["sources", "Sources"],
  ["cite", "Cite"],
] as const;

const RUNTIME_NAMES: Record<string, string> = {
  vllm: "vLLM",
  sglang: "SGLang",
  tensorrt_llm: "TensorRT-LLM",
  llama_cpp: "llama.cpp",
  ollama: "Ollama",
  hf_transformers: "Transformers",
  generic: "Other",
};
const TRAINER_NAMES: Record<string, string> = {
  unsloth: "Unsloth",
  torchtune: "torchtune",
  hf_trainer: "Transformers Trainer",
  generic: "Other",
};
const FORMATS: [string, string][] = [
  ["fp16", "FP16 / BF16"],
  ["fp8", "FP8"],
  ["int8", "INT8"],
  ["q8_0", "GGUF Q8_0"],
  ["nvfp6", "NVFP6"],
  ["q6_k", "GGUF Q6_K"],
  ["q5_k_m", "GGUF Q5_K_M"],
  ["q4_k_m", "GGUF Q4_K_M"],
  ["q4_0", "GGUF Q4_0"],
  ["nvfp4", "NVFP4"],
  ["int4", "INT4 (bitsandbytes)"],
  ["exl2", "EXL2"],
  ["mxfp4", "MXFP4"],
  ["awq", "AWQ / GPTQ"],
  ["q3_k_m", "GGUF Q3_K_M"],
  ["q2_k", "GGUF Q2_K"],
];
const LINK_NAMES: Record<string, string> = {
  nvlink_gen5: "NVLink 5",
  nvlink_gen4: "NVLink 4",
  nvlink_gen3: "NVLink 3",
  pcie5: "PCIe 5",
  pcie4: "PCIe 4",
  infiniband_ndr: "InfiniBand NDR",
  infiniband_hdr: "InfiniBand HDR",
  ethernet400g: "400G Ethernet",
  ethernet100g: "100G Ethernet",
  ethernet10g: "10G Ethernet",
  thunderbolt5: "Thunderbolt 5",
};
const METRICS: Record<Benchmark["metric"], string> = {
  decode_tps: "Decode speed",
  prefill_tps: "Prompt processing",
  ttft_ms: "Time to first token",
};
const VENDORS: Record<string, string> = { nvidia: "NVIDIA", amd: "AMD", apple: "Apple", intel: "Intel" };
const classLabel = (k: string) => {
  if (k === "all") return "All GPUs";
  const [vendor, category] = k.split("-");
  return vendor === "apple" ? "Apple" : `${VENDORS[vendor] ?? vendor} ${category}`;
};

const gpuLabel = (key: string) => GPUS.find((g) => g.key === key)?.label ?? key;
const modelName = (slug: string) => MODELS.find((m) => m.slug === slug)?.name ?? slug;
const sourceName = (url: string) =>
  url.includes("llama.cpp/discussions/")
    ? `llama.cpp #${url.match(/discussions\/(\d+)/)?.[1]}`
    : url.includes("nvidia")
      ? "NVIDIA NIM"
      : url.includes("InferenceX")
        ? "InferenceX"
        : "Contributed run";
const pct = (x: number) => `${Math.round(x * 100)}%`;
const num = (x: number) => (x >= 100 ? Math.round(x).toLocaleString("en-US") : x.toFixed(1));

const benchmarkRows = BENCHMARKS.map((b) => ({ b, predicted: predict(b) }));
const accuracy = Object.keys(RUNTIME_NAMES).flatMap((runtime) =>
  (Object.keys(METRICS) as Benchmark["metric"][]).flatMap((metric) => {
    const rows = BENCHMARKS.filter((b) => b.runtime === runtime && b.metric === metric);
    return rows.length
      ? [{ runtime, metric, rows: rows.length, gpus: new Set(rows.map((b) => b.gpu)).size, error: typicalError(rows) }]
      : [];
  }),
);
const fittedRows = Object.entries(FITTED)
  .filter(([runtime]) => runtime !== "ollama")
  .flatMap(([runtime, classes]) => Object.entries(classes).map(([cls, v]) => ({ runtime, cls, ...v })));

// ─── Building blocks ──────────────────────────────────────────────────────────

function H2({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <h2 id={id} className="mt-16 scroll-mt-20 border-t pt-8 text-xl font-semibold text-heading first:mt-0">
      {children}
    </h2>
  );
}

function H3({ children }: { children: React.ReactNode }) {
  return <h3 className="mt-8 text-base font-semibold text-heading">{children}</h3>;
}

/** A displayed formula. */
function Eq({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-4 overflow-x-auto rounded-r-md border-l-2 border-primary bg-secondary/60 px-4 py-3 text-[15px] leading-8 text-heading">
      {children}
    </div>
  );
}

/** A variable inside a formula or the text. */
function V({ children }: { children: React.ReactNode }) {
  return <i className="whitespace-nowrap">{children}</i>;
}

function Where({ items }: { items: [React.ReactNode, React.ReactNode][] }) {
  return (
    <dl className="my-4 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-sm">
      {items.map(([term, def], i) => (
        <div key={i} className="contents">
          <dt className="text-heading">{term}</dt>
          <dd className="text-muted-foreground">{def}</dd>
        </div>
      ))}
    </dl>
  );
}

function Table({ head, rows, align }: { head: string[]; rows: React.ReactNode[][]; align?: ("l" | "r")[] }) {
  return (
    <div className="my-5 overflow-x-auto">
      <table className="w-full text-sm tabular-nums">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            {head.map((h, i) => (
              <th key={h} className={`py-2 pr-4 font-normal last:pr-0 ${align?.[i] === "r" ? "text-right" : ""}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, r) => (
            <tr key={r} className="border-b border-border/60">
              {row.map((cell, i) => (
                <td
                  key={i}
                  className={`py-2 pr-4 align-top last:pr-0 ${align?.[i] === "r" ? "text-right" : ""} ${i === 0 ? "text-heading" : ""}`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function A({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} className="text-link underline-offset-2 hover:underline">
      {children}
    </a>
  );
}

const code = (path: string) => `${REPO_URL}/blob/main/${path}`;

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function MethodologyPage() {
  return (
    <div className="flex flex-1 flex-col">
      <header className="sticky top-0 z-40 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1320px] items-center gap-4 px-4 sm:px-6">
          <Link href="/" className="mr-auto text-[15px] font-semibold text-heading">
            VRAM Calculator
          </Link>
          <Link href="/" className="text-[13px] text-muted-foreground hover:text-heading">
            Open the calculator
          </Link>
          <a
            href={REPO_URL}
            title="Source code on GitHub"
            aria-label="Source code on GitHub"
            className="text-muted-foreground hover:text-heading"
          >
            <Github className="h-4 w-4" />
          </a>
        </div>
      </header>

      <div className="mx-auto w-full max-w-[1320px] flex-1 px-4 py-10 sm:px-6 lg:grid lg:grid-cols-[200px_minmax(0,1fr)] lg:gap-16">
        <nav aria-label="Contents" className="sticky top-24 hidden self-start lg:block">
          <ol className="space-y-1.5 text-[13px]">
            {SECTIONS.map(([id, label]) => (
              <li key={id}>
                <a href={`#${id}`} className="text-muted-foreground hover:text-heading">
                  {label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <article className="max-w-[46rem] text-[15px] leading-7 [&_p]:mt-4">
          <h1 className="text-3xl font-semibold tracking-tight text-heading">How the numbers are made</h1>
          <p className="!mt-3 text-muted-foreground">
            Every estimate in the calculator comes from one plain TypeScript model,{" "}
            <A href={code("src/lib/calculator/engine.ts")}>engine.ts</A>, which runs in your browser. This page writes out its
            formulas, the constants they use and the published measurements the speed constants are fitted to.
          </p>

          <details className="mt-6 rounded-md border px-4 py-2 text-sm lg:hidden">
            <summary className="cursor-pointer py-1 text-heading">Contents</summary>
            <ol className="space-y-1.5 py-2">
              {SECTIONS.map(([id, label]) => (
                <li key={id}>
                  <a href={`#${id}`} className="text-muted-foreground">
                    {label}
                  </a>
                </li>
              ))}
            </ol>
          </details>

          <H2 id="overview">Overview</H2>
          <p>
            Memory is counted part by part: weights, KV cache, activations, runtime overhead and, when used, a speculative
            drafter. Speed follows from the hardware: generating a token means reading the weights and the KV cache from memory,
            so decode is limited by memory bandwidth; reading a prompt is a large matrix multiplication, so prompt processing is
            limited by compute. Fixed costs per step and per layer (kernel launches, scheduling) are added on top.
          </p>
          <p>
            How close a runtime gets to the hardware&apos;s peak depends on the runtime and the GPU, so those efficiencies are{" "}
            <a href="#calibration" className="text-link underline-offset-2 hover:underline">
              fitted to {BENCHMARKS.length} published measurements
            </a>
            . The results are estimates. Drivers, kernels, settings and the prompts you send all move real numbers.
          </p>

          <H2 id="model">Model shape</H2>
          <p>
            Each model is described by its Hugging Face <code className="text-link">config.json</code>: layers <V>L</V>, hidden
            size <V>h</V>, attention heads <V>n</V>
            <sub>h</sub>, KV heads <V>n</V>
            <sub>kv</sub>, head size <V>d</V>
            <sub>h</sub>, feed-forward size <V>d</V>
            <sub>ff</sub>, vocabulary <V>V</V>, and the parameter count <V>P</V> from the model&apos;s published weights. Hybrid
            models also give the number of sliding-window layers (window <V>w</V>), linear-attention or state-space layers, and
            Multi-head Latent Attention (MLA) models give the size of their compressed KV latent.
          </p>
          <p>
            For a mixture-of-experts (MoE) model with <V>E</V> experts and <V>k</V> active per token, the parameters used per
            token <V>P</V>
            <sub>active</sub> come from the model card when published, and otherwise from:
          </p>
          <Eq>
            <V>P</V>
            <sub>active</sub> = <V>P</V> / <V>E</V> × <V>k</V>
          </Eq>

          <H2 id="memory">Memory</H2>
          <H3>Weights</H3>
          <p>
            Weights take their bits per weight <V>b</V>. GGUF figures are whole-file averages, which already include the
            embeddings. Other quantized formats keep the embeddings and output head (<V>P</V>
            <sub>emb</sub> = <V>V</V>·<V>h</V>, twice when they aren&apos;t tied) in 16-bit:
          </p>
          <Eq>
            <V>W</V> = (<V>P</V> − <V>P</V>
            <sub>emb</sub>) × <V>b</V> / 8 + 2 × <V>P</V>
            <sub>emb</sub> bytes
          </Eq>
          <Table
            head={["Format", "Bits per weight"]}
            align={["l", "r"]}
            rows={FORMATS.map(([k, label]) => [label, WEIGHT_BITS[k]])}
          />

          <H3>KV cache</H3>
          <p>
            Each sequence of <V>s</V> tokens stores keys and values in every attention layer:
          </p>
          <Eq>
            <V>KV</V>(<V>s</V>) = [<V>L</V>
            <sub>full</sub> × <V>s</V> + <V>L</V>
            <sub>window</sub> × min(<V>s</V>, <V>w</V>)] × <V>c</V> × <V>β</V> + <V>L</V>
            <sub>state</sub> × <V>S</V>
          </Eq>
          <Where
            items={[
              [
                <V key="c">c</V>,
                <>
                  values per token per layer: 2 × <V>n</V>
                  <sub>kv</sub> × <V>d</V>
                  <sub>h</sub>, or the latent size for MLA (576 by default)
                </>,
              ],
              [
                <V key="b">β</V>,
                `bytes per value: ${KV_BYTES.fp16} for FP16/BF16, ${KV_BYTES.fp8} for FP8 and INT8, ${KV_BYTES.int4} for INT4`,
              ],
              [<V key="s">S</V>, "the fixed recurrent state of a linear-attention or state-space layer, whatever the length"],
            ]}
          />
          <p>
            One copy of the model serves <V>n</V>
            <sub>seq</sub> = ⌈users / copies⌉ × batch sequences at once. With prefix caching, the shared share <V>f</V> of the
            prompt is stored once: <V>KV</V>(<V>s</V>) × (<V>n</V>
            <sub>seq</sub> × (1 − <V>f</V>) + <V>f</V>). Tensor parallelism over <V>tp</V> GPUs splits the KV heads, so each GPU
            keeps ⌈<V>n</V>
            <sub>kv</sub> / <V>tp</V>⌉ / <V>n</V>
            <sub>kv</sub> of the cache (MLA latents are copied to every GPU). Pipeline parallelism over <V>pp</V> stages divides
            it by <V>pp</V>.
          </p>

          <H3>Activations and runtime overhead</H3>
          <p>
            Runtimes process prompts in chunks of <V>C</V> tokens ({RUNTIMES.vllm.chunk.toLocaleString("en-US")} for vLLM,{" "}
            {RUNTIMES.llama_cpp.chunk} for llama.cpp), and keep the logits of every running sequence:
          </p>
          <Eq>
            <V>A</V> = [<V>C</V> × (4<V>h</V> + 2<V>d</V>
            <sub>ff</sub> / <V>tp</V>) × 2 + <V>n</V>
            <sub>seq</sub> × <V>V</V> × 4 / <V>tp</V>] / <V>pp</V>
          </Eq>
          <p>
            Without fused (flash) attention, llama.cpp, Ollama and Transformers materialize the score matrix of a chunk in FP32,{" "}
            <V>C</V> × <V>s</V> × <V>n</V>
            <sub>h</sub> × 4 bytes. Each runtime also holds a fixed amount for its CUDA context, graphs and buffers, plus 0.5 GB
            when a copy spans several GPUs.
          </p>

          <H3>Speculative decoding drafters</H3>
          <p>
            A draft model adds its parameters in BF16. Learned drafters are sized from the target model, with one transformer
            layer counted as 12<V>h</V>² parameters: an EAGLE-3 head is one layer plus a 32K-token output head (12<V>h</V>² +
            32,000<V>h</V>), a DFlash drafter about five layers, and multi-token prediction (MTP) one extra layer at the
            model&apos;s own precision. N-gram and suffix lookup add nothing.
          </p>

          <H3>Usable memory and verdict</H3>
          <p>
            Per GPU, the total is <V>W</V> × (1 − <V>o</V>) / (<V>tp</V> × <V>pp</V>) plus the KV cache, activations, scores,
            overhead and drafter, where <V>o</V> is the share of weights offloaded to CPU memory (offloading the KV cache keeps
            30% of it on the GPU). It is compared with the memory the runtime lets itself use. Apple Silicon can only wire about
            75% of unified memory to the GPU, two thirds on machines with 36 GB or less.
          </p>
          <Table
            head={["Runtime", "Memory it uses", "Fixed overhead", "Prompt chunk"]}
            align={["l", "r", "r", "r"]}
            rows={Object.entries(RUNTIMES).map(([k, r]) => [
              RUNTIME_NAMES[k] ?? k,
              pct(r.reserve),
              `${r.overheadGb} GB`,
              r.chunk.toLocaleString("en-US"),
            ])}
          />
          <p>
            Below 50% of usable memory the setup is comfortable, then moderate up to 75%, high up to 90%, very high up to 100%,
            and it doesn&apos;t fit beyond. The number of users that fit is the memory left after everything but the KV cache,
            divided by the KV cache of one user, times the number of copies.
          </p>

          <H2 id="decode">Decode speed</H2>
          <p>
            Each decode step produces one token for each of the <V>b</V> sequences in flight. Its time is the larger of reading
            memory and computing, plus communication and fixed costs:
          </p>
          <Eq>
            <V>t</V>(<V>b</V>) = max(<V>W</V>
            <sub>read</sub> / (<V>tp</V> × <V>B</V> × <V>η</V>) + <V>b</V> × <V>KV</V>
            <sub>gpu</sub> / (<V>B</V> × <V>η</V>), 2<V>P</V>
            <sub>active</sub> × <V>b</V> / (0.6 × <V>F</V> × <V>tp</V>)) + <V>t</V>
            <sub>cpu</sub> + <V>t</V>
            <sub>tp</sub> + <V>t</V>
            <sub>pp</sub> + <V>t</V>
            <sub>step</sub> + <V>L</V> × <V>t</V>
            <sub>layer</sub>
          </Eq>
          <Where
            items={[
              [<V key="B">B</V>, "the GPU's memory bandwidth from its spec sheet"],
              [<V key="e">η</V>, "the share of it the runtime achieves (fitted, see Calibration)"],
              [<V key="F">F</V>, "dense FP16/BF16 tensor throughput with FP32 accumulate"],
              [
                <>
                  <V>t</V>
                  <sub>step</sub>, <V>t</V>
                  <sub>layer</sub>
                </>,
                "fixed cost per step and per layer (launches, scheduling); MoE layers add routing and small expert matmuls",
              ],
            ]}
          />
          <p>
            <V>W</V>
            <sub>read</sub> is the weights streamed per step. The input embedding table is only looked up, so it isn&apos;t
            streamed when it is separate from the output head. An MoE model always reads its shared weights, plus the experts the
            batch touches. With <V>b</V> tokens each picking <V>k</V> of <V>E</V> experts, the expected number touched is:
          </p>
          <Eq>
            <V>E</V>
            <sub>touched</sub> = <V>E</V> × (1 − (1 − <V>k</V> / <V>E</V>)
            <sup>
              <V>b</V>
            </sup>
            )
          </Eq>
          <p>
            Weights and KV cache offloaded to the CPU are read at {CPU_GBPS} GB/s (dual-channel DDR5), which adds <V>t</V>
            <sub>cpu</sub>.
          </p>

          <H3>Several GPUs</H3>
          <p>
            Tensor parallelism splits every layer across <V>tp</V> GPUs, which then all-reduce activations twice per layer. An
            all-reduce of <V>x</V> bytes over <V>n</V> GPUs costs a latency term plus the ring algorithm&apos;s traffic:
          </p>
          <Eq>
            <V>AR</V>(<V>x</V>, <V>n</V>) = <V>λ</V> × (1 + log<sub>2</sub>
            <V>n</V> / 2) + 2(<V>n</V> − 1) / <V>n</V> × <V>x</V> / <V>B</V>
            <sub>link</sub>
          </Eq>
          <p>
            So <V>t</V>
            <sub>tp</sub> = 2<V>L</V> × <V>AR</V>(2<V>bh</V>, <V>tp</V>). Pipeline parallelism passes activations between{" "}
            <V>pp</V> stages: <V>t</V>
            <sub>pp</sub> = (<V>pp</V> − 1) × <V>AR</V>(2<V>bh</V>, 2). Tensor parallelism that spans machines uses the network
            between them.
          </p>
          <Table
            head={["Link", "Bandwidth per direction", "Latency λ"]}
            align={["l", "r", "r"]}
            rows={Object.entries(LINK_NAMES).map(([k, label]) => [label, `${LINKS[k].gbps} GB/s`, `${LINKS[k].us} µs`])}
          />
          <p>
            Several copies of a model (replicas) each get <V>tp</V> × <V>pp</V> GPUs and split the users between them. Per-user
            speed is 1 / <V>t</V>(<V>b</V>); total throughput is that speed times the sequences in flight times the copies.
            Requests per second assume 512 output tokens per request.
          </p>

          <H3>Speculative decoding</H3>
          <p>
            A drafter proposes <V>k</V> tokens and the model checks them in one step over <V>k</V> + 1 tokens per sequence, so
            that step is <V>t</V>(<V>b</V>) with <V>b</V>(<V>k</V> + 1) tokens: more compute and, for MoE models, more experts
            touched. With acceptance rate <V>a</V> per drafted token, a step yields on average (1 − <V>a</V>
            <sup>
              <V>k</V>+1
            </sup>
            ) / (1 − <V>a</V>) tokens (Leviathan et al.). Drafting takes its own time: each pass reads the drafter and runs its
            layers, <V>t</V>
            <sub>draft</sub> = passes × (bytes read / (<V>tp</V> × <V>B</V> × <V>η</V>) + layers × <V>t</V>
            <sub>layer</sub>). Draft models, EAGLE-3 heads and MTP make one pass per drafted token, DFlash one pass for the whole
            block; MTP and DFlash also run the model&apos;s output head, which is large for big vocabularies. N-gram and suffix
            lookup take no GPU time. Verifying costs about 3% more per drafted token (rejection sampling, KV writes):
          </p>
          <Eq>
            <V>g</V> = [(1 − <V>a</V>
            <sup>
              <V>k</V>+1
            </sup>
            ) / (1 − <V>a</V>)] × <V>t</V>(<V>b</V>) / ([<V>t</V>(<V>b</V>, <V>b</V>(<V>k</V>+1)) + <V>t</V>
            <sub>draft</sub>] × (1 + 0.03<V>k</V>))
          </Eq>
          <p>
            Measured gains shrink with the batch size faster than this, as kernels don&apos;t overlap compute and memory
            perfectly, so the speedup used is 1 + (<V>g</V> − 1) / (1 + <V>b</V> / 8). vLLM reports an acceptance length (tokens
            per step); the calculator&apos;s acceptance rate is the <V>a</V> that gives the same length. The rates it suggests
            (from 0.4 for n-gram lookup to 0.85 for MTP) are typical values; yours depend on the model and the prompts.
          </p>

          <H2 id="prefill">Prompt processing</H2>
          <p>
            Reading an <V>s</V>-token prompt costs 2<V>P</V>
            <sub>active</sub> FLOPs per token plus attention, which grows with the square of the length:
          </p>
          <Eq>
            <V>F</V>
            <sub>attn</sub> = 2 × [<V>L</V>
            <sub>full</sub> × <V>s</V>² + <V>L</V>
            <sub>window</sub> × <V>s</V> × min(<V>s</V>, <V>w</V>)] × <V>n</V>
            <sub>h</sub> × <V>d</V>
            <sub>h</sub>
          </Eq>
          <p>
            Time to first token is the larger of that compute and re-reading the weights once per chunk, plus communication and a
            fixed cost:
          </p>
          <Eq>
            <V>TTFT</V> = max((2<V>P</V>
            <sub>active</sub>
            <V>s</V> + <V>F</V>
            <sub>attn</sub>) / (<V>F</V> × <V>tp</V> × <V>φ</V> × <V>μ</V>), ⌈<V>s</V> / <V>C</V>⌉ × <V>W</V> / (<V>tp</V> ×{" "}
            <V>B</V> × <V>η</V>)) + <V>t</V>
            <sub>tp</sub> + <V>t</V>
            <sub>0</sub>
          </Eq>
          <Where
            items={[
              [
                <V key="phi">φ</V>,
                "1.6 for FP8 weights on GPUs with FP8 tensor cores, 2.4 for NVFP4 and MXFP4 on FP4 GPUs, otherwise 1",
              ],
              [<V key="mu">μ</V>, "the share of peak compute the runtime achieves on prompts (fitted)"],
              [
                <>
                  <V>t</V>
                  <sub>0</sub>
                </>,
                "scheduling, tokenization and sampling: 15 ms unless fitted",
              ],
            ]}
          />

          <H2 id="calibration">Calibration</H2>
          <p>
            The efficiencies <V>η</V>, <V>μ</V>, <V>t</V>
            <sub>layer</sub> and <V>t</V>
            <sub>0</sub> start from typical values per runtime. Where published measurements exist, they are fitted to them by{" "}
            <A href={code("scripts/fit-speed.ts")}>scripts/fit-speed.ts</A>. The fit minimizes the squared log error, so being 2×
            too fast costs as much as being 2× too slow. It runs first over all GPUs of a runtime, then per GPU class (vendor plus
            consumer, workstation or datacenter), but only for a class with more GPUs than constants to fit. That matters: on
            prompts, llama.cpp reaches about 1.2× the catalog&apos;s FP16 figure on GeForce cards (its integer kernels beat the
            half-rate figure GeForce cards are rated at), about 0.5× on NVIDIA workstation cards and 0.15× to 0.3× on datacenter
            GPUs. Ollama runs llama.cpp and uses its constants.
          </p>
          <Table
            head={["Runtime", "GPUs", "Bandwidth η", "Per layer", "Prompt μ", "Fixed t₀"]}
            align={["l", "l", "r", "r", "r", "r"]}
            rows={fittedRows.map((r) => [
              RUNTIME_NAMES[r.runtime],
              classLabel(r.cls),
              r.bwEff != null ? pct(r.bwEff) : "",
              r.layerUs != null ? `${r.layerUs} µs` : "",
              r.prefillMfu != null ? pct(r.prefillMfu) : "",
              r.ttftMs != null ? `${r.ttftMs} ms` : "",
            ])}
          />
          <p>
            Empty cells use the runtime&apos;s All GPUs line. Runtimes without measurements here (SGLang, Transformers) keep the
            starting values:
          </p>
          <Table
            head={["Runtime", "Bandwidth η", "Per step", "Per layer", "Per MoE layer", "Prompt μ"]}
            align={["l", "r", "r", "r", "r", "r"]}
            rows={Object.entries(RUNTIMES).map(([k, r]) => [
              RUNTIME_NAMES[k] ?? k,
              pct(r.bwEff),
              `${r.stepMs} ms`,
              `${r.layerUs} µs`,
              `+${r.moeLayerUs} µs`,
              pct(r.prefillMfu),
            ])}
          />

          <H3>Accuracy</H3>
          <p>
            Typical error is exp(mean |log(predicted / measured)|) − 1, so 10% means predictions are usually within about 10%. The
            figures below are on the measurements used for the fit. On a GPU left out of the fit, which is what a GPU without
            measurements gets, decode speed is typically within 10% and llama.cpp prompt processing within 25%.
          </p>
          <Table
            head={["Runtime", "Measurement", "Benchmarks", "GPUs", "Typical error"]}
            align={["l", "l", "r", "r", "r"]}
            rows={accuracy.map((a) => [RUNTIME_NAMES[a.runtime], METRICS[a.metric], a.rows, a.gpus, pct(a.error)])}
          />
          <details className="my-5 rounded-md border px-4 py-2">
            <summary className="cursor-pointer py-1 text-sm text-heading">
              All {BENCHMARKS.length} benchmarks, measured and predicted
            </summary>
            <Table
              head={["GPU", "Model", "Runtime", "Measurement", "Measured", "Predicted", "Off by", "Source"]}
              align={["l", "l", "l", "l", "r", "r", "r", "l"]}
              rows={benchmarkRows.map(({ b, predicted }) => [
                `${b.gpus ? `${b.gpus}× ` : ""}${gpuLabel(b.gpu)}`,
                `${modelName(b.model)} ${b.quant.toUpperCase()}`,
                RUNTIME_NAMES[b.runtime],
                `${METRICS[b.metric]}, ${b.tokens.toLocaleString("en-US")} tokens${b.batch ? `, ${b.batch} at once` : ""}${
                  b.spec ? `, ${b.spec.method.toUpperCase()} ${b.spec.tokens} tokens (${b.spec.acceptance_length} per step)` : ""
                }`,
                num(b.value),
                num(predicted),
                `${predicted >= b.value ? "+" : ""}${pct(predicted / b.value - 1)}`,
                b.source.startsWith("http") ? (
                  <A key="s" href={b.source}>
                    {sourceName(b.source)}
                  </A>
                ) : (
                  <span key="s" title={b.source}>
                    {sourceName(b.source)}
                  </span>
                ),
              ])}
            />
          </details>
          <p>
            Known gaps: llama.cpp prompt processing on datacenter GPUs varies more than one constant per class can follow (the
            H100 is predicted about 2× too fast, the L40 about 2.4× too slow), very large MoE models are on the optimistic side,
            and speeds for SGLang and Transformers are not checked against measurements yet.
          </p>
          <p>
            Measurements from more hardware make the fit better. The <A href={code("scripts/benchmark.sh")}>benchmark script</A>{" "}
            measures a running vLLM server or a GGUF file with llama-bench and writes a report to paste into a{" "}
            <A href={`${REPO_URL}/issues/new?template=benchmark.yml`}>benchmark issue</A>.
          </p>

          <H2 id="finetuning">Fine-tuning</H2>
          <H3>Memory</H3>
          <p>
            Full fine-tuning keeps the weights at the training precision (2 bytes for BF16, 4 for FP32, 1 for FP8), gradients of
            at least 2 bytes, an FP32 master copy (4 bytes) in mixed precision, and Adam&apos;s two moments (8 bytes, 2 with an
            8-bit optimizer, 4 when the trainer stays in pure BF16). ZeRO stage 1 splits the optimizer states across GPUs, stage 2
            also the gradients, stage 3 also the weights.
          </p>
          <p>
            LoRA freezes the model in BF16 (QLoRA stores it in 4-bit NF4 with double quantization, 0.516 bytes per parameter, and
            keeps embeddings in 16-bit) and trains rank-<V>r</V> adapters on the attention (q, k, v, o) and MLP (gate, up, down)
            projections of every layer:
          </p>
          <Eq>
            <V>P</V>
            <sub>LoRA</sub> = <V>L</V> × <V>r</V> × [(<V>h</V> + <V>q</V>) + 2(<V>h</V> + <V>kv</V>) + (<V>q</V> + <V>h</V>) + 2(
            <V>h</V> + <V>d</V>
            <sub>ff</sub>) + (<V>d</V>
            <sub>ff</sub> + <V>h</V>)]
          </Eq>
          <p>
            with <V>q</V> = <V>n</V>
            <sub>h</sub>
            <V>d</V>
            <sub>h</sub> and <V>kv</V> = <V>n</V>
            <sub>kv</sub>
            <V>d</V>
            <sub>h</sub>. Adapters take 2 bytes per parameter, their gradients 2, and their optimizer states 12 (4 for the FP32
            copy and 8 for Adam, or 2 with an 8-bit optimizer).
          </p>
          <p>
            Activations kept for the backward pass follow Korthikanti et al.: 34 × <V>s</V> × <V>m</V> × <V>h</V> bytes per layer
            for micro-batch <V>m</V>, plus 5 × <V>n</V>
            <sub>h</sub> × <V>s</V>² × <V>m</V> without flash attention. Gradient checkpointing keeps only each layer&apos;s input
            (2<V>smh</V> bytes) and recomputes one layer at a time; Unsloth also offloads those checkpoints, keeping about 30% on
            the GPU. Logits take 6 bytes per vocabulary entry per token, 0.5 with chunked loss (Unsloth, torchtune). The trainer
            gets 95% of GPU memory plus 1 GB of overhead.
          </p>
          <H3>Speed</H3>
          <Eq>
            tokens/s = <V>N</V>
            <sub>GPU</sub> × <V>F</V> × MFU × <V>φ</V> × <V>q</V> / (<V>κ</V> × <V>P</V>
            <sub>active</sub>)
          </Eq>
          <p>
            <V>κ</V> is 6 FLOPs per parameter per token for full fine-tuning (forward 2, backward 4) and 4 for LoRA, which skips
            weight gradients, plus 2 when checkpointing recomputes the forward pass. <V>φ</V> is 1.4 for FP8 training on FP8 GPUs
            and <V>q</V> is 0.75 for QLoRA&apos;s dequantization. Training time is samples × epochs × sequence length divided by
            tokens per second.
          </p>
          <Table
            head={["Trainer", "Model FLOPs utilization (MFU)"]}
            align={["l", "r"]}
            rows={Object.entries(TRAINERS).map(([k, t]) => [TRAINER_NAMES[k] ?? k, pct(t.mfu)])}
          />

          <H2 id="encoders">Embedding and reranking</H2>
          <p>
            Encoders read each input once and keep no KV cache. Memory is the weights, activations of the batch (batch × <V>s</V>{" "}
            × (4<V>h</V> + 2<V>d</V>
            <sub>ff</sub>) × 2 bytes) and 0.75 GB of overhead, out of 92% of GPU memory. A pass over <V>n</V> tokens takes:
          </p>
          <Eq>
            <V>t</V> = max(<V>n</V> × (2<V>P</V> + 4<V>Lsh</V>) / (<V>F</V> × MFU), <V>W</V> / (0.7 × <V>B</V>)) + 2 ms
          </Eq>
          <p>with an MFU of 25% under 200M parameters, 40% under 2B and 50% above, and 85% scaling efficiency across GPUs.</p>

          <H2 id="planner">Several models</H2>
          <p>
            The Advanced view places several models on a pool of machines with a greedy plan that is simple to follow rather than
            optimal. Pinned models are placed first, then the largest. Each model takes the fewest GPUs that fit and meet its
            targets (minimum speed per user, maximum time to first token), adding tensor parallelism or copies when one GPU group
            is too slow; a model spans identical machines with pipeline parallelism only when no single machine works. Automatic
            precision starts every model at the highest quality and steps all of them down together while that places more models
            or meets more targets, then lets each step back up if nothing gets worse. Idle GPUs end up as tensor parallelism or
            extra copies for the model that gains the most speed per user. Every placement is scored with the same engine.
          </p>
          <p>
            Find hardware runs the current model on every GPU of the catalog at full, 8-bit and 4-bit precision; Find models runs
            every catalog model on your hardware.
          </p>

          <H2 id="cost">Cost and energy</H2>
          <Eq>
            power = TDP × <V>N</V>
            <sub>GPU</sub> × load
          </Eq>
          <p>
            Load is 75% for inference, 90% for training and 60% for embedding. Energy per day is power × 24 h; emissions are that
            energy times the grid&apos;s carbon intensity (380 g CO₂ per kWh unless you change it); electricity costs $0.15 per
            kWh unless you change it. Cloud rental is a rough on-demand price per GPU-hour times the GPUs, over 720 hours a month.
          </p>

          <H2 id="commands">Launch commands</H2>
          <p>
            The commands under Run it are built from the same setup: tensor and pipeline parallelism, context length, users,
            precision, KV cache format, speculative decoding, offloading and, for several copies, one process per copy with its
            own GPUs and port. They cover vLLM, SGLang, TensorRT-LLM, llama.cpp, Ollama, Transformers and Text Embeddings
            Inference, and Unsloth, TRL and torchtune for fine-tuning. Flags follow each project&apos;s documentation; check them
            against the version you install.
          </p>

          <H2 id="sources">Sources</H2>
          <H3>Data</H3>
          <ul className="mt-3 list-disc space-y-2 pl-5">
            <li>
              Models: the <A href="https://huggingface.co/docs/hub/api">Hugging Face Hub API</A>, each model&apos;s config.json
              and parameter count, rebuilt weekly by <A href={code("scripts/build-catalog.ts")}>build-catalog.ts</A>.
            </li>
            <li>
              GPUs: vendor spec sheets for memory, bandwidth, tensor throughput and power, with rough rental prices, in{" "}
              <A href={code("scripts/gpus.py")}>gpus.py</A>.
            </li>
            <li>
              Speed measurements: llama.cpp scoreboards for{" "}
              <A href="https://github.com/ggml-org/llama.cpp/discussions/15013">CUDA</A>,{" "}
              <A href="https://github.com/ggml-org/llama.cpp/discussions/15021">ROCm</A> and{" "}
              <A href="https://github.com/ggml-org/llama.cpp/discussions/4167">Apple Silicon</A>;{" "}
              <A href="https://docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html">NVIDIA NIM benchmarks</A>;{" "}
              <A href="https://github.com/SemiAnalysisAI/InferenceX">SemiAnalysis InferenceX</A>. All rows are in{" "}
              <A href={code("src/lib/data/benchmarks.json")}>benchmarks.json</A> with their links.
            </li>
            <li>The quantization guide&apos;s perplexity figures are indicative; the cost of a format varies by model.</li>
          </ul>
          <H3>Methods</H3>
          <ul className="mt-3 list-disc space-y-2 pl-5">
            <li>
              Korthikanti et al.,{" "}
              <A href="https://arxiv.org/abs/2205.05198">Reducing Activation Recomputation in Large Transformer Models</A>, 2022.
              Activation memory.
            </li>
            <li>
              Rajbhandari et al.,{" "}
              <A href="https://arxiv.org/abs/1910.02054">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</A>,
              2020.
            </li>
            <li>
              Hu et al., <A href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of Large Language Models</A>, 2021.
              Dettmers et al., <A href="https://arxiv.org/abs/2305.14314">QLoRA: Efficient Finetuning of Quantized LLMs</A>, 2023.
            </li>
            <li>
              Dao et al., <A href="https://arxiv.org/abs/2205.14135">FlashAttention</A>, 2022. Kwon et al.,{" "}
              <A href="https://arxiv.org/abs/2309.06180">Efficient Memory Management for LLM Serving with PagedAttention</A>,
              2023.
            </li>
            <li>
              DeepSeek-AI, <A href="https://arxiv.org/abs/2405.04434">DeepSeek-V2</A> (Multi-head Latent Attention), 2024, and{" "}
              <A href="https://arxiv.org/abs/2412.19437">DeepSeek-V3</A> (multi-token prediction), 2024.
            </li>
            <li>
              Leviathan, Kalman and Matias,{" "}
              <A href="https://arxiv.org/abs/2211.17192">Fast Inference from Transformers via Speculative Decoding</A>, 2023. Li
              et al., <A href="https://arxiv.org/abs/2503.01840">EAGLE-3</A>, 2025. Chen, Liang and Liu,{" "}
              <A href="https://arxiv.org/abs/2602.06036">DFlash: Block Diffusion for Flash Speculative Decoding</A>, 2026.
            </li>
            <li>
              Patarasuk and Yuan, Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations, 2009. Ring all-reduce
              traffic.
            </li>
          </ul>
          <H3>Related work on predicting LLM speed</H3>
          <ul className="mt-3 list-disc space-y-2 pl-5">
            <li>
              Patwari, Sirasao and Das,{" "}
              <A href="https://arxiv.org/abs/2508.00904">
                Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling
              </A>{" "}
              (LIFE), 2025.
            </li>
            <li>
              <A href="https://arxiv.org/abs/2607.02391">
                WattGPU: Predicting Inference Power and Latency on Unseen GPUs and LLMs
              </A>
              , 2026.
            </li>
            <li>
              Bambhaniya et al.,{" "}
              <A href="https://arxiv.org/abs/2406.01698">
                Demystifying Platform Requirements for Diverse LLM Inference Use Cases
              </A>{" "}
              (GenZ), 2024.
            </li>
            <li>
              Agrawal et al.,{" "}
              <A href="https://arxiv.org/abs/2405.05465">Vidur: A Large-Scale Simulation Framework for LLM Inference</A>, 2024.
            </li>
          </ul>

          <H2 id="cite">Cite</H2>
          <p>If the calculator helps your work, please cite it:</p>
          <p className="text-heading">
            {AUTHOR.name}. VRAM Calculator: memory and speed estimates for LLM inference and fine-tuning. 2026.{" "}
            <A href={REPO_URL}>{REPO_URL.replace("https://", "")}</A>
          </p>
          <pre className="mt-4 overflow-x-auto rounded-md bg-secondary/60 p-4 text-[13px] leading-6 text-heading">
            {`@software{brouty_vram_calculator_2026,
  author = {Brouty, Marius},
  title  = {VRAM Calculator: memory and speed estimates for LLM inference and fine-tuning},
  year   = {2026},
  url    = {${REPO_URL}}
}`}
          </pre>
        </article>
      </div>
    </div>
  );
}
