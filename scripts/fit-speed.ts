/**
 * Fits the speed constants of each runtime (engine.ts RUNTIMES) to the measured benchmarks in
 * src/lib/data/benchmarks.json: first for every GPU, then per GPU class (vendor + category) where there is data.
 * Minimizes the squared log error, so a 2× overestimate costs as much as a 2× underestimate.
 * Run: bun scripts/fit-speed.ts         prints the error of the current fit, row by row
 *      bun scripts/fit-speed.ts --write  refits and writes src/lib/data/speed-fit.json
 */
import { FITTED, gpuClass, RUNTIMES } from "../src/lib/calculator/engine";
import { BENCHMARKS, predict, typicalError, type Benchmark } from "../src/lib/calculator/benchmarks";
import { GPUS } from "../src/lib/data";

const gpuOf = (b: Benchmark) => GPUS.find((g) => g.key === b.gpu)!;
const loss = (rows: Benchmark[]) => rows.reduce((s, b) => s + Math.log(predict(b) / b.value) ** 2, 0);

// Which constants each kind of measurement pins down, and the range they may take. prefillMfu is relative to the
// catalog's FP16 figure, which for GeForce cards is the half-rate FP32-accumulate number, so it can exceed 1.
const RANGES = { bwEff: [0.2, 1], layerUs: [0, 200], prefillMfu: [0.03, 2], ttftMs: [0, 50] } as const;
const PARAMS: Record<Benchmark["metric"], (keyof typeof RANGES)[]> = {
  decode_tps: ["bwEff", "layerUs"],
  prefill_tps: ["prefillMfu"],
  ttft_ms: ["prefillMfu", "ttftMs"],
};

/** Pattern search: moves along each constant plus a few random directions, so coupled constants can move together. */
function fit(group: Record<string, number>, rows: Benchmark[], base: Record<string, number>) {
  const names = [...new Set(rows.flatMap((b) => PARAMS[b.metric]))];
  for (const k of names) group[k] = base[k] ?? 15;
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  let best = loss(rows);
  for (let step = 0.25; step > 1e-4; step /= 2)
    for (let improved = true; improved;) {
      improved = false;
      const dirs = [
        ...names.flatMap((k) => [{ [k]: 1 }, { [k]: -1 }]),
        ...Array.from({ length: 8 }, () => Object.fromEntries(names.map((k) => [k, rand()]))),
      ];
      for (const dir of dirs) {
        const old = { ...group };
        for (const [k, d] of Object.entries(dir)) {
          const [lo, hi] = RANGES[k as keyof typeof RANGES];
          group[k] = Math.min(hi, Math.max(lo, group[k] + d * step * (hi - lo)));
        }
        const l = loss(rows);
        if (l < best - 1e-12) ((best = l), (improved = true));
        else Object.assign(group, old);
      }
    }
  for (const k of names) group[k] = Number(group[k].toPrecision(3));
}

/** Refits FITTED in place from these rows: each runtime for all GPUs, then per GPU class where the data allows. */
export function fitAll(rows: Benchmark[]) {
  for (const k of Object.keys(FITTED)) delete FITTED[k];
  for (const runtime of new Set(rows.map((b) => b.runtime))) {
    const own = rows.filter((b) => b.runtime === runtime);
    const fitted: Record<string, Record<string, number>> = { all: {} };
    FITTED[runtime] = fitted;
    fit(fitted.all, own, RUNTIMES[runtime] as unknown as Record<string, number>);
    const classes = new Set(own.map((b) => gpuClass(gpuOf(b))));
    if (classes.size < 2) continue;
    for (const cls of classes) {
      // A class gets its own constants only with more GPUs than constants; otherwise they would just interpolate.
      const sub = own.filter((b) => gpuClass(gpuOf(b)) === cls);
      const usable = sub.filter(
        (b) => new Set(sub.filter((x) => x.metric === b.metric).map((x) => x.gpu)).size > PARAMS[b.metric].length,
      );
      if (!usable.length) continue;
      fitted[cls] = {};
      fit(fitted[cls], usable, { ...RUNTIMES[runtime], ...fitted.all } as Record<string, number>);
    }
  }
  // Ollama runs llama.cpp.
  if (FITTED.llama_cpp) FITTED.ollama = FITTED.llama_cpp;
}

if (import.meta.main) {
  const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
  if (process.argv.includes("--cv")) {
    // Leave one GPU out: the error on GPUs the fit has not seen, which is what a new catalog GPU gets.
    const held: Benchmark[] = [];
    const errors = new Map<Benchmark, number>();
    for (const gpu of new Set(BENCHMARKS.map((b) => b.gpu))) {
      fitAll(BENCHMARKS.filter((b) => b.gpu !== gpu));
      for (const b of BENCHMARKS.filter((b) => b.gpu === gpu))
        (held.push(b), errors.set(b, Math.abs(Math.log(predict(b) / b.value))));
    }
    for (const runtime of new Set(held.map((b) => b.runtime)))
      for (const metric of ["decode_tps", "prefill_tps", "ttft_ms"]) {
        const e = held.filter((b) => b.runtime === runtime && b.metric === metric).map((b) => errors.get(b)!);
        if (e.length)
          console.log(runtime, metric, "held-out typical error", pct(Math.exp(e.reduce((s, x) => s + x, 0) / e.length) - 1));
      }
    process.exit(0);
  }
  if (process.argv.includes("--write")) {
    fitAll(BENCHMARKS);
    await Bun.write(new URL("../src/lib/data/speed-fit.json", import.meta.url), JSON.stringify(FITTED, null, 1) + "\n");
  }
  for (const b of BENCHMARKS) {
    const p = predict(b);
    console.log(
      [b.runtime, b.gpu, b.quant, b.metric, b.value.toFixed(1), p.toFixed(1), pct(p / b.value - 1)]
        .map((c) => String(c).padEnd(14))
        .join(""),
    );
  }
  for (const runtime of new Set(BENCHMARKS.map((b) => b.runtime)))
    for (const metric of ["decode_tps", "prefill_tps", "ttft_ms"]) {
      const rows = BENCHMARKS.filter((b) => b.runtime === runtime && b.metric === metric);
      if (rows.length) console.log(runtime, metric, rows.length, "rows, typical error", pct(typicalError(rows)));
    }
}
