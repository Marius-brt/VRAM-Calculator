import type { GpuSpec, ModelVariant } from "../calculator/types";
import gpuList from "./gpus.json";
import modelList from "./models.json";

/** GPUs from public spec sheets (scripts/gpus.py), each vendor's newest first. */
export const GPUS = gpuList as GpuSpec[];
/** Models built from Hugging Face configs (scripts/build-catalog.ts). */
export const MODELS = modelList as ModelVariant[];
