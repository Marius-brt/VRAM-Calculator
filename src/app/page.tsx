import { VramCalculatorApp } from "@/components/calculator/VramCalculatorApp";
import { GPUS, MODELS } from "@/lib/data";

export default function HomePage() {
  return <VramCalculatorApp initialModels={MODELS} initialGpus={GPUS} />;
}
