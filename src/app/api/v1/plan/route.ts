import { NextRequest, NextResponse } from "next/server";
import { planPool, PoolConfig } from "@/lib/calculator/planner";
import { GPUS } from "@/lib/data";

// Bounds keep one request from fanning out into a very long search.
const MAX_NODES = 32;
const MAX_MODELS = 16;

export async function POST(req: NextRequest) {
  const body = (await req.json().catch(() => null)) as PoolConfig | null;
  if (!Array.isArray(body?.nodes) || !Array.isArray(body?.models)) {
    return NextResponse.json({ error: "Expected { nodes: [], models: [], network }" }, { status: 400 });
  }
  if (body.nodes.length > MAX_NODES || body.models.length > MAX_MODELS) {
    return NextResponse.json({ error: `Up to ${MAX_NODES} nodes and ${MAX_MODELS} models` }, { status: 400 });
  }
  const config: PoolConfig = {
    network: String(body.network || "ethernet100g"),
    nodes: body.nodes.map((n) => ({ ...n, count: Math.min(16, Math.max(1, Math.round(Number(n.count) || 1))) })),
    models: body.models
      .filter((m) => m?.model)
      .map((m) => ({
        ...m,
        contextLength: Math.max(64, Math.round(Number(m.contextLength) || 4096)),
        users: Math.min(1024, Math.max(1, Math.round(Number(m.users) || 1))),
        quant: m.quant || "auto",
      })),
  };
  return NextResponse.json(planPool(config, GPUS));
}
