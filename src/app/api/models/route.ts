import { NextRequest, NextResponse } from "next/server";
import { MODELS } from "@/lib/data";

/** The model catalog; ?q= filters by name, provider or family. */
export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get("q")?.toLowerCase();
  const models = q ? MODELS.filter((m) => `${m.name} ${m.provider} ${m.family} ${m.slug}`.toLowerCase().includes(q)) : MODELS;
  return NextResponse.json(models);
}
