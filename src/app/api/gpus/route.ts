import { NextResponse } from "next/server";
import { GPUS } from "@/lib/data";

export async function GET() {
  return NextResponse.json(GPUS);
}
