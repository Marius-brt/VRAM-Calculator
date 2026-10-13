"use client";

import React, { useState } from "react";
import { ModelVariant } from "@/lib/calculator/types";
import { importFromHuggingFace } from "@/lib/calculator/hf";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Search, Download, Loader2, Check, AlertCircle } from "lucide-react";

interface HuggingFaceImportModalProps {
  open: boolean;
  onClose: () => void;
  onModelImported: (model: ModelVariant) => void;
}

export function HuggingFaceImportModal({ open, onClose, onModelImported }: HuggingFaceImportModalProps) {
  const [query, setQuery] = useState("");
  const [customRepoId, setCustomRepoId] = useState("");
  const [hfToken, setHfToken] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Hugging Face allows these requests straight from the browser, so the token never touches our server.
  const authHeaders = (): Record<string, string> => (hfToken.trim() ? { Authorization: `Bearer ${hfToken.trim()}` } : {});

  const handleSearch = async () => {
    if (!query.trim()) return;
    setIsSearching(true);
    setError(null);
    try {
      const res = await fetch(
        `https://huggingface.co/api/models?search=${encodeURIComponent(query.trim())}&limit=20&sort=downloads&direction=-1`,
        { headers: authHeaders() },
      );
      if (!res.ok) throw new Error(`Hugging Face search failed (${res.status}).`);
      const data: any[] = await res.json();
      setSearchResults(data.map((m) => ({ id: m.id, downloads: m.downloads ?? 0, likes: m.likes ?? 0 })));
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsSearching(false);
    }
  };

  const handleImport = async (repoId: string) => {
    setIsImporting(true);
    setError(null);
    setSuccessMsg(null);
    try {
      const id = repoId.trim().replace(/^https?:\/\/huggingface\.co\//, "");
      if (!/^[\w.-]+\/[\w.-]+$/.test(id)) throw new Error("Use a model id like Qwen/Qwen3.5-122B-A10B.");
      const spec = await importFromHuggingFace(id, hfToken.trim() || undefined);
      const model = { ...spec, slug: `hf-${spec.slug}`, is_preset: 0 };
      setSuccessMsg(`Imported ${model.name}.`);
      onModelImported(model);
      setTimeout(() => {
        onClose();
        setSuccessMsg(null);
      }, 1200);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsImporting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Import from Hugging Face</DialogTitle>
          <DialogDescription>
            Reads the model&#39;s <code className="text-link">config.json</code> for layers, hidden size, KV heads and attention
            type, then adds it to your model list.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <div className="p-2.5 rounded-md bg-danger/10 text-danger text-[13px] flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {successMsg && (
          <div className="p-2.5 rounded-md bg-success/10 text-success text-[13px] flex items-center gap-2">
            <Check className="h-4 w-4 shrink-0" />
            <span>{successMsg}</span>
          </div>
        )}

        {/* Quick Repo ID Direct Import */}
        <div className="space-y-1.5">
          <Label className="text-[13px] font-normal text-muted-foreground">Model ID</Label>
          <div className="flex gap-2">
            <Input
              placeholder="meta-llama/Llama-3.1-8B-Instruct"
              value={customRepoId}
              onChange={(e) => setCustomRepoId(e.target.value)}
              className="h-9"
            />
            <Button
              size="sm"
              onClick={() => handleImport(customRepoId)}
              disabled={!customRepoId.trim() || isImporting}
              className="shrink-0 gap-1"
            >
              {isImporting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
              Import
            </Button>
          </div>
        </div>

        {/* Live Search */}
        <div className="space-y-2">
          <Label className="text-[13px] font-normal text-muted-foreground">Or search the Hub</Label>
          <div className="flex gap-2">
            <Input
              placeholder="DeepSeek-V3, Qwen2.5, Mistral"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="h-9 text-sm"
            />
            <Button
              size="sm"
              variant="secondary"
              onClick={handleSearch}
              disabled={!query.trim() || isSearching}
              className="shrink-0 gap-1"
            >
              {isSearching ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />}
              Search
            </Button>
          </div>

          {/* Results List */}
          <div className="max-h-60 overflow-y-auto space-y-1.5 pr-1 mt-2">
            {searchResults.map((item) => (
              <div
                key={item.id}
                className="px-2 py-2 rounded-md hover:bg-accent flex items-center justify-between text-[13px] transition-colors"
              >
                <div className="min-w-0 pr-2">
                  <div className="flex items-center gap-1.5">
                    <span className="truncate text-heading">{item.id}</span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground mt-0.5">
                    <span>{item.downloads.toLocaleString()} downloads</span>
                    <span>{item.likes.toLocaleString()} likes</span>
                  </div>
                </div>
                <Button
                  size="xs"
                  variant="outline"
                  onClick={() => handleImport(item.id)}
                  disabled={isImporting}
                  className="shrink-0 text-[11px] h-6 px-2 gap-1"
                >
                  <Download className="h-2.5 w-2.5" /> Import
                </Button>
              </div>
            ))}
          </div>
        </div>

        {/* Optional HF Token */}
        <div className="pt-2 border-t">
          <Label className="text-[13px] font-normal text-muted-foreground block mb-1.5">
            Access token, only needed for gated models such as Llama
          </Label>
          <Input
            type="password"
            placeholder="hf_..."
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
            className="h-9 text-sm"
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
