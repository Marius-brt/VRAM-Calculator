"use client";

import React, { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Trash2 } from "lucide-react";

export type SetupKind = "inference" | "finetuning" | "pool";

export interface SavedSetup {
  id: string;
  name: string;
  notes: string;
  createdAt: number;
  kind: SetupKind;
  config: any;
}

const KEY = "vram-calc:saved-setups";
const KIND_LABELS: Record<SetupKind, string> = { inference: "Inference", finetuning: "Fine-tuning", pool: "Model pool" };

// Setups live in this browser only (localStorage), so they never leave the device.
const readSetups = (): SavedSetup[] => {
  try {
    const list = JSON.parse(localStorage.getItem(KEY) || "[]");
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
};
const writeSetups = (list: SavedSetup[]) => {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
    return true;
  } catch {
    return false;
  }
};

interface SavedConfigsModalProps {
  open: boolean;
  onClose: () => void;
  current: { kind: SetupKind; config: any };
  onLoad: (setup: SavedSetup) => void;
  /** Downloads the current setup and its results as JSON. */
  onDownload?: () => void;
}

export function SavedConfigsModal({ open, onClose, current, onLoad, onDownload }: SavedConfigsModalProps) {
  const [setups, setSetups] = useState<SavedSetup[]>([]);
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setSetups(readSetups());
      setError(null);
    }
  }, [open]);

  const update = (next: SavedSetup[]) => {
    if (!writeSetups(next)) {
      setError("This browser blocked storage (private window or storage full), so nothing was saved.");
      return false;
    }
    setSetups(next);
    setError(null);
    return true;
  };

  const handleSave = () => {
    if (!name.trim()) return;
    const setup: SavedSetup = {
      id: crypto.randomUUID(),
      name: name.trim(),
      notes: notes.trim(),
      createdAt: Date.now(),
      kind: current.kind,
      config: current.config,
    };
    if (update([setup, ...setups])) {
      setName("");
      setNotes("");
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Saved setups</DialogTitle>
          <DialogDescription>Stored in this browser only. Clearing site data removes them.</DialogDescription>
        </DialogHeader>

        <form
          className="space-y-2"
          onSubmit={(e) => {
            e.preventDefault();
            handleSave();
          }}
        >
          <div className="grid gap-2 sm:grid-cols-2">
            <Input
              placeholder="Name, e.g. 70B on 2× H100"
              aria-label="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <Input placeholder="Notes (optional)" aria-label="Notes" value={notes} onChange={(e) => setNotes(e.target.value)} />
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-muted-foreground">
              Saves the current {KIND_LABELS[current.kind].toLowerCase()} setup.
            </span>
            <div className="flex items-center gap-2">
              {onDownload && (
                <Button type="button" variant="ghost" size="sm" onClick={onDownload}>
                  Download JSON
                </Button>
              )}
              <Button type="submit" size="sm" disabled={!name.trim()}>
                Save current setup
              </Button>
            </div>
          </div>
          {error && <p className="text-[13px] text-danger">{error}</p>}
        </form>

        <div className="border-t pt-1">
          {setups.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">Nothing saved yet.</p>
          ) : (
            <ul>
              {setups.map((s) => (
                <li
                  key={s.id}
                  className="flex items-center justify-between gap-3 border-b border-border/60 py-3 text-[13px] last:border-b-0"
                >
                  <div className="min-w-0">
                    <div className="flex items-baseline gap-2">
                      <span className="truncate font-medium text-heading">{s.name}</span>
                      <span className="shrink-0 text-xs text-muted-foreground">{KIND_LABELS[s.kind] ?? s.kind}</span>
                    </div>
                    {s.notes && <p className="mt-0.5 truncate text-muted-foreground">{s.notes}</p>}
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      Saved {new Date(s.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        onLoad(s);
                        onClose();
                      }}
                    >
                      Load
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => update(setups.filter((x) => x.id !== s.id))}
                      aria-label={`Delete ${s.name}`}
                      title="Delete"
                      className="h-8 w-8 hover:text-danger"
                    >
                      <Trash2 />
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
