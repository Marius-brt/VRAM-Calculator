"use client";

import React, { useEffect, useState } from "react";
import { ModelVariant } from "@/lib/calculator/types";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, Select } from "./controls";

interface CustomizeModelModalProps {
  open: boolean;
  onClose: () => void;
  initialModel: ModelVariant;
  onSaveModel: (model: ModelVariant) => void;
}

export function CustomizeModelModal({ open, onClose, initialModel, onSaveModel }: CustomizeModelModalProps) {
  const [model, setModel] = useState<ModelVariant>({ ...initialModel });

  // Start from whichever model is selected each time the dialog opens.
  useEffect(() => {
    // The catalog stores attention types in lowercase ("gqa"); the engine and this form expect "GQA".
    const attention = initialModel.attention_structure?.toUpperCase() as ModelVariant["attention_structure"];
    if (open) setModel({ ...initialModel, attention_structure: attention });
  }, [open, initialModel]);

  const set = (patch: Partial<ModelVariant>) => setModel((m) => ({ ...m, ...patch }));
  const int = (v: string, fallback: number) => parseInt(v) || fallback;

  const handleSave = () => {
    onSaveModel(model);
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>Edit model specs</DialogTitle>
          <DialogDescription>
            Changes apply to a copy of {initialModel.name}, saved as a custom model in this session.
          </DialogDescription>
        </DialogHeader>

        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            handleSave();
          }}
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Name">
              <Input value={model.name || ""} onChange={(e) => set({ name: e.target.value })} />
            </Field>
            <Field label="Parameters (billions)">
              <Input
                type="number"
                step="0.1"
                value={model.num_of_params || 7}
                onChange={(e) => set({ num_of_params: parseFloat(e.target.value) || 7 })}
              />
            </Field>
            <Field label="Architecture">
              <Select value={model.architecture || "dense"} onChange={(e) => set({ architecture: e.target.value as any })}>
                <option value="dense">Dense</option>
                <option value="moe">Mixture of experts</option>
              </Select>
            </Field>
            <Field label="Attention">
              <Select
                value={model.attention_structure || "MHA"}
                onChange={(e) => set({ attention_structure: e.target.value as any })}
              >
                <option value="MHA">Multi-head (MHA)</option>
                <option value="GQA">Grouped-query (GQA)</option>
                <option value="MLA">Multi-head latent (MLA)</option>
              </Select>
            </Field>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Field label="Context length">
              <Input
                type="number"
                value={model.context_length || 4096}
                onChange={(e) => set({ context_length: int(e.target.value, 4096) })}
              />
            </Field>
            <Field label="Layers">
              <Input
                type="number"
                value={model.num_of_layers || 32}
                onChange={(e) => set({ num_of_layers: int(e.target.value, 32) })}
              />
            </Field>
            <Field label="Hidden size">
              <Input
                type="number"
                value={model.hidden_size || 4096}
                onChange={(e) => set({ hidden_size: int(e.target.value, 4096) })}
              />
            </Field>
            <Field label="Attention heads">
              <Input
                type="number"
                value={model.num_attention_heads || 32}
                onChange={(e) => set({ num_attention_heads: int(e.target.value, 32) })}
              />
            </Field>
            <Field label="KV heads">
              <Input
                type="number"
                value={model.num_key_value_heads || 8}
                onChange={(e) => set({ num_key_value_heads: int(e.target.value, 8) })}
              />
            </Field>
            <Field label="Vocabulary size">
              <Input
                type="number"
                value={model.vocab_size || 32000}
                onChange={(e) => set({ vocab_size: int(e.target.value, 32000) })}
              />
            </Field>
            <Field label="Full-attention layers">
              <Input
                type="number"
                value={model.kv_layers ?? model.num_of_layers ?? 32}
                onChange={(e) => set({ kv_layers: int(e.target.value, model.num_of_layers || 32) })}
              />
            </Field>
            <Field label="FFN size">
              <Input
                type="number"
                value={model.ffn_intermediate_size || 14336}
                onChange={(e) => set({ ffn_intermediate_size: int(e.target.value, 14336) })}
              />
            </Field>
            {model.architecture === "moe" && (
              <>
                <Field label="Experts">
                  <Input
                    type="number"
                    value={model.num_experts || 8}
                    onChange={(e) => set({ num_experts: int(e.target.value, 8) })}
                  />
                </Field>
                <Field label="Active experts">
                  <Input
                    type="number"
                    value={model.num_active_experts || 2}
                    onChange={(e) => set({ num_active_experts: int(e.target.value, 2) })}
                  />
                </Field>
                <Field label="Active parameters (B)">
                  <Input
                    type="number"
                    step="0.1"
                    value={model.active_params ?? ""}
                    placeholder="Estimated"
                    onChange={(e) => set({ active_params: parseFloat(e.target.value) || undefined })}
                  />
                </Field>
              </>
            )}
            {model.attention_structure === "MLA" && (
              <Field label="KV latent size">
                <Input
                  type="number"
                  value={model.kv_latent_dim || 576}
                  onChange={(e) => set({ kv_latent_dim: int(e.target.value, 576) })}
                />
              </Field>
            )}
          </div>

          <div className="flex justify-end gap-2 border-t pt-4">
            <Button type="button" variant="ghost" size="sm" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" size="sm">
              Use these specs
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
