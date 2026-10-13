"use client";

import React, { useEffect, useId, useMemo, useRef, useState } from "react";
import * as Popover from "@radix-ui/react-popover";
import NumberFlow from "@number-flow/react";
import { Check, ChevronDown, Minus, Plus, Search } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

/** Animated number (NumberFlow); shows an em dash when the engine has no estimate. */
export function Num({ value, digits = 1 }: { value: number | null | undefined; digits?: number }) {
  if (value == null || !Number.isFinite(value)) return <>—</>;
  return <NumberFlow value={value} format={{ maximumFractionDigits: digits }} />;
}

/** A titled block inside a panel; consecutive sections are separated by a rule. */
export function Section({
  title,
  action,
  children,
  className,
}: {
  title?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("space-y-4 border-t px-5 py-5 first:border-t-0", className)}>
      {(title || action) && (
        <div className="flex min-h-7 flex-wrap items-center justify-between gap-x-3 gap-y-2">
          <h2 className="text-sm font-semibold text-heading">{title}</h2>
          {action && <div className="flex items-center gap-1">{action}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

/** Label + single control (select or input). */
export function Field({
  label,
  hint,
  children,
  className,
  group,
}: {
  label: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  /** Use for composite controls (e.g. Stepper) so clicks on the label don't hit the first button. */
  group?: boolean;
}) {
  const Tag = group ? "div" : "label";
  return (
    <Tag className={cn("block min-w-0 space-y-1.5", className)}>
      <span className="block text-[13px] text-muted-foreground">{label}</span>
      {children}
      {hint && <span className="block text-xs text-muted-foreground">{hint}</span>}
    </Tag>
  );
}

export function Select({ className, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="relative">
      <select
        className={cn(
          "h-9 w-full appearance-none truncate rounded-md border border-input bg-secondary pl-3 pr-8 text-sm text-heading transition-colors hover:border-muted-foreground/60",
          className,
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  className,
  label,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: React.ReactNode }[];
  className?: string;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className={cn("flex rounded-md border bg-secondary p-0.5", className)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "flex-1 whitespace-nowrap rounded-[5px] px-3 py-1 text-[13px] font-medium transition-colors",
            value === o.value ? "bg-accent text-heading shadow-sm" : "text-muted-foreground hover:text-heading",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

const nearestIndex = (steps: number[], v: number) =>
  steps.reduce((best, s, i) => (Math.abs(Math.log(s / v)) < Math.abs(Math.log(steps[best] / v)) ? i : best), 0);

/** 8192 → "8K", 131072 → "128K", 1048576 → "1M"; handles both binary and decimal sizes. */
export const formatTokens = (n: number) => {
  const [div, unit] =
    n >= 1e6 ? [n % 1048576 === 0 ? 1048576 : 1e6, "M"] : n >= 1000 ? [n % 1024 === 0 ? 1024 : 1000, "K"] : [1, ""];
  return `${Number((n / div).toFixed(1))}${unit}`;
};

/** Slider that snaps to a fixed list of values, with evenly spaced tick labels under each stop. */
export function StepSlider({
  label,
  steps,
  value,
  onChange,
  unit,
  format = (n) => `${n}`,
  note,
}: {
  label: string;
  steps: number[];
  value: number;
  onChange: (value: number) => void;
  unit?: string;
  format?: (n: number) => string;
  note?: React.ReactNode;
}) {
  const last = steps.length - 1;
  // Long ranges (e.g. 512 to 10M tokens) label every other stop, counted from the end.
  const every = Math.ceil(steps.length / 9);
  return (
    <div className="space-y-2.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[13px] text-muted-foreground">{label}</span>
        <span className="text-sm tabular-nums text-heading">
          {value.toLocaleString()}
          {unit && <span className="ml-1 text-muted-foreground">{unit}</span>}
        </span>
      </div>
      <Slider
        aria-label={label}
        min={0}
        max={last}
        step={1}
        value={[nearestIndex(steps, value)]}
        onValueChange={([i]) => onChange(steps[i])}
      />
      <div className="relative h-3.5 text-[11px] leading-none text-muted-foreground tabular-nums" aria-hidden>
        {steps.map(
          (s, i) =>
            (last - i) % every === 0 && (
              <span
                key={s}
                className="absolute -translate-x-1/2"
                // Thumb centre travels from 8px to (100% - 8px).
                style={{ left: `calc(8px + (100% - 16px) * ${i / last})` }}
              >
                {format(s)}
              </span>
            ),
        )}
      </div>
      {note}
    </div>
  );
}

export function Stepper({
  value,
  onChange,
  min = 1,
  max = 64,
  label,
}: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  label: string;
}) {
  const set = (n: number) => onChange(Math.min(max, Math.max(min, Math.round(n) || min)));
  const btn =
    "grid h-full w-8 place-items-center text-muted-foreground transition-colors hover:text-heading disabled:pointer-events-none disabled:opacity-40";
  return (
    <div className="inline-flex h-9 items-center rounded-md border border-input bg-secondary">
      <button
        type="button"
        aria-label={`Decrease ${label}`}
        onClick={() => set(value - 1)}
        disabled={value <= min}
        className={btn}
      >
        <Minus className="h-3.5 w-3.5" />
      </button>
      <input
        aria-label={label}
        type="number"
        inputMode="numeric"
        min={min}
        max={max}
        value={value}
        onChange={(e) => set(Number(e.target.value))}
        className="h-full w-11 border-x border-input bg-transparent text-center text-sm tabular-nums text-heading [appearance:textfield] focus-visible:outline-none [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      />
      <button
        type="button"
        aria-label={`Increase ${label}`}
        onClick={() => set(value + 1)}
        disabled={value >= max}
        className={btn}
      >
        <Plus className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

export function ToggleRow({
  title,
  description,
  checked,
  onCheckedChange,
  children,
}: {
  title: string;
  description: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  children?: React.ReactNode;
}) {
  const id = useId();
  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-4">
        <label htmlFor={id} className="cursor-pointer">
          <span className="block text-[13px] font-medium text-heading">{title}</span>
          <span className="mt-0.5 block text-xs text-muted-foreground">{description}</span>
        </label>
        <Switch id={id} checked={checked} onCheckedChange={onCheckedChange} className="mt-0.5" />
      </div>
      {checked && children}
    </div>
  );
}

/** Small key/value grid used under the model and GPU pickers. */
export function Specs({ items, className = "grid-cols-3" }: { items: [string, React.ReactNode][]; className?: string }) {
  return (
    <dl className={cn("grid gap-3", className)}>
      {items.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className="text-xs text-muted-foreground">{k}</dt>
          <dd className="mt-0.5 truncate text-[13px] tabular-nums text-heading">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Every whitespace-separated term must appear somewhere in the haystack. */
export const matchesQuery = (haystack: string, query: string) => {
  const h = haystack.toLowerCase();
  return query
    .toLowerCase()
    .split(/\s+/)
    .every((t) => h.includes(t));
};

export interface PickerGroup<T> {
  label: string;
  items: T[];
}

/**
 * Searchable, grouped dropdown. `groups` must be memoized by the caller,
 * otherwise the keyboard highlight resets on every render.
 */
export function Picker<T>({
  groups,
  getKey,
  selectedKey,
  onSelect,
  renderItem,
  trigger,
  query,
  onQueryChange,
  placeholder,
  toolbar,
  emptyState,
}: {
  groups: PickerGroup<T>[];
  getKey: (item: T) => string;
  selectedKey?: string;
  onSelect: (item: T) => void;
  renderItem: (item: T) => React.ReactNode;
  trigger: React.ReactNode;
  query: string;
  onQueryChange: (q: string) => void;
  placeholder: string;
  toolbar?: React.ReactNode;
  emptyState?: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const listId = useId();
  const flat = useMemo(() => groups.flatMap((g) => g.items), [groups]);

  // Open on the current selection; after typing, highlight the first match.
  useEffect(() => {
    const selected = flat.findIndex((it) => getKey(it) === selectedKey);
    setActive(query || selected < 0 ? 0 : selected);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flat, open]);

  useEffect(() => {
    if (!open) return;
    const id = requestAnimationFrame(() =>
      listRef.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" }),
    );
    return () => cancelAnimationFrame(id);
  }, [active, open]);

  const setOpenAndReset = (o: boolean) => {
    setOpen(o);
    if (!o) onQueryChange("");
  };
  const choose = (item: T) => {
    onSelect(item);
    setOpenAndReset(false);
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(flat.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter" && flat[active]) {
      e.preventDefault();
      choose(flat[active]);
    }
  };

  let index = -1;
  return (
    <Popover.Root open={open} onOpenChange={setOpenAndReset}>
      <Popover.Trigger asChild>
        <button
          type="button"
          className="flex w-full items-center gap-3 rounded-md border border-input bg-secondary px-3 py-2.5 text-left transition-colors hover:border-muted-foreground/60 data-[state=open]:border-primary"
        >
          <span className="min-w-0 flex-1">{trigger}</span>
          <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="start"
          sideOffset={6}
          collisionPadding={12}
          onOpenAutoFocus={(e) => {
            e.preventDefault();
            inputRef.current?.focus();
          }}
          className="z-50 flex max-h-[min(24rem,var(--radix-popover-content-available-height,24rem))] w-[var(--radix-popover-trigger-width)] min-w-[17rem] flex-col overflow-hidden rounded-lg border bg-popover text-popover-foreground shadow-2xl shadow-black/40"
        >
          <div className="space-y-2 border-b p-2">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                ref={inputRef}
                role="combobox"
                aria-expanded
                aria-controls={listId}
                aria-activedescendant={flat.length ? `${listId}-${active}` : undefined}
                value={query}
                onChange={(e) => onQueryChange(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={placeholder}
                className="h-9 w-full rounded-md bg-secondary pl-8 pr-3 text-sm text-heading placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0"
              />
            </div>
            {toolbar}
          </div>
          <div ref={listRef} id={listId} role="listbox" className="flex-1 overflow-y-auto overscroll-contain px-1 pb-1">
            {flat.length === 0 ? (
              <div className="px-3 py-8 text-center text-sm text-muted-foreground">{emptyState ?? "No matches."}</div>
            ) : (
              groups.map(
                (g) =>
                  g.items.length > 0 && (
                    <div key={g.label} role="group" aria-label={g.label}>
                      <div className="sticky top-0 z-10 flex justify-between bg-popover px-2 pb-1 pt-2.5 text-xs font-medium text-muted-foreground">
                        <span>{g.label}</span>
                        <span className="tabular-nums">{g.items.length}</span>
                      </div>
                      {g.items.map((item) => {
                        const i = ++index;
                        const selected = getKey(item) === selectedKey;
                        return (
                          <div
                            key={getKey(item)}
                            id={`${listId}-${i}`}
                            data-index={i}
                            role="option"
                            aria-selected={selected}
                            onClick={() => choose(item)}
                            onMouseMove={() => i !== active && setActive(i)}
                            className={cn(
                              "flex cursor-pointer scroll-mt-9 items-center gap-2 rounded-md px-2 py-1.5 text-sm",
                              i === active && "bg-accent",
                            )}
                          >
                            <Check className={cn("h-3.5 w-3.5 shrink-0 text-link", !selected && "invisible")} />
                            <div className="min-w-0 flex-1">{renderItem(item)}</div>
                          </div>
                        );
                      })}
                    </div>
                  ),
              )
            )}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

/** Filter chips inside a Picker toolbar. They don't take focus, so arrow keys keep working in the search box. */
export function Chips<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          tabIndex={-1}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => onChange(o.value)}
          aria-pressed={value === o.value}
          className={cn(
            "rounded-full border px-2.5 py-0.5 text-xs transition-colors",
            value === o.value
              ? "border-primary bg-primary/20 text-heading"
              : "border-transparent text-muted-foreground hover:bg-accent hover:text-heading",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
