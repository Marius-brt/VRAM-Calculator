"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Bookmark, Check, Link2, Moon, Sun } from "lucide-react";
import { Segmented } from "./controls";
import type { View } from "./VramCalculatorApp";

interface CalculatorHeaderProps {
  onOpenSavedConfigs: () => void;
  isDark: boolean;
  setIsDark: (dark: boolean) => void;
  view: View;
  setView: (view: View) => void;
}

function HeaderButton({ icon: Icon, label, onClick }: { icon: typeof Bookmark; label: string; onClick: () => void }) {
  return (
    <Button variant="ghost" size="sm" onClick={onClick} title={label} className="gap-1.5 px-2 text-[13px] md:px-2.5">
      <Icon />
      <span className="sr-only md:not-sr-only">{label}</span>
    </Button>
  );
}

export function CalculatorHeader({ onOpenSavedConfigs, isDark, setIsDark, view, setView }: CalculatorHeaderProps) {
  const [copied, setCopied] = useState(false);

  // The page keeps the URL in sync with the current setup, so the address is the share link.
  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      window.prompt("Copy this link", window.location.href);
    }
  };

  return (
    <header className="sticky top-0 z-40 border-b bg-background/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1320px] items-center gap-0.5 px-4 sm:gap-2 sm:px-6">
        <h1 className="hidden min-w-0 truncate pr-2 text-[15px] font-semibold text-heading sm:block">VRAM Calculator</h1>
        <Segmented
          label="View"
          value={view}
          onChange={setView}
          className="mr-auto"
          options={[
            { value: "simple", label: "Simple" },
            { value: "advanced", label: "Advanced" },
          ]}
        />
        <HeaderButton icon={Bookmark} label="Saved setups" onClick={onOpenSavedConfigs} />
        <HeaderButton icon={copied ? Check : Link2} label={copied ? "Link copied" : "Copy link"} onClick={handleShare} />
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={() => setIsDark(!isDark)}
          title={isDark ? "Switch to light theme" : "Switch to dark theme"}
          aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
        >
          {isDark ? <Sun /> : <Moon />}
        </Button>
      </div>
    </header>
  );
}
