import type { Metadata } from "next";
import "./globals.css";
import { SiteFooter } from "@/components/SiteFooter";

export const metadata: Metadata = {
  title: "VRAM Calculator",
  description:
    "Check whether an LLM fits on your GPU and how fast it runs, on NVIDIA, AMD, Apple and Intel hardware, multi-GPU clusters and fine-tuning.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    // Theme class is toggled client-side, so allow the mismatch.
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Apply the saved theme before first paint, on every page. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{if(localStorage.getItem("theme")==="light")document.documentElement.classList.add("light")}catch(e){}`,
          }}
        />
        {/* Visitor counts (Plausible: no cookies, no personal data). */}
        <script
          defer
          data-domain="vram-calculator.mariusbrt.com"
          src="https://plausible.mariusbrt.com/js/script.outbound-links.js"
        />
      </head>
      <body className="flex min-h-screen flex-col bg-background text-foreground antialiased selection:bg-primary/30 font-sans">
        {children}
        <SiteFooter />
      </body>
    </html>
  );
}
