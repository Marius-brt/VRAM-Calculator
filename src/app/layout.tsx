import type { Metadata } from "next";
import { IBM_Plex_Sans } from "next/font/google";
import "./globals.css";

const plex = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: "VRAM Calculator",
  description:
    "Check whether an LLM fits on your GPU and how fast it runs, on NVIDIA, AMD, Apple and Intel hardware, multi-GPU clusters and fine-tuning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // Theme class is toggled client-side, so allow the mismatch.
    <html lang="en" className={plex.variable} suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground antialiased selection:bg-primary/30 font-sans">
        {children}
      </body>
    </html>
  );
}
