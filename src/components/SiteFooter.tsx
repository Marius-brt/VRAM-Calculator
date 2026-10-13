import Link from "next/link";
import { Github } from "lucide-react";

export const REPO_URL = "https://github.com/Marius-brt/VRAM-Calculator";
export const AUTHOR = { name: "Marius Brouty", url: "https://github.com/Marius-brt" };

export function SiteFooter() {
  return (
    <footer className="border-t">
      <div className="mx-auto flex max-w-[1320px] flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-5 text-[13px] text-muted-foreground sm:px-6">
        <p>
          VRAM Calculator by{" "}
          <a href={AUTHOR.url} className="text-foreground underline-offset-2 hover:underline">
            {AUTHOR.name}
          </a>
          . Estimates, not measurements of your system.
        </p>
        <nav className="flex items-center gap-5">
          <Link href="/methodology" className="hover:text-heading">
            Methodology
          </Link>
          <a href={REPO_URL} className="inline-flex items-center gap-1.5 hover:text-heading">
            <Github className="h-4 w-4" />
            GitHub
          </a>
        </nav>
      </div>
    </footer>
  );
}
