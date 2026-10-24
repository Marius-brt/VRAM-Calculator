import type { NextConfig } from "next";

// GitHub Pages: a static export served under /<repo> (see .github/workflows/pages.yml). Unset, it's a normal Next server.
const pagesBasePath = process.env.PAGES_BASE_PATH;

const nextConfig: NextConfig = {
  experimental: { agentUpgrade: "security" },
  ...(pagesBasePath === undefined ? {} : { output: "export", basePath: pagesBasePath }),
};

export default nextConfig;
