// From the repo root: node gui/wails/build/make-icons.mjs
// Requires installed frontend Playwright/Chromium and Python with Pillow; never installs packages.
// With shared dependencies: --playwright-package C:/path/to/frontend/package.json
// Optional: --evidence C:/path/to/report (both candidates, previews and validation JSON).
import { createRequire } from 'node:module';
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { execFileSync } from 'node:child_process';

const build = dirname(fileURLToPath(import.meta.url));
const wails = dirname(build);
const { values } = parseArgs({ options: {
  'playwright-package': { type: 'string' },
  evidence: { type: 'string' },
  python: { type: 'string', default: 'python' },
} });
const source = await readFile(join(wails, 'icons/owl.svg'));
const css = await readFile(join(wails, 'frontend/src/styles/app.css'), 'utf8');
const sidebarToken = css.match(/--color-sidebar:\s*var\((--[a-z-]+)\)/)?.[1];
const darkTokens = css.match(/:root\[data-theme='dark'\]\s*\{([^}]+)\}/)?.[1];
const sidebarBackground = sidebarToken && darkTokens?.match(
  new RegExp(`${sidebarToken}:\\s*(#[\\da-f]{6});`, 'i'),
)?.[1];
if (!sidebarBackground) throw new Error('Cannot resolve the dark sidebar background token');
const evidence = values.evidence ? resolve(values.evidence) : null;
if (evidence) await mkdir(evidence, { recursive: true });
await mkdir(join(build, 'windows'), { recursive: true });

// A: brand red on cream. B (chosen): the original coral owl on the dark sidebar.
const candidates = {
  A: { paper: '#f4efe6', owl: '#a00000' },
  B: { paper: sidebarBackground, owl: '#f35555' },
};
function luminance(hex) {
  const rgb = hex.slice(1).match(/../g).map(c => parseInt(c, 16) / 255)
    .map(c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
}
for (const palette of Object.values(candidates)) {
  const light = Math.max(luminance(palette.paper), luminance(palette.owl));
  const dark = Math.min(luminance(palette.paper), luminance(palette.owl));
  palette.contrast = (light + 0.05) / (dark + 0.05);
  if (palette.contrast < 4.5) throw new Error('Owl contrast must be at least 4.5:1');
}
const require = createRequire(values['playwright-package']
  ? resolve(values['playwright-package']) : join(wails, 'frontend/package.json'));
const { chromium } = require('playwright');
const { format } = require('prettier');
const browser = await chromium.launch({ headless: true, args: ['--disable-gpu'] });
try {
  const page = await browser.newPage({ viewport: { width: 1024, height: 1024 }, deviceScaleFactor: 1 });
  for (const [name, palette] of Object.entries(candidates)) {
    if (name !== 'B' && !evidence) continue;
    // Preserve the source path; 28/32 inset and .9 stroke keep the tiny outline readable.
    const owl = source.toString('utf8')
      .replace('<svg ', '<svg x="2" y="2" width="28" height="28" ')
      .replaceAll('#a00000', palette.owl).replace('stroke-width=".5"', 'stroke-width=".9"');
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 32 32">
      <rect width="32" height="32" rx="7.04" fill="${palette.paper}"/>${owl}</svg>`;
    await page.setContent(`<style>html,body{margin:0;background:transparent}svg{display:block}</style>${svg}`);
    const output = evidence ? join(evidence, `candidate-${name}.png`) : join(build, 'appicon.png');
    await page.screenshot({ path: output, omitBackground: true, animations: 'disabled' });
    if (name === 'B' && evidence) await copyFile(output, join(build, 'appicon.png'));
    if (name === 'B') await writeFile(join(wails, 'frontend/public/favicon.svg'), await format(svg, { parser: 'html' }));
  }
} finally {
  await browser.close();
}
execFileSync(values.python, ['-I', join(build, 'pack-icons.py'),
  ...(evidence ? ['--evidence', evidence] : [])], { stdio: 'inherit' });
console.log(JSON.stringify({ chosen: 'B', candidates }, null, 2));
