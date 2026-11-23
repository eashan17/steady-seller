// Frame capture for demo/static/video.html through Edge's DevTools protocol (no extra packages).
//   node demo/capture.mjs <outDir> stills 10,40,100        -> <outDir>/still_<t>.png
//   node demo/capture.mjs <outDir> frames [fps] [from] [to] -> <outDir>/f_000000.jpg ...
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [outDir, mode = "stills", a1, a2, a3] = process.argv.slice(2);
const URL_ = process.env.VIDEO_URL || "http://127.0.0.1:8765/video.html";
const EDGE = process.env.EDGE || "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const PORT = 9333 + Math.floor(Math.random() * 500);
mkdirSync(outDir, { recursive: true });

const proc = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${PORT}`, "--disable-gpu", "--hide-scrollbars",
  "--force-device-scale-factor=1", `--user-data-dir=${join(process.env.TEMP || ".", "edge-cap-" + PORT)}`,
  "--window-size=1920,1080", "about:blank"], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws, id = 0;
const pending = new Map();
function send(method, params = {}) {
  return new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
}

async function main() {
  let target;
  for (let i = 0; i < 60 && !target; i++) {
    try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find((t) => t.type === "page"); } catch { await sleep(250); }
  }
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.rej(new Error(JSON.stringify(m.error))) : p.res(m.result); } };
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: URL_ });
  let ready = false;
  for (let i = 0; i < 480 && !ready; i++) {
    const r = await send("Runtime.evaluate", { expression: "window.videoReady && window.videoReady.then(()=>true)", awaitPromise: true, returnByValue: true }).catch(() => null);
    ready = !!(r && r.result && r.result.value === true);
    if (!ready) await sleep(250);
  }
  if (!ready) throw new Error("video page never became ready");
  const shot = async (t, file, fmt) => {
    const r = await send("Runtime.evaluate", { expression: `renderAt(${t})`, returnByValue: true });
    if (!r.result || typeof r.result.value !== "string") throw new Error(`renderAt(${t}) failed: ${JSON.stringify(r.exceptionDetails || r.result)}`);
    const s = await send("Page.captureScreenshot", fmt === "png" ? { format: "png" } : { format: "jpeg", quality: 92 });
    writeFileSync(file, Buffer.from(s.data, "base64"));
  };
  if (mode === "stills") {
    for (const t of (a1 || "0").split(",").map(Number)) await shot(t, join(outDir, `still_${t}.png`), "png");
  } else {
    const fps = +(a1 || 30), from = +(a2 || 0), to = +(a3 || (await send("Runtime.evaluate", { expression: "DURATION", returnByValue: true })).result.value);
    const n0 = Math.round(from * fps), n1 = Math.round(to * fps);
    const t0 = Date.now();
    for (let n = n0; n < n1; n++) {
      await shot(n / fps, join(outDir, `f_${String(n).padStart(6, "0")}.jpg`), "jpeg");
      if (n % 300 === 0) console.log(`frame ${n}/${n1} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
  }
  ws.close();
  proc.kill();
}
main().then(() => process.exit(0), (e) => { console.error(e); proc.kill(); process.exit(1); });
