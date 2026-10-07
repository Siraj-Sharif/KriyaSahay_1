// Copies the built web UI (../dist/index.html) into electron/renderer/.
// With --build it rebuilds the UI first.
const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..", "..");
const dist = path.join(root, "dist", "index.html");
const out = path.join(__dirname, "..", "renderer");

if (process.argv.includes("--build") || !fs.existsSync(dist)) {
  console.log("› Building the NeuroGrip UI…");
  execSync("npm run build", { cwd: root, stdio: "inherit", shell: true });
}
if (!fs.existsSync(dist)) {
  console.error("✗ dist/index.html not found. Run `npm install && npm run build` in the project root first.");
  process.exit(1);
}
fs.mkdirSync(out, { recursive: true });
fs.copyFileSync(dist, path.join(out, "index.html"));
console.log("✓ UI copied to electron/renderer/index.html");
