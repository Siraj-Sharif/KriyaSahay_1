"use strict";

/**
 * Owns the single Python process started by the Electron desktop shell.
 * This module deliberately knows nothing about CV or hardware; Python remains the
 * owner of the pipeline, camera and serial link. It is kept Electron-free so its
 * process-selection and lifecycle behavior can be verified in Node tests.
 */
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

function pythonLaunchSpec(projectRoot, env, platform, existsSync = fs.existsSync) {
  if (env.NEUROGRIP_PYTHON) {
    return { command: env.NEUROGRIP_PYTHON, prefixArgs: [] };
  }

  const virtualEnv = env.VIRTUAL_ENV;
  const candidates = [
    virtualEnv && path.join(virtualEnv, platform === "win32" ? "Scripts" : "bin", platform === "win32" ? "python.exe" : "python"),
    path.join(projectRoot, ".venv", platform === "win32" ? "Scripts" : "bin", platform === "win32" ? "python.exe" : "python"),
    path.join(projectRoot, "pc", ".venv", platform === "win32" ? "Scripts" : "bin", platform === "win32" ? "python.exe" : "python"),
  ].filter(Boolean);

  const executable = candidates.find((candidate) => existsSync(candidate));
  if (executable) return { command: executable, prefixArgs: [] };
  return { command: platform === "win32" ? "python" : "python3", prefixArgs: [] };
}

function createBackendSupervisor(options = {}) {
  const projectRoot = path.resolve(options.projectRoot || path.resolve(__dirname, ".."));
  const env = options.env || process.env;
  const platform = options.platform || process.platform;
  const spawnProcess = options.spawn || spawn;
  const existsSync = options.existsSync || fs.existsSync;
  const onStatus = options.onStatus || (() => {});
  const onOutput = options.onOutput || (() => {});

  let child = null;
  let stopping = false;
  let status = { state: "stopped", message: "Python backend has not been started", pid: null };
  let startPromise = null;

  function publish(next) {
    status = { ...status, ...next };
    try {
      onStatus({ ...status });
    } catch {
      // A closed window must not interrupt process lifecycle.
    }
  }

  function isAlive(target = child) {
    return Boolean(target && target.exitCode === null && target.signalCode === null);
  }

  function waitForExit(target, timeoutMs) {
    if (!isAlive(target)) return Promise.resolve(true);
    return new Promise((resolve) => {
      let settled = false;
      const finish = (exited) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        target.removeListener("exit", onExit);
        resolve(exited);
      };
      const onExit = () => finish(true);
      const timer = setTimeout(() => finish(!isAlive(target)), timeoutMs);
      timer.unref?.();
      target.once("exit", onExit);
    });
  }

  function start() {
    if (isAlive()) return Promise.resolve({ ok: true, pid: child.pid, status: { ...status } });
    if (startPromise) return startPromise;

    const pcRoot = path.join(projectRoot, "pc");
    const packageDir = path.join(pcRoot, "src", "neurogrip");
    if (!existsSync(packageDir)) {
      const error = `Python package not found at ${packageDir}. Set NEUROGRIP_PROJECT_ROOT to the project directory.`;
      publish({ state: "error", message: error, pid: null });
      return Promise.resolve({ ok: false, error });
    }

    const python = pythonLaunchSpec(projectRoot, env, platform, existsSync);
    const pythonPath = [path.join(pcRoot, "src"), env.PYTHONPATH].filter(Boolean).join(path.delimiter);
    const childEnv = {
      ...env,
      PYTHONPATH: pythonPath,
      PYTHONUNBUFFERED: "1",
    };
    const args = [...python.prefixArgs, "-m", "neurogrip", "--no-gui", "--desktop"];

    publish({ state: "starting", message: `Starting Python backend (${python.command})…`, pid: null });
    stopping = false;

    startPromise = new Promise((resolve) => {
      let settled = false;
      const finish = (result) => {
        if (settled) return;
        settled = true;
        startPromise = null;
        resolve(result);
      };

      let proc;
      try {
        proc = spawnProcess(python.command, args, {
          cwd: pcRoot,
          env: childEnv,
          windowsHide: true,
          stdio: ["ignore", "pipe", "pipe"],
        });
      } catch (error) {
        const message = `Could not start Python backend: ${error.message || error}`;
        publish({ state: "error", message, pid: null });
        finish({ ok: false, error: message });
        return;
      }

      child = proc;
      proc.stdout?.on("data", (data) => onOutput("stdout", String(data)));
      proc.stderr?.on("data", (data) => onOutput("stderr", String(data)));

      proc.once("spawn", () => {
        publish({ state: "starting", message: "Python started; waiting for the pipeline bridge…", pid: proc.pid || null });
        finish({ ok: true, pid: proc.pid || null });
      });

      proc.once("error", (error) => {
        const message = `Could not start Python backend: ${error.message || error}`;
        publish({ state: "error", message, pid: null });
        finish({ ok: false, error: message });
      });

      proc.once("exit", (code, signal) => {
        const expected = stopping;
        if (child === proc) child = null;
        stopping = false;
        const detail = signal ? `signal ${signal}` : `exit code ${code}`;
        publish(
          expected
            ? { state: "stopped", message: "Python backend stopped cleanly", pid: null }
            : { state: "error", message: `Python backend exited unexpectedly (${detail}).`, pid: null },
        );
        finish(expected ? { ok: true, stopped: true } : { ok: false, error: status.message });
      });
    });

    return startPromise;
  }

  async function stop(options = {}) {
    const target = child;
    if (!isAlive(target)) return { ok: true, stopped: true };

    stopping = true;
    publish({ state: "stopping", message: "Stopping Python backend and releasing devices…" });

    if (typeof options.requestShutdown === "function") {
      try {
        await Promise.race([
          Promise.resolve(options.requestShutdown()),
          new Promise((resolve) => setTimeout(resolve, options.requestTimeoutMs || 1500)),
        ]);
      } catch {
        // Fall back to process termination below if the bridge has already gone away.
      }
    }

    const graceful = await waitForExit(target, options.gracefulTimeoutMs || 3500);
    if (!graceful && isAlive(target)) {
      try {
        target.kill("SIGTERM");
      } catch {
        // Continue to the final kill attempt.
      }
    }

    const terminated = graceful || await waitForExit(target, options.terminateTimeoutMs || 1000);
    if (!terminated && isAlive(target)) {
      try {
        target.kill("SIGKILL");
      } catch {
        // The OS will still close the child when Electron exits; report status below.
      }
      await waitForExit(target, 1000);
    }

    if (child === target && !isAlive(target)) child = null;
    if (status.state !== "stopped") {
      publish({ state: "stopped", message: "Python backend stopped", pid: null });
    }
    stopping = false;
    return { ok: true, stopped: !isAlive(target) };
  }

  function getStatus() {
    return { ...status };
  }

  return { start, stop, getStatus, isRunning: () => isAlive() };
}

module.exports = { createBackendSupervisor, pythonLaunchSpec };
