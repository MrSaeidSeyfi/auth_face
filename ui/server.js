import express from "express";
import fetch from "node-fetch";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { exec } from "child_process";
import onvif from "onvif";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.UI_PORT || 3000;
const FACE_AUTH_URL = process.env.FACE_AUTH_URL || "http://localhost:8000";

app.use(express.json({ limit: "10mb" }));
app.use(express.static(path.join(__dirname, "public")));

const forward = async (endpoint, options = {}) => {
  const response = await fetch(`${FACE_AUTH_URL}${endpoint}`, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
};

const execPromise = (command) =>
  new Promise((resolve) => {
    exec(command, { timeout: 4000 }, (error, stdout, stderr) => {
      if (error) {
        resolve({ ok: false, error: stderr || error.message });
        return;
      }
      resolve({ ok: true, output: stdout });
    });
  });

const detectWindowsCameras = async () => {
  const { ok, output } = await execPromise('ffmpeg -list_devices true -f dshow -i dummy');
  if (!ok) return [];
  const lines = output.split('\n').filter((line) => line.includes('DirectShow video devices'));
  const devices = [];
  let capture = false;
  lines.forEach((line) => {
    if (line.includes('DirectShow video devices')) capture = true;
    else if (capture && line.includes('"')) {
      const name = line.match(/"(.+?)"/)?.[1];
      if (name) devices.push({ id: name, label: name, source: 'local' });
    }
  });
  return devices;
};


const parseV4L = (output) => {
  const devices = [];
  const lines = output.split("\n");
  let current;
  lines.forEach((line) => {
    if (!line.trim()) return;
    if (!line.startsWith("\t")) {
      current = { label: line.trim(), nodes: [] };
      devices.push(current);
    } else if (current) {
      current.nodes.push(line.trim());
    }
  });
  return devices.map((device) => ({
    id: device.nodes[0] || device.label,
    label: device.label,
    source: "local",
  }));
};

const detectLinuxCameras = async () => {
  const v4l = await execPromise("v4l2-ctl --list-devices");
  if (v4l.ok) return parseV4L(v4l.output);
  const fallback = await execPromise("ls /dev/video*");
  if (!fallback.ok) return [];
  return fallback.output
    .split(/\s+/)
    .filter(Boolean)
    .map((node) => ({ id: node, label: node, source: "local" }));
};

const { Discovery } = onvif;

const detectNetworkCameras = () =>
  new Promise((resolve) => {
    const devices = [];
    Discovery.probe((err, cams) => {
      if (!err && cams) {
        cams.forEach((cam) => {
          devices.push({
            id: cam.urn || cam.name || cam.address,
            label: cam.name || cam.address,
            host: cam.address,
            source: "network",
          });
        });
      }
    });
    setTimeout(() => resolve(devices), 4000);
  });

const getLocalCameras = async () => {
  const platform = os.platform();
  if (platform === "win32") return { platform, devices: await detectWindowsCameras() };
  if (platform === "linux") return { platform, devices: await detectLinuxCameras() };
  return { platform, devices: [] };
};

app.get("/api/health", async (_req, res) => {
  try {
    const status = await forward("/health");
    res.json({ upstream: status.status, ui: "ok" });
  } catch (error) {
    res.status(503).json({ upstream: "down", error: error.message });
  }
});

app.get("/api/cameras/local", async (_req, res) => {
  try {
    const response = await getLocalCameras();
    res.json(response);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/cameras/network", async (_req, res) => {
  try {
    const devices = await detectNetworkCameras();
    res.json({ devices });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/cameras/scan", async (_req, res) => {
  try {
    const [local, network] = await Promise.all([getLocalCameras(), detectNetworkCameras()]);
    res.json({ local: local.devices, network });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/login", async (req, res) => {
  try {
    const payload = {
      image: req.body.image,
      issue_session: true,
    };
    const result = await forward("/auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post("/api/enroll", async (req, res) => {
  try {
    const result = await forward("/enroll", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get("/api/faces", async (_req, res) => {
  try {
    const faces = await forward("/faces");
    res.json(faces);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post("/api/logout", async (req, res) => {
  try {
    const result = await forward("/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`UI server running on http://localhost:${PORT}`);
});

