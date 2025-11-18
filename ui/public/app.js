const loginPreview = document.getElementById("login-preview");
const managePreview = document.getElementById("manage-preview");
const snapshot = document.getElementById("snapshot");
const cameraSelect = document.getElementById("camera-select");
const refreshButton = document.getElementById("refresh-cameras");
const scanButton = document.getElementById("scan-host");
const loginCapture = document.getElementById("login-capture");
const manageCapture = document.getElementById("manage-capture");
const loginButton = document.getElementById("login");
const enrollButton = document.getElementById("enroll");
const logoutButton = document.getElementById("logout");
const refreshFacesButton = document.getElementById("refresh-faces");
const loginStatus = document.getElementById("login-status");
const manageStatus = document.getElementById("manage-status");
const sessionBox = document.getElementById("session-info");
const faceList = document.getElementById("face-list");
const cameraList = document.getElementById("camera-list");
const enrollInput = document.getElementById("enroll-name");
const tabs = document.querySelectorAll("[data-panel-target]");
const panels = document.querySelectorAll("[data-panel]");

let stream;
let lastToken = null;

const setActivePanel = (targetId) => {
  tabs.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.panelTarget === targetId);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("hidden", panel.id !== targetId);
  });
};

tabs.forEach((btn) =>
  btn.addEventListener("click", () => setActivePanel(btn.dataset.panelTarget))
);

const startStream = async (deviceId) => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  const constraints = {
    video: deviceId ? { deviceId: { exact: deviceId } } : true,
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  [loginPreview, managePreview].forEach((video) => {
    if (video) video.srcObject = stream;
  });
};

const populateDevices = async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  cameraSelect.innerHTML = "";
  const videoDevices = devices.filter((device) => device.kind === "videoinput");
  videoDevices.forEach((device, idx) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${idx + 1}`;
    cameraSelect.appendChild(option);
  });
  if (videoDevices.length && !cameraSelect.value) {
    cameraSelect.value = videoDevices[0].deviceId;
  }
};

const getImageData = (source) => {
  const context = snapshot.getContext("2d");
  context.drawImage(source, 0, 0, snapshot.width, snapshot.height);
  return snapshot.toDataURL("image/jpeg");
};

const request = async (url, options) => {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
};

const display = (element, data) => {
  element.textContent = JSON.stringify(data, null, 2);
};

const login = async () => {
  const image = getImageData(loginPreview);
  const data = await request("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image }),
  });
  lastToken = data.token;
  display(loginStatus, data);
  display(sessionBox, { token: data.token, label: data.label });
};

const enroll = async () => {
  const name = enrollInput.value.trim();
  if (!name) throw new Error("Name required");
  const image = getImageData(managePreview);
  const data = await request("/api/enroll", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, image }),
  });
  display(manageStatus, data);
  await loadFaces();
};

const logout = async () => {
  if (!lastToken) return;
  await request("/api/logout", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: lastToken }),
  });
  lastToken = null;
  display(sessionBox, { status: "logged out" });
};

const scanCameras = async () => {
  const [local, network] = await Promise.all([
    request("/api/cameras/local"),
    request("/api/cameras/network"),
  ]);
  display(cameraList, { local: local.devices, network: network.devices });
};

const loadFaces = async () => {
  const faces = await request("/api/faces");
  display(faceList, faces);
};

cameraSelect.addEventListener("change", (event) => {
  const deviceId = event.target.value;
  startStream(deviceId).catch((err) => display(loginStatus, { error: err.message }));
});

refreshButton.addEventListener("click", () =>
  populateDevices().then(() => startStream(cameraSelect.value)).catch((err) =>
    display(loginStatus, { error: err.message })
  )
);

scanButton.addEventListener("click", () =>
  scanCameras().catch((err) => display(cameraList, { error: err.message }))
);

loginCapture.addEventListener("click", () => {
  getImageData(loginPreview);
  display(loginStatus, { snapshot: "ready" });
});

manageCapture.addEventListener("click", () => {
  getImageData(managePreview);
  display(manageStatus, { snapshot: "ready" });
});

loginButton.addEventListener("click", () =>
  login().catch((err) => display(loginStatus, { error: err.message }))
);

enrollButton.addEventListener("click", () =>
  enroll().catch((err) => display(manageStatus, { error: err.message }))
);

refreshFacesButton.addEventListener("click", () =>
  loadFaces().catch((err) => display(manageStatus, { error: err.message }))
);

logoutButton.addEventListener("click", () =>
  logout().catch((err) => display(sessionBox, { error: err.message }))
);

const initialize = async () => {
  await populateDevices();
  await startStream(cameraSelect.value);
  await scanCameras();
  await loadFaces();
  setActivePanel("login-panel");
};

initialize().catch((err) => display(loginStatus, { error: err.message }));

