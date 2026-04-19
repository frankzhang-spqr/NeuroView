/* global THREE */

/* ═══════════════════════════════════════════════════════════
   NeuroView MRI — Desktop Dashboard Controller
   ═══════════════════════════════════════════════════════════ */

const modalityMeta = {
    t1c: { label: "T1 Contrast", colorClass: "cyan", fullLabel: "T1 Contrast-Enhanced" },
    t1n: { label: "T1 Native", colorClass: "lilac", fullLabel: "T1 Native" },
    t2f: { label: "T2 FLAIR", colorClass: "green", fullLabel: "T2 FLAIR" },
    t2w: { label: "T2 Weighted", colorClass: "orange", fullLabel: "T2 Weighted" }
};

const axisLabels = { axial: "AXI", coronal: "COR", sagittal: "SAG" };

const state = {
    results: null,
    currentAxis: "axial",
    selectedSliceIndex: 0,
    currentSlicePreview: null,
    panel3dVisible: true,
    panelSlicesVisible: true,
    current3DModality: "t1c",
    viewMode: "dots",
    renderer: null,
    animationId: null
};

// ── DOM refs ──
const fileInputs = Array.from(document.querySelectorAll(".file-input"));
const sliceGrid = document.getElementById("slice-grid");
const tumorSlicesContainer = document.getElementById("tumor-slices");
const tumorSummary = document.getElementById("tumor-summary");
const sliceSlider = document.getElementById("slice-slider");
const sliceCounter = document.getElementById("slice-counter");
const sliceCounterStatus = document.getElementById("slice-counter-status");
const axisButtons = Array.from(document.querySelectorAll(".axis-btn"));
const modalityTabs = Array.from(document.querySelectorAll(".modality-tab"));
const sidebar = document.getElementById("sidebar");
const statusBar = document.getElementById("status-bar");
const fileBadge = document.getElementById("file-count-badge");
const statusText = document.getElementById("status-text");
const statusIndicator = document.getElementById("status-indicator");
const statusAxis = document.getElementById("status-axis");
const statusSlice = document.getElementById("status-slice");
const statusFiles = document.getElementById("status-files");
const sliceStatusDot = document.getElementById("slice-status-indicator");
const viewModeButtons = Array.from(document.querySelectorAll(".view-mode-btn"));

// ── File input handling ──
fileInputs.forEach((input) => {
    input.addEventListener("change", syncUploadState);
});

// ── Volume modality tabs (toolbar) ──
modalityTabs.forEach((tab) => {
    tab.addEventListener("click", async () => {
        if (!state.results) return;
        await switch3DModality(tab.dataset.modality);
    });
});

// ── View mode buttons (toolbar) ──
viewModeButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
        if (btn.dataset.mode === state.viewMode) return;
        state.viewMode = btn.dataset.mode;
        updateViewModeButtons();
        if (state.results) {
            if (state.viewMode === "volume") {
                await fetchAndRenderVolume();
            } else {
                await switch3DModality(state.current3DModality, true); // Force refresh points
            }
        }
    });
});

// ── Axis buttons (toolbar) ──
axisButtons.forEach((button) => {
    button.addEventListener("click", async () => {
        if (!state.results) return;
        await switchAxis(button.dataset.axis);
    });
});

// ── Toolbar: Analyze ──
document.getElementById("tb-analyze").addEventListener("click", runAnalysis);

// ── Toolbar: Reset ──
document.getElementById("tb-reset").addEventListener("click", resetWorkspace);

// ── Toolbar: Open ──
document.getElementById("tb-open").addEventListener("click", () => {
    // Click the first empty file input, or the first one
    const emptyInput = fileInputs.find(i => !i.files.length) || fileInputs[0];
    emptyInput.click();
});

// ── Toolbar: Toggle sidebar ──
document.getElementById("tb-toggle-sidebar").addEventListener("click", toggleSidebar);

// ── Toolbar: Panel toggles ──
document.getElementById("tb-toggle-3d").addEventListener("click", () => togglePanel("3d"));
document.getElementById("tb-toggle-slices").addEventListener("click", () => togglePanel("slices"));

// ── Slice nav ──
document.getElementById("prev-slice").addEventListener("click", async () => {
    if (!state.results) return;
    state.selectedSliceIndex = Math.max(0, state.selectedSliceIndex - 1);
    await refreshCurrentSlice();
});

document.getElementById("next-slice").addEventListener("click", async () => {
    if (!state.results) return;
    const maxIndex = state.results.slice_counts[state.currentAxis] - 1;
    state.selectedSliceIndex = Math.min(maxIndex, state.selectedSliceIndex + 1);
    await refreshCurrentSlice();
});

sliceSlider.addEventListener("input", async (event) => {
    state.selectedSliceIndex = Number(event.target.value);
    await refreshCurrentSlice();
});

// ── Shortcuts modal ──
document.getElementById("close-shortcuts").addEventListener("click", () => {
    document.getElementById("shortcuts-dialog").close();
});

// ═══════════════════════════════════════════════════════════
//  CORE FUNCTIONS
// ═══════════════════════════════════════════════════════════

async function runAnalysis() {
    const formData = new FormData();
    let uploadedCount = 0;

    fileInputs.forEach((input) => {
        if (input.files.length > 0) {
            formData.append(input.id, input.files[0]);
            uploadedCount += 1;
        }
    });

    if (uploadedCount < 4) {
        alert("Please upload all four modality files.");
        return;
    }

    setLoadingState(true);

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        if (!response.ok) throw new Error("Error processing the file.");
        const results = await response.json();
        if (results.error) throw new Error(results.error);
        await displayResults(results);
    } catch (error) {
        alert(`Analysis failed: ${error.message}`);
        setLoadingState(false);
    }
}

function resetWorkspace() {
    fileInputs.forEach((input) => { input.value = ""; });
    state.results = null;
    state.currentAxis = "axial";
    state.selectedSliceIndex = 0;
    state.currentSlicePreview = null;
    state.current3DModality = "t1c";
    state.viewMode = "dots";
    updateViewModeButtons();
    resetResultsUI();
    syncUploadState();
}

function toggleSidebar() {
    state.sidebarVisible = !state.sidebarVisible;
    sidebar.classList.toggle("collapsed", !state.sidebarVisible);
}

function toggleStatusBar() {
    state.statusBarVisible = !state.statusBarVisible;
    statusBar.classList.toggle("hidden", !state.statusBarVisible);
}

function togglePanel(panel) {
    const panel3d = document.getElementById("panel-volume");
    const panelSlices = document.getElementById("panel-slices");
    const btn3d = document.getElementById("tb-toggle-3d");
    const btnSlices = document.getElementById("tb-toggle-slices");

    if (panel === "3d") {
        // Don't allow hiding both
        if (state.panel3dVisible && !state.panelSlicesVisible) return;
        state.panel3dVisible = !state.panel3dVisible;
    } else {
        if (state.panelSlicesVisible && !state.panel3dVisible) return;
        state.panelSlicesVisible = !state.panelSlicesVisible;
    }

    panel3d.classList.toggle("hidden-panel", !state.panel3dVisible);
    panelSlices.classList.toggle("hidden-panel", !state.panelSlicesVisible);

    // Solo class for full-width when only one visible
    const onlyOne = !state.panel3dVisible || !state.panelSlicesVisible;
    panel3d.classList.toggle("solo", onlyOne && state.panel3dVisible);
    panelSlices.classList.toggle("solo", onlyOne && state.panelSlicesVisible);

    btn3d.classList.toggle("active", state.panel3dVisible);
    btnSlices.classList.toggle("active", state.panelSlicesVisible);

    // Trigger resize for the 3D renderer
    window.dispatchEvent(new Event("resize"));
}

async function displayResults(results) {
    state.results = results;
    state.currentAxis = "axial";
    state.selectedSliceIndex = getInitialSliceIndex("axial");

    renderSummary(results);
    renderVolume(results.volume_points);
    updateAxisButtons();
    updateModalityTabs();
    configureSliceControls();
    renderTumorSlices();
    await refreshCurrentSlice();
    setLoadingState(false);
    updateStatus("Analysis complete", "ready");
}

function syncUploadState() {
    let count = 0;
    fileInputs.forEach((input) => {
        const modality = input.id.replace("_file", "");
        const nameNode = document.getElementById(`${modality}_name`);
        const slot = input.closest(".upload-slot");
        const hasFile = input.files.length > 0;

        if (hasFile) {
            count += 1;
            nameNode.textContent = input.files[0].name;
            slot.classList.add("has-file");
        } else {
            nameNode.textContent = "No file selected";
            slot.classList.remove("has-file");
        }
    });

    fileBadge.textContent = `${count} / 4`;
    statusFiles.textContent = `Files: ${count}/4`;
    const analyzeBtn = document.getElementById("tb-analyze");
    analyzeBtn.disabled = count < 4;
}

function setLoadingState(isLoading) {
    const analyzeBtn = document.getElementById("tb-analyze");
    analyzeBtn.disabled = isLoading || fileInputs.filter(i => i.files.length).length < 4;
    const label = analyzeBtn.querySelector("span");
    if (label) label.textContent = isLoading ? "Analyzing…" : "Analyze";

    if (isLoading) {
        updateStatus("Running analysis…", "busy");
    }
}

function updateStatus(text, mode) {
    statusText.textContent = text;
    statusIndicator.className = "status-indicator";
    if (mode === "busy") statusIndicator.classList.add("busy");
    else if (mode === "error") statusIndicator.classList.add("error");
}

function getInitialSliceIndex(axis) {
    const tumorSlices = state.results?.tumor_slices_by_axis?.[axis] || [];
    return tumorSlices.length ? tumorSlices[0] : 0;
}

function updateAxisButtons() {
    axisButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.axis === state.currentAxis);
    });
    statusAxis.textContent = `Axis: ${state.currentAxis.charAt(0).toUpperCase() + state.currentAxis.slice(1)}`;
}

function updateModalityTabs() {
    modalityTabs.forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.modality === state.current3DModality);
    });
}

function updateViewModeButtons() {
    viewModeButtons.forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.mode === state.viewMode);
    });
}

async function switch3DModality(modality, force = false) {
    if (!force && modality === state.current3DModality) return;
    state.current3DModality = modality;
    updateModalityTabs();
    
    if (state.viewMode === "volume") {
        await fetchAndRenderVolume();
        return;
    }
    
    updateStatus(`Loading 3D ${modality.toUpperCase()}…`, "busy");
    
    try {
        const response = await fetch(`/volume-preview/${state.results.scan_id}?modality=${modality}`);
        const payload = await response.json();
        if (payload.error) throw new Error(payload.error);
        
        renderVolume(payload.volume_points);
        updateStatus("Ready", "ready");
    } catch (error) {
        updateStatus(`Failed to load ${modality}: ${error.message}`, "error");
    }
}

function configureSliceControls() {
    const maxValue = Math.max(0, (state.results?.slice_counts?.[state.currentAxis] || 1) - 1);
    sliceSlider.max = maxValue;
    sliceSlider.disabled = !state.results;
    sliceSlider.value = state.selectedSliceIndex;
}

async function switchAxis(axis) {
    if (axis === state.currentAxis) return;
    state.currentAxis = axis;
    const maxIndex = state.results.slice_counts[axis] - 1;
    const tumorSlices = state.results.tumor_slices_by_axis[axis] || [];
    state.selectedSliceIndex = tumorSlices.length ? tumorSlices[0] : Math.min(state.selectedSliceIndex, maxIndex);

    updateAxisButtons();
    configureSliceControls();
    renderTumorSlices();
    await refreshCurrentSlice();
}

async function refreshCurrentSlice() {
    if (!state.results) {
        renderEmptySliceGrid();
        return;
    }

    const response = await fetch(`/slice-preview/${state.results.scan_id}?axis=${state.currentAxis}&index=${state.selectedSliceIndex}`);
    const payload = await response.json();
    if (payload.error) {
        renderEmptySliceGrid(payload.error);
        return;
    }

    state.currentSlicePreview = payload;
    renderSelectedSlice();
}

// ═══════════════════════════════════════════════════════════
//  RENDERERS
// ═══════════════════════════════════════════════════════════

function renderSummary(results) {
    const affectedCount = results.tumor_slices_by_axis.axial.length;

    if (results.has_tumor) {
        tumorSummary.className = "result-card positive";
        tumorSummary.innerHTML = `
            <div class="result-card-icon">!</div>
            <div class="result-card-body">
                <strong>Tumor Detected</strong>
                <p>Type: <b>${results.tumor_type}</b> — ${affectedCount} affected slices</p>
            </div>
        `;
        return;
    }

    tumorSummary.className = "result-card negative";
    tumorSummary.innerHTML = `
        <div class="result-card-icon">✓</div>
        <div class="result-card-body">
            <strong>No Tumor Detected</strong>
            <p>No suspicious regions identified in this scan set.</p>
        </div>
    `;
}

function renderTumorSlices() {
    const tumorSlices = state.results?.tumor_slices_by_axis?.[state.currentAxis] || [];

    if (!tumorSlices.length) {
        tumorSlicesContainer.innerHTML = '<span class="empty-pill">No tumor slices</span>';
        return;
    }

    tumorSlicesContainer.innerHTML = "";
    tumorSlices.forEach((sliceIndex) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `tumor-pill${sliceIndex === state.selectedSliceIndex ? " active" : ""}`;
        button.textContent = sliceIndex;
        button.addEventListener("click", async () => {
            state.selectedSliceIndex = sliceIndex;
            configureSliceControls();
            await refreshCurrentSlice();
        });
        tumorSlicesContainer.appendChild(button);
    });
}

function renderSelectedSlice() {
    const preview = state.currentSlicePreview;
    if (!preview) { renderEmptySliceGrid(); return; }

    const slicesByModality = {};
    preview.slices.forEach((item) => { slicesByModality[item.modality] = item; });

    sliceGrid.innerHTML = Object.keys(modalityMeta).map((modality) => {
        const slice = slicesByModality[modality];
        const meta = modalityMeta[modality];
        const hasTumor = Boolean(slice && slice.has_tumor);
        const bboxMarkup = slice && slice.bbox ? createBBoxMarkup(slice.bbox) : "";
        const imageMarkup = slice
            ? `<div class="slice-image-wrap"><img src="data:image/png;base64,${slice.image}" alt="${meta.fullLabel} ${axisLabels[state.currentAxis]} slice ${preview.slice_index}">${bboxMarkup}</div>`
            : "<p>Unavailable</p>";

        return `
            <div class="slice-panel${hasTumor ? " has-tumor" : ""}">
                <span class="slice-label ${meta.colorClass}">${meta.label}</span>
                ${hasTumor ? '<span class="tumor-tag">TUMOR</span>' : ""}
                <div class="slice-frame">
                    ${imageMarkup}
                    <div class="slice-crosshair-v"></div>
                    <div class="slice-crosshair-h"></div>
                </div>
            </div>
        `;
    }).join("");

    sliceSlider.value = preview.slice_index;
    sliceCounter.textContent = `${preview.slice_index} / ${state.results.slice_counts[state.currentAxis] - 1}`;
    const hasTumor = preview.has_tumor;
    sliceCounterStatus.textContent = hasTumor ? "TUMOR" : "CLEAR";
    sliceStatusDot.className = "slice-status-dot " + (hasTumor ? "tumor" : "clear");
    statusSlice.textContent = `Slice: ${preview.slice_index}/${state.results.slice_counts[state.currentAxis] - 1}`;

    document.querySelectorAll(".tumor-pill").forEach((pill) => {
        pill.classList.toggle("active", Number(pill.textContent) === preview.slice_index);
    });
}

function renderVolume(volumePoints) {
    if (state.viewMode === "volume") {
        // volume rendering is handled by fetchAndRenderVolume
        return;
    }

    const container = document.getElementById("canvas-container");
    
    // Kill existing renderer if it exists
    if (state.renderer) {
        if (state.animationId) cancelAnimationFrame(state.animationId);
        state.renderer.dispose();
        if (state.renderer.domElement.parentElement) {
            state.renderer.domElement.parentElement.removeChild(state.renderer.domElement);
        }
        state.renderer = null;
    }

    container.innerHTML = "";

    const brainPoints = volumePoints?.brain || [];
    const tumorPoints = volumePoints?.tumor || [];
    if (!brainPoints.length && !tumorPoints.length) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none"><path d="m12 2 8 4.5v11L12 22 4 17.5v-11L12 2Zm0 0v9.5m8-5-8 5m0 0-8-5" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg></div>
                <h3>Volume Preview Unavailable</h3>
                <p>Not enough volumetric data to render the preview.</p>
            </div>
        `;
        return;
    }

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    state.renderer = renderer;
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(1.8, 1.5, 2.1);

    const controls = THREE.OrbitControls ? new THREE.OrbitControls(camera, renderer.domElement) : null;
    if (controls) {
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, 0);
    }

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const light = new THREE.DirectionalLight(0xffffff, 0.8);
    light.position.set(2, 3, 4);
    scene.add(light);

    // DICOM-style helpers
    const grid = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
    grid.position.y = -0.6;
    scene.add(grid);

    const axesHelper = new THREE.AxesHelper(0.5);
    axesHelper.position.set(-0.8, -0.5, -0.8);
    scene.add(axesHelper);

    addPointCloud(scene, brainPoints, 0xa8aaad, 0.018, 0.18);
    addPointCloud(scene, tumorPoints, 0xff545e, 0.03, 0.92);
    addBoundingBoxes(scene);

    const animate = () => {
        state.animationId = requestAnimationFrame(animate);
        controls?.update();
        scene.rotation.y += 0.0015;
        renderer.render(scene, camera);
    };
    animate();

    const resizeObserver = new ResizeObserver(() => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    });
    resizeObserver.observe(container);
}

function addPointCloud(scene, points, color, size, opacity) {
    if (!points.length) return;
    const geometry = new THREE.BufferGeometry();
    const attribute = new THREE.BufferAttribute(new Float32Array(points.flat()), 3);
    if (typeof geometry.setAttribute === "function") {
        geometry.setAttribute("position", attribute);
    } else {
        geometry.addAttribute("position", attribute);
    }
    const material = new THREE.PointsMaterial({ color, size, transparent: true, opacity, sizeAttenuation: true });
    scene.add(new THREE.Points(geometry, material));
}

function addBoundingBoxes(scene) {
    const outer = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(1.6, 1.6, 1.6)),
        new THREE.LineBasicMaterial({ color: 0x004f72, transparent: true, opacity: 0.9 })
    );
    scene.add(outer);
    const inner = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(0.95, 0.95, 0.95)),
        new THREE.LineBasicMaterial({ color: 0xff2b36, transparent: true, opacity: 0.95 })
    );
    scene.add(inner);
}

function createBBoxMarkup(bbox) {
    const [x, y, width, height] = bbox;
    return `<div class="bbox" style="left:${x}%;top:${y}%;width:${width}%;height:${height}%"></div>`;
}

function renderEmptySliceGrid(message = "Awaiting data") {
    sliceGrid.innerHTML = Object.keys(modalityMeta).map((modality) => {
        const meta = modalityMeta[modality];
        return `
            <div class="slice-panel placeholder">
                <span class="slice-label ${meta.colorClass}">${meta.label}</span>
                <div class="slice-frame"><p>${message}</p></div>
            </div>
        `;
    }).join("");
    sliceCounter.textContent = "0 / 0";
    sliceCounterStatus.textContent = "NO DATA";
    sliceStatusDot.className = "slice-status-dot";
}

function resetResultsUI() {
    tumorSummary.className = "result-card neutral";
    tumorSummary.innerHTML = `
        <div class="result-card-icon">?</div>
        <div class="result-card-body">
            <strong>Awaiting Scan</strong>
            <p>Upload all 4 modalities and run analysis.</p>
        </div>
    `;
    tumorSlicesContainer.innerHTML = '<span class="empty-pill">No slices yet</span>';
    document.getElementById("canvas-container").innerHTML = `
        <div class="empty-state">
            <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none"><path d="m12 2 8 4.5v11L12 22 4 17.5v-11L12 2Zm0 0v9.5m8-5-8 5m0 0-8-5" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg></div>
            <h3>Upload MRI Scans</h3>
            <p>Upload all four modalities to generate the 3D volume preview.</p>
        </div>
    `;
    sliceSlider.max = 0;
    sliceSlider.value = 0;
    sliceSlider.disabled = true;
    renderEmptySliceGrid();
    setLoadingState(false);
    updateAxisButtons();
    updateStatus("Ready", "ready");
    statusSlice.textContent = "Slice: —";
}

// ═══════════════════════════════════════════════════════════
//  ELECTRON IPC HANDLERS
// ═══════════════════════════════════════════════════════════

// ── Volume Rendering Logic ──

async function fetchAndRenderVolume() {
    const id = state.results.scan_id;
    const modality = state.current3DModality;
    updateStatus(`Fetching 3D Volume (${modality.toUpperCase()})…`, "busy");

    try {
        // Fetch modality data
        const modResp = await fetch(`/volume-binary/${id}?modality=${modality}`);
        const modBuffer = await modResp.arrayBuffer();
        const modData = new Uint8Array(modBuffer);

        // Fetch mask data
        const maskResp = await fetch(`/volume-binary/${id}?modality=mask`);
        const maskBuffer = await maskResp.arrayBuffer();
        const maskData = new Uint8Array(maskBuffer);

        render3DVolume(modData, maskData);
        updateStatus("Ready", "ready");
    } catch (e) {
        console.error("Volume fetch failed:", e);
        updateStatus("Volume data error", "error");
    }
}

function render3DVolume(modData, maskData) {
    const container = document.getElementById("canvas-container");
    
    if (state.renderer) {
        if (state.animationId) cancelAnimationFrame(state.animationId);
        state.renderer.dispose();
        if (state.renderer.domElement.parentElement) {
            state.renderer.domElement.parentElement.removeChild(state.renderer.domElement);
        }
        state.renderer = null;
    }

    container.innerHTML = "";

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    state.renderer = renderer;
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(1.5, 1.2, 1.8);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    // Grid & Axes
    const grid = new THREE.GridHelper(2, 20, 0x333333, 0x111111);
    grid.position.y = -0.6;
    scene.add(grid);
    const axes = new THREE.AxesHelper(0.4);
    axes.position.set(-0.7, -0.5, -0.7);
    scene.add(axes);

    // Create 3D Texture
    // The volume dimensions are fixed by the backend: 240x240x155 (actually we should get them from results)
    const dims = state.results.slice_counts;
    const width = dims.sagittal; // x
    const height = dims.coronal; // y
    const depth = dims.axial;    // z

    const modTex = new THREE.DataTexture3D(modData, width, height, depth);
    modTex.format = THREE.RedFormat;
    modTex.type = THREE.UnsignedByteType;
    modTex.minFilter = THREE.LinearFilter;
    modTex.magFilter = THREE.LinearFilter;
    modTex.unpackAlignment = 1;
    modTex.needsUpdate = true;

    const maskTex = new THREE.DataTexture3D(maskData, width, height, depth);
    maskTex.format = THREE.RedFormat;
    maskTex.type = THREE.UnsignedByteType;
    maskTex.minFilter = THREE.LinearFilter;
    maskTex.magFilter = THREE.LinearFilter;
    maskTex.unpackAlignment = 1;
    maskTex.needsUpdate = true;

    // Simple multi-slice representation or Volume Shader
    // For "Professional" look without external shader files, we'll use a fast Raymarching shader
    const shader = getVolumeShader();
    const material = new THREE.ShaderMaterial({
        uniforms: {
            u_data: { value: modTex },
            u_mask: { value: maskTex },
            u_size: { value: new THREE.Vector3(width, height, depth) },
            u_threshold: { value: 0.15 },
            u_opacity: { value: 0.25 },
            u_range: { value: 0.1 }
        },
        vertexShader: shader.vertex,
        fragmentShader: shader.fragment,
        transparent: true,
        side: THREE.BackSide
    });

    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1.2, 1.2, 1.2), material);
    scene.add(mesh);

    const animate = () => {
        state.animationId = requestAnimationFrame(animate);
        controls.update();
        mesh.rotation.y += 0.002;
        renderer.render(scene, camera);
    };
    animate();

    const observer = new ResizeObserver(() => {
        renderer.setSize(container.clientWidth, container.clientHeight);
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
    });
    observer.observe(container);
}

function getVolumeShader() {
    return {
        vertex: `
            varying vec3 v_pos;
            varying vec3 v_cameraPos;
            void main() {
                v_pos = position + vec3(0.5);
                vec4 worldPos = modelMatrix * vec4(position, 1.0);
                v_cameraPos = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz + vec3(0.5);
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragment: `
            precision highp float;
            precision highp sampler3D;
            varying vec3 v_pos;
            varying vec3 v_cameraPos;
            uniform sampler3D u_data;
            uniform sampler3D u_mask;
            uniform vec3 u_size;
            uniform float u_threshold;
            uniform float u_opacity;

            void main() {
                vec3 rayDir = normalize(v_pos - v_cameraPos);
                vec3 currPos = v_pos;
                vec4 accum = vec4(0.0);
                
                // Step size based on depth
                float stepSize = 0.015; 
                int maxSteps = 80;
                
                for(int i=0; i < 80; i++) {
                    float val = texture(u_data, currPos).r;
                    float mask = texture(u_mask, currPos).r;
                    
                    // Basic transfer function
                    if(val > u_threshold) {
                        float alpha = (val - u_threshold) / (1.0 - u_threshold);
                        alpha = pow(alpha, 1.5) * u_opacity;
                        
                        vec3 color = vec3(val * 1.2); // Grayscale brain
                        if(mask > 0.4) {
                            color = vec3(1.0, 0.1, 0.1); // Bright red tumor
                            alpha = val * 0.95; // More opaque tumor
                        }
                        
                        // Alpha compositing
                        vec4 src = vec4(color, alpha);
                        accum.rgb += (1.0 - accum.a) * src.rgb * src.a;
                        accum.a += (1.0 - accum.a) * src.a;
                    }
                    
                    currPos += rayDir * stepSize;
                    
                    // Boundary check
                    if(currPos.x < 0.0 || currPos.x > 1.0 || 
                       currPos.y < 0.0 || currPos.y > 1.0 || 
                       currPos.z < 0.0 || currPos.z > 1.0 || 
                       accum.a >= 0.95) break;
                }
                
                if(accum.a < 0.01) discard;
                gl_FragColor = accum;
            }
        `
    };
}

if (window.electronAPI) {
    window.electronAPI.onResetWorkspace(() => resetWorkspace());
    window.electronAPI.onRunAnalysis(() => {
        const analyzeBtn = document.getElementById("tb-analyze");
        if (!analyzeBtn.disabled) runAnalysis();
    });
    window.electronAPI.onToggleSidebar(() => toggleSidebar());
    window.electronAPI.onToggleStatusBar(() => toggleStatusBar());
    window.electronAPI.onChangeAxis((axis) => {
        if (state.results) switchAxis(axis);
    });
    window.electronAPI.onOpenDocs(() => {
        alert("NeuroView MRI Documentation\n\nUpload T1C, T1N, T2F, and T2W NIfTI scans to begin analysis.\n\nUse the toolbar or File menu to load scans.\nView keyboard shortcuts via Help > Keyboard Shortcuts.");
    });
    window.electronAPI.onShowShortcuts(() => {
        document.getElementById("shortcuts-dialog").showModal();
    });
    window.electronAPI.onExportResults(() => {
        if (!state.results) {
            alert("No results to export. Run an analysis first.");
            return;
        }
        alert("Export functionality coming soon.");
    });
    window.electronAPI.onCopySlice(() => {
        alert("Copy slice functionality coming soon.");
    });
    window.electronAPI.onPreferences(() => {
        alert("Preferences panel coming soon.");
    });
    window.electronAPI.onToggleOverlay(() => {
        alert("Overlay toggle coming soon.");
    });
    window.electronAPI.onOpenFiles((filePaths) => {
        alert(`Selected ${filePaths.length} file(s).\nDrag-and-drop file assignment coming soon.\n\nFor now, use the sidebar upload slots.`);
    });
}

// ── Init ──
resetResultsUI();
syncUploadState();
