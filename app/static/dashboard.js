/* global THREE */

const modalityMeta = {
    t1c: { label: "T1 Contrast", colorClass: "cyan", fullLabel: "T1 Contrast-Enhanced" },
    t1n: { label: "T1 Native", colorClass: "lilac", fullLabel: "T1 Native" },
    t2f: { label: "T2 FLAIR", colorClass: "green", fullLabel: "T2 FLAIR" },
    t2w: { label: "T2 Weighted", colorClass: "orange", fullLabel: "T2 Weighted" }
};

const axisLabels = {
    axial: "AXI",
    coronal: "COR",
    sagittal: "SAG"
};

const state = {
    results: null,
    currentAxis: "axial",
    selectedSliceIndex: 0,
    currentSlicePreview: null,
    stackItems: []
};

const fileInputs = Array.from(document.querySelectorAll(".file-input"));
const uploadButton = document.getElementById("upload-button");
const fileCount = document.getElementById("file-count");
const sliceGrid = document.getElementById("slice-grid");
const sliceStack = document.getElementById("slice-stack");
const tumorSlicesContainer = document.getElementById("tumor-slices");
const tumorSummary = document.getElementById("tumor-summary");
const sliceSlider = document.getElementById("slice-slider");
const sliceCounter = document.getElementById("slice-counter");
const sliceCounterStatus = document.getElementById("slice-counter-status");
const axisButtons = Array.from(document.querySelectorAll(".axis-pill"));

fileInputs.forEach((input) => {
    input.addEventListener("change", syncUploadState);
});

axisButtons.forEach((button) => {
    button.addEventListener("click", async () => {
        if (!state.results) {
            return;
        }
        await switchAxis(button.dataset.axis);
    });
});

document.getElementById("upload-button").addEventListener("click", async () => {
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
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Error processing the file.");
        }

        const results = await response.json();
        if (results.error) {
            throw new Error(results.error);
        }

        await displayResults(results);
    } catch (error) {
        alert(`Analysis failed: ${error.message}`);
        setLoadingState(false);
    }
});

document.getElementById("reset-dashboard").addEventListener("click", () => {
    fileInputs.forEach((input) => {
        input.value = "";
    });
    state.results = null;
    state.currentAxis = "axial";
    state.selectedSliceIndex = 0;
    state.currentSlicePreview = null;
    state.stackItems = [];
    resetResultsUI();
    syncUploadState();
});

document.getElementById("prev-slice").addEventListener("click", async () => {
    if (!state.results) {
        return;
    }
    state.selectedSliceIndex = Math.max(0, state.selectedSliceIndex - 1);
    await refreshCurrentSlice();
});

document.getElementById("next-slice").addEventListener("click", async () => {
    if (!state.results) {
        return;
    }
    const maxIndex = state.results.slice_counts[state.currentAxis] - 1;
    state.selectedSliceIndex = Math.min(maxIndex, state.selectedSliceIndex + 1);
    await refreshCurrentSlice();
});

sliceSlider.addEventListener("input", async (event) => {
    state.selectedSliceIndex = Number(event.target.value);
    await refreshCurrentSlice();
});

async function displayResults(results) {
    state.results = results;
    state.currentAxis = "axial";
    state.selectedSliceIndex = getInitialSliceIndex("axial");

    renderSummary(results);
    renderVolume(results.volume_points);
    updateAxisButtons();
    configureSliceControls();
    renderTumorSlices();
    await refreshCurrentSlice();
    await refreshSliceStack();
    setLoadingState(false);
}

function syncUploadState() {
    let count = 0;

    fileInputs.forEach((input) => {
        const modality = input.id.replace("_file", "");
        const nameNode = document.getElementById(`${modality}_name`);
        const card = input.closest(".modality-card");
        const hasFile = input.files.length > 0;

        if (hasFile) {
            count += 1;
            nameNode.textContent = input.files[0].name;
            card.classList.add("has-file");
        } else {
            nameNode.textContent = "Choose NIfTI file";
            card.classList.remove("has-file");
        }
    });

    fileCount.textContent = count;
    uploadButton.disabled = count < 4;
}

function setLoadingState(isLoading) {
    uploadButton.disabled = isLoading || Number(fileCount.textContent) < 4;
    uploadButton.textContent = isLoading ? "Analyzing..." : "Analyze Scan";
}

function getInitialSliceIndex(axis) {
    const tumorSlices = state.results?.tumor_slices_by_axis?.[axis] || [];
    return tumorSlices.length ? tumorSlices[0] : 0;
}

function updateAxisButtons() {
    axisButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.axis === state.currentAxis);
    });
}

function configureSliceControls() {
    const maxValue = Math.max(0, (state.results?.slice_counts?.[state.currentAxis] || 1) - 1);
    sliceSlider.max = maxValue;
    sliceSlider.disabled = !state.results;
    sliceSlider.value = state.selectedSliceIndex;
}

async function switchAxis(axis) {
    if (axis === state.currentAxis) {
        return;
    }

    state.currentAxis = axis;
    const maxIndex = state.results.slice_counts[axis] - 1;
    const tumorSlices = state.results.tumor_slices_by_axis[axis] || [];
    state.selectedSliceIndex = tumorSlices.length ? tumorSlices[0] : Math.min(state.selectedSliceIndex, maxIndex);

    updateAxisButtons();
    configureSliceControls();
    renderTumorSlices();
    await refreshCurrentSlice();
    await refreshSliceStack();
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

async function refreshSliceStack() {
    if (!state.results) {
        return;
    }

    sliceStack.innerHTML = `
        <div class="stack-card empty">
            <p>Loading affected ${axisLabels[state.currentAxis]} slices...</p>
        </div>
    `;

    const response = await fetch(`/slice-stack/${state.results.scan_id}?axis=${state.currentAxis}`);
    const payload = await response.json();
    if (payload.error) {
        sliceStack.innerHTML = `
            <div class="stack-card empty">
                <p>${payload.error}</p>
            </div>
        `;
        return;
    }

    state.stackItems = payload.items || [];
    renderSliceStack();
}

function renderSummary(results) {
    const affectedCount = results.tumor_slices_by_axis.axial.length;

    if (results.has_tumor) {
        tumorSummary.className = "summary-card positive";
        tumorSummary.innerHTML = `
            <div class="summary-icon" aria-hidden="true">!</div>
            <div>
                <strong>Tumor Detected</strong>
                <p>Type: <b>${results.tumor_type}</b></p>
                <p>Affected slices: ${affectedCount}</p>
            </div>
        `;
        return;
    }

    tumorSummary.className = "summary-card negative";
    tumorSummary.innerHTML = `
        <div class="summary-icon" aria-hidden="true">&#10003;</div>
        <div>
            <strong>No Tumor Detected</strong>
            <p>No suspicious slice regions were identified in this scan set.</p>
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
    if (!preview) {
        renderEmptySliceGrid();
        return;
    }

    const slicesByModality = {};
    preview.slices.forEach((item) => {
        slicesByModality[item.modality] = item;
    });

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
                <span class="slice-label ${meta.colorClass}">${meta.fullLabel}</span>
                ${hasTumor ? '<span class="tumor-tag">TUMOR</span>' : ""}
                <div class="slice-frame">
                    ${imageMarkup}
                </div>
            </div>
        `;
    }).join("");

    sliceSlider.value = preview.slice_index;
    sliceCounter.textContent = `${preview.slice_index} / ${state.results.slice_counts[state.currentAxis] - 1}`;
    sliceCounterStatus.textContent = preview.has_tumor ? "TUMOR" : "CLEAR";

    document.querySelectorAll(".tumor-pill").forEach((pill) => {
        pill.classList.toggle("active", Number(pill.textContent) === preview.slice_index);
    });

    document.querySelectorAll(".stack-card").forEach((card) => {
        card.classList.toggle("active", Number(card.dataset.sliceIndex) === preview.slice_index);
    });
}

function renderSliceStack() {
    if (!state.stackItems.length) {
        sliceStack.innerHTML = `
            <div class="stack-card empty">
                <p>No affected ${axisLabels[state.currentAxis]} slices for this scan.</p>
            </div>
        `;
        return;
    }

    sliceStack.innerHTML = state.stackItems.map((item) => {
        const previews = item.slices.map((slice) => `
            <button class="stack-preview" type="button" data-slice-index="${item.slice_index}">
                <div class="stack-preview-frame">
                    <img src="data:image/png;base64,${slice.image}" alt="${slice.modality.toUpperCase()} slice ${item.slice_index}">
                </div>
                <div class="stack-preview-label ${modalityMeta[slice.modality].colorClass}">${slice.modality.toUpperCase()}</div>
            </button>
        `).join("");

        return `
            <article class="stack-card${item.slice_index === state.selectedSliceIndex ? " active" : ""}" data-slice-index="${item.slice_index}">
                <div class="stack-header">
                    <h3>Slice ${item.slice_index}</h3>
                    ${item.has_tumor ? '<span class="tumor-tag">TUMOR</span>' : ""}
                </div>
                <div class="stack-preview-grid">${previews}</div>
            </article>
        `;
    }).join("");

    sliceStack.querySelectorAll("[data-slice-index]").forEach((node) => {
        node.addEventListener("click", async () => {
            state.selectedSliceIndex = Number(node.dataset.sliceIndex);
            configureSliceControls();
            await refreshCurrentSlice();
        });
    });
}

function renderVolume(volumePoints) {
    const container = document.getElementById("canvas-container");
    container.innerHTML = "";

    const brainPoints = volumePoints?.brain || [];
    const tumorPoints = volumePoints?.tumor || [];
    if (!brainPoints.length && !tumorPoints.length) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none">
                        <path d="m12 2 8 4.5v11L12 22 4 17.5v-11L12 2Zm0 0v9.5m8-5-8 5m0 0-8-5" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
                    </svg>
                </div>
                <h3>Volume Preview Unavailable</h3>
                <p>The scan finished, but there was not enough volumetric data to draw the preview.</p>
            </div>
        `;
        return;
    }

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
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

    addPointCloud(scene, brainPoints, 0xa8aaad, 0.018, 0.18);
    addPointCloud(scene, tumorPoints, 0xff545e, 0.03, 0.92);
    addBoundingBoxes(scene);

    const animate = () => {
        controls?.update();
        scene.rotation.y += 0.0015;
        renderer.render(scene, camera);
        requestAnimationFrame(animate);
    };
    animate();

    window.onresize = () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    };
}

function addPointCloud(scene, points, color, size, opacity) {
    if (!points.length) {
        return;
    }

    const geometry = new THREE.BufferGeometry();
    const attribute = new THREE.BufferAttribute(new Float32Array(points.flat()), 3);
    if (typeof geometry.setAttribute === "function") {
        geometry.setAttribute("position", attribute);
    } else {
        geometry.addAttribute("position", attribute);
    }
    const material = new THREE.PointsMaterial({
        color,
        size,
        transparent: true,
        opacity,
        sizeAttenuation: true
    });
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
    return `<div class="bbox" style="left:${x}%; top:${y}%; width:${width}%; height:${height}%;"></div>`;
}

function renderEmptySliceGrid(message = "Awaiting data") {
    sliceGrid.innerHTML = Object.keys(modalityMeta).map((modality) => {
        const meta = modalityMeta[modality];
        return `
            <div class="slice-panel placeholder">
                <span class="slice-label ${meta.colorClass}">${meta.fullLabel}</span>
                <div class="slice-frame">
                    <p>${message}</p>
                </div>
            </div>
        `;
    }).join("");

    sliceCounter.textContent = "0 / 0";
    sliceCounterStatus.textContent = "NO DATA";
}

function resetResultsUI() {
    tumorSummary.className = "summary-card neutral";
    tumorSummary.innerHTML = `
        <div class="summary-icon" aria-hidden="true">!</div>
        <div>
            <strong>Awaiting Analysis</strong>
            <p>Upload all modalities to run detection and populate the review panels.</p>
        </div>
    `;
    tumorSlicesContainer.innerHTML = '<span class="empty-pill">No slices yet</span>';
    sliceStack.innerHTML = `
        <div class="stack-card empty">
            <p>Slice cards will appear here after analysis.</p>
        </div>
    `;
    document.getElementById("canvas-container").innerHTML = `
        <div class="empty-state">
            <div class="empty-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none">
                    <path d="m12 2 8 4.5v11L12 22 4 17.5v-11L12 2Zm0 0v9.5m8-5-8 5m0 0-8-5" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
                </svg>
            </div>
            <h3>Upload MRI Scans</h3>
            <p>Complete the four modality inputs to generate the volume and slice review workspace.</p>
        </div>
    `;
    sliceSlider.max = 0;
    sliceSlider.value = 0;
    sliceSlider.disabled = true;
    renderEmptySliceGrid();
    setLoadingState(false);
    updateAxisButtons();
}

resetResultsUI();
syncUploadState();
