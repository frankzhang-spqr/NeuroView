const modalityMeta = {
    t1c: { label: "T1 Contrast", colorClass: "cyan", fullLabel: "T1 Contrast-Enhanced" },
    t1n: { label: "T1 Native", colorClass: "lilac", fullLabel: "T1 Native" },
    t2f: { label: "T2 FLAIR", colorClass: "green", fullLabel: "T2 FLAIR" },
    t2w: { label: "T2 Weighted", colorClass: "orange", fullLabel: "T2 Weighted" }
};

const state = {
    results: null,
    groupedSlices: [],
    selectedSliceIndex: 0
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

fileInputs.forEach((input) => {
    input.addEventListener("change", () => {
        syncUploadState();
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

        displayResults(results);
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
    state.groupedSlices = [];
    state.selectedSliceIndex = 0;
    resetResultsUI();
    syncUploadState();
});

document.getElementById("prev-slice").addEventListener("click", () => {
    if (!state.groupedSlices.length) {
        return;
    }
    state.selectedSliceIndex = Math.max(0, state.selectedSliceIndex - 1);
    renderSelectedSlice();
});

document.getElementById("next-slice").addEventListener("click", () => {
    if (!state.groupedSlices.length) {
        return;
    }
    state.selectedSliceIndex = Math.min(state.groupedSlices.length - 1, state.selectedSliceIndex + 1);
    renderSelectedSlice();
});

sliceSlider.addEventListener("input", (event) => {
    state.selectedSliceIndex = Number(event.target.value);
    renderSelectedSlice();
});

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

function init3DView(scanUrl) {
    const container = document.getElementById("canvas-container");
    container.innerHTML = "";

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
    camera.position.set(150, 150, 150);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1);
    scene.add(light);

    const loader = new AMI.VolumeLoader(container);
    loader.load(scanUrl)
        .then(() => {
            const series = loader.data[0].mergeSeries(loader.data)[0];
            loader.free();

            const stackHelper = new AMI.StackHelper(series);
            stackHelper.bbox.visible = true;
            stackHelper.border.color = 0x12aee2;
            scene.add(stackHelper);

            const worldBBox = stackHelper.stack.worldBoundingBox();
            const lpsDims = new THREE.Vector3(
                worldBBox[1] - worldBBox[0],
                worldBBox[3] - worldBBox[2],
                worldBBox[5] - worldBBox[4]
            );

            const bboxGeometry = new THREE.BoxGeometry(lpsDims.x, lpsDims.y, lpsDims.z);
            const bboxEdges = new THREE.EdgesGeometry(bboxGeometry);

            const outerBox = new THREE.LineSegments(
                bboxEdges,
                new THREE.LineBasicMaterial({ color: 0x00597f, transparent: true, opacity: 0.85 })
            );
            outerBox.position.copy(stackHelper.stack.worldCenter());
            scene.add(outerBox);

            const innerBox = new THREE.LineSegments(
                bboxEdges,
                new THREE.LineBasicMaterial({ color: 0xff2936, transparent: true, opacity: 0.9 })
            );
            innerBox.scale.set(0.58, 0.58, 0.58);
            innerBox.position.copy(stackHelper.stack.worldCenter());
            scene.add(innerBox);

            const center = stackHelper.stack.worldCenter();
            const maxDim = Math.max(lpsDims.x, lpsDims.y, lpsDims.z);
            camera.position.set(center.x + maxDim * 1.2, center.y + maxDim * 0.8, center.z + maxDim * 1.3);
            camera.lookAt(center);
            controls.target.copy(center);

            const animate = () => {
                controls.update();
                renderer.render(scene, camera);
                requestAnimationFrame(animate);
            };
            animate();

            window.addEventListener("resize", () => {
                const width = container.clientWidth;
                const height = container.clientHeight;
                renderer.setSize(width, height);
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
            });
        })
        .catch((error) => {
            console.error("Error loading NIfTI file:", error);
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none">
                            <path d="m12 2 8 4.5v11L12 22 4 17.5v-11L12 2Zm0 0v9.5m8-5-8 5m0 0-8-5" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <h3>3D View Unavailable</h3>
                    <p>The scan loaded for inference, but the browser could not render the interactive volume.</p>
                </div>
            `;
        });
}

function displayResults(results) {
    state.results = results;
    state.groupedSlices = groupSlicesByIndex(results.slice_data);
    state.selectedSliceIndex = getInitialSliceIndex(results.slices_with_tumor);

    renderSummary(results);
    renderTumorSlices(results.slices_with_tumor);
    renderSliceStack(state.groupedSlices, results.slices_with_tumor);
    configureSliceControls();
    renderSelectedSlice();

    if (results.has_tumor && results.scan_url) {
        init3DView(results.scan_url);
    } else {
        document.getElementById("canvas-container").innerHTML = `
            <div class="empty-state">
                <div class="empty-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none">
                        <path d="m12 2 8 4.5v11L12 22 4 17.5v-11L12 2Zm0 0v9.5m8-5-8 5m0 0-8-5" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/>
                    </svg>
                </div>
                <h3>No Tumor Volume</h3>
                <p>The uploaded scan completed analysis without a detected tumor volume.</p>
            </div>
        `;
    }

    setLoadingState(false);
}

function groupSlicesByIndex(sliceData) {
    const grouped = {};

    sliceData.forEach((slice) => {
        if (!grouped[slice.slice_index]) {
            grouped[slice.slice_index] = {};
        }
        grouped[slice.slice_index][slice.modality] = slice;
    });

    return Object.keys(grouped)
        .map((key) => ({
            index: Number(key),
            modalities: grouped[key]
        }))
        .sort((a, b) => a.index - b.index);
}

function getInitialSliceIndex(tumorSlices) {
    if (!tumorSlices || !tumorSlices.length) {
        return 0;
    }

    const firstTumorSlice = tumorSlices[0];
    const groupedIndex = state.groupedSlices.findIndex((group) => group.index === firstTumorSlice);
    return groupedIndex >= 0 ? groupedIndex : 0;
}

function renderSummary(results) {
    const affectedCount = results.slices_with_tumor.length;

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

function renderTumorSlices(tumorSlices) {
    if (!tumorSlices.length) {
        tumorSlicesContainer.innerHTML = '<span class="empty-pill">No tumor slices</span>';
        return;
    }

    tumorSlicesContainer.innerHTML = "";
    tumorSlices.forEach((sliceIndex, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `tumor-pill${index === 0 ? " active" : ""}`;
        button.textContent = sliceIndex;
        button.addEventListener("click", () => {
            const newIndex = state.groupedSlices.findIndex((group) => group.index === sliceIndex);
            if (newIndex >= 0) {
                state.selectedSliceIndex = newIndex;
                renderSelectedSlice();
            }
        });
        tumorSlicesContainer.appendChild(button);
    });
}

function renderSliceStack(groupedSlices, tumorSlices) {
    const highlighted = groupedSlices.filter((group) => tumorSlices.includes(group.index)).slice(0, 5);
    const source = highlighted.length ? highlighted : groupedSlices.slice(0, 3);

    if (!source.length) {
        sliceStack.innerHTML = `
            <div class="stack-card empty">
                <p>Slice cards will appear here after analysis.</p>
            </div>
        `;
        return;
    }

    sliceStack.innerHTML = source.map((group) => {
        const previews = Object.keys(modalityMeta).map((modality) => {
            const slice = group.modalities[modality];
            return `
                <div class="stack-preview">
                    <div class="stack-preview-frame">
                        ${slice ? `<img src="data:image/png;base64,${slice.image}" alt="${modalityMeta[modality].fullLabel} slice ${group.index}">` : ""}
                    </div>
                    <div class="stack-preview-label ${modalityMeta[modality].colorClass}">${modality.toUpperCase()}</div>
                </div>
            `;
        }).join("");

        return `
            <article class="stack-card">
                <div class="stack-header">
                    <h3>Slice ${group.index}</h3>
                    ${tumorSlices.includes(group.index) ? '<span class="tumor-tag">TUMOR</span>' : ""}
                </div>
                <div class="stack-preview-grid">${previews}</div>
            </article>
        `;
    }).join("");
}

function configureSliceControls() {
    const maxValue = Math.max(0, state.groupedSlices.length - 1);
    sliceSlider.max = maxValue;
    sliceSlider.disabled = !state.groupedSlices.length;
    sliceSlider.value = state.selectedSliceIndex;
}

function renderSelectedSlice() {
    if (!state.groupedSlices.length) {
        renderEmptySliceGrid();
        return;
    }

    const group = state.groupedSlices[state.selectedSliceIndex];
    const tumorSlices = state.results.slices_with_tumor;

    sliceGrid.innerHTML = Object.keys(modalityMeta).map((modality) => {
        const slice = group.modalities[modality];
        const meta = modalityMeta[modality];
        const hasTumor = Boolean(slice && slice.has_tumor);
        const bboxMarkup = slice && slice.bbox ? createBBoxMarkup(slice.bbox) : "";
        const imageMarkup = slice
            ? `<div class="slice-image-wrap"><img src="data:image/png;base64,${slice.image}" alt="${meta.fullLabel} slice ${group.index}">${bboxMarkup}</div>`
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

    sliceSlider.value = state.selectedSliceIndex;
    sliceCounter.textContent = `${state.selectedSliceIndex + 1} / ${state.groupedSlices.length}`;
    sliceCounterStatus.textContent = tumorSlices.includes(group.index) ? "TUMOR" : "CLEAR";

    document.querySelectorAll(".tumor-pill").forEach((pill) => {
        pill.classList.toggle("active", Number(pill.textContent) === group.index);
    });
}

function createBBoxMarkup(bbox) {
    const [x, y, width, height] = bbox;
    return `<div class="bbox" style="left:${x}px; top:${y}px; width:${width}px; height:${height}px;"></div>`;
}

function renderEmptySliceGrid() {
    sliceGrid.innerHTML = Object.keys(modalityMeta).map((modality) => {
        const meta = modalityMeta[modality];
        return `
            <div class="slice-panel placeholder">
                <span class="slice-label ${meta.colorClass}">${meta.fullLabel}</span>
                <div class="slice-frame">
                    <p>Awaiting data</p>
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
}

resetResultsUI();
syncUploadState();
