document.getElementById('upload-button').addEventListener('click', async () => {
    const fileInputs = document.querySelectorAll('.file-input');
    const formData = new FormData();
    let fileCount = 0;

    fileInputs.forEach(input => {
        if (input.files.length > 0) {
            formData.append(input.id, input.files[0]);
            fileCount++;
        }
    });

    if (fileCount < 4) {
        alert('Please upload all four modality files.');
        return;
    }

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        const results = await response.json();
        if (results.error) {
            alert(`An error occurred on the server: ${results.error}`);
            return;
        }
        displayResults(results);
    } else {
        alert('Error processing the file.');
    }
});

function init3DView(scanUrl) {
    const container = document.getElementById('canvas-container');
    container.innerHTML = '';

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
        45,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.x = 150;
    camera.position.y = 150;
    camera.position.z = 150;

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1);
    scene.add(light);

    const loader = new AMI.VolumeLoader(container);
    loader.load(scanUrl)
        .then(() => {
            const series = loader.data[0].mergeSeries(loader.data)[0];
            loader.free();

            const stackHelper = new AMI.StackHelper(series);
            stackHelper.bbox.visible = false;
            stackHelper.border.color = 0x0F9F00;
            
            scene.add(stackHelper);

            // center camera and controls
            const center = stackHelper.stack.worldCenter();
            camera.position.x = center.x;
            camera.position.y = center.y;
            camera.position.z = center.z + 150;
            camera.lookAt(center);
            controls.target.copy(center);

            const animate = () => {
                controls.update();
                renderer.render(scene, camera);
                requestAnimationFrame(animate);
            };
            animate();
        })
        .catch(error => {
            console.error('Error loading NIfTI file:', error);
            container.innerHTML = '<p>Error loading 3D model.</p>';
        });
}

function displayResults(results) {
    const resultsSection = document.getElementById('results-section');
    const tumorInfo = document.getElementById('tumor-info');
    const sliceContainer = document.getElementById('slice-container');

    resultsSection.style.display = 'block';
    sliceContainer.innerHTML = ''; // Clear previous results

    if (results.has_tumor) {
        tumorInfo.innerHTML = `
            <p><strong>Tumor Detected:</strong> Yes</p>
            <p><strong>Tumor Type:</strong> ${results.tumor_type}</p>
            <p><strong>Tumor appears on slices:</strong> ${results.slices_with_tumor.join(', ')}</p>
        `;
        init3DView(results.scan_url);
    } else {
        tumorInfo.innerHTML = `<p><strong>Tumor Detected:</strong> No</p>`;
        document.getElementById('canvas-container').innerHTML = '<p>No 3D model to display.</p>';
    }

    // Group slices by index
    const slicesByIndex = {};
    results.slice_data.forEach(slice => {
        if (!slicesByIndex[slice.slice_index]) {
            slicesByIndex[slice.slice_index] = [];
        }
        slicesByIndex[slice.slice_index].push(slice);
    });

    for (const index in slicesByIndex) {
        const sliceGroup = document.createElement('div');
        sliceGroup.className = 'slice-group';
        
        const sliceLabel = document.createElement('p');
        sliceLabel.textContent = `Slice ${index}`;
        sliceGroup.appendChild(sliceLabel);

        const imageContainer = document.createElement('div');
        imageContainer.className = 'image-container';
        sliceGroup.appendChild(imageContainer);

        slicesByIndex[index].forEach(slice => {
            const sliceDiv = document.createElement('div');
            sliceDiv.className = 'slice';
            if (slice.has_tumor) {
                sliceDiv.style.borderColor = 'red';
            }

            const img = document.createElement('img');
            img.src = `data:image/png;base64,${slice.image}`;
            sliceDiv.appendChild(img);

            if (slice.bbox) {
                const bboxDiv = document.createElement('div');
                bboxDiv.className = 'bbox';
                const scale = 150 / 240; // image size / original size
                bboxDiv.style.left = `${slice.bbox[0] * scale}px`;
                bboxDiv.style.top = `${slice.bbox[1] * scale}px`;
                bboxDiv.style.width = `${slice.bbox[2] * scale}px`;
                bboxDiv.style.height = `${slice.bbox[3] * scale}px`;
                sliceDiv.appendChild(bboxDiv);
            }
            
            const modalityLabel = document.createElement('p');
            modalityLabel.textContent = slice.modality;
            sliceDiv.appendChild(modalityLabel);

            imageContainer.appendChild(sliceDiv);
        });
        sliceContainer.appendChild(sliceGroup);
    }
}