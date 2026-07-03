document.addEventListener('DOMContentLoaded', () => {
    // --- State Variables ---
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let activeTab = 'canvas-tab';
    
    // --- DOM Elements ---
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const datasetSelect = document.getElementById('active-dataset');
    const trainingAlert = document.getElementById('training-alert');
    const datasetPlaceholder = trainingAlert ? trainingAlert.querySelector('.dataset-placeholder') : null;
    
    // Audio Toggle
    const audioToggle = document.getElementById('audio-feedback-toggle');
    
    // Canvas Tab Elements
    const canvas = document.getElementById('drawing-pad');
    const ctx = canvas.getContext('2d');
    const btnClear = document.getElementById('btn-clear');
    const btnPredictCanvas = document.getElementById('btn-predict-canvas');
    const realtimeMode = document.getElementById('realtime-mode');
    const canvasPredChar = document.getElementById('canvas-pred-char');
    const canvasPredLabel = document.getElementById('canvas-pred-label');
    const canvasPredScore = document.getElementById('canvas-pred-score');
    const canvasTop5 = document.getElementById('canvas-top5-container');
    const canvasScanner = document.getElementById('canvas-scanner');
    
    // Upload Tab Elements
    const dropzone = document.getElementById('file-dropzone');
    const fileInput = document.getElementById('file-input');
    const fileBadge = document.getElementById('file-name-badge');
    const uploadPreviewWrapper = document.getElementById('upload-preview-wrapper');
    const uploadPreview = document.getElementById('upload-preview');
    const uploadPredChar = document.getElementById('upload-pred-char');
    const uploadPredLabel = document.getElementById('upload-pred-label');
    const uploadPredScore = document.getElementById('upload-pred-score');
    const uploadTop5 = document.getElementById('upload-top5-container');
    const uploadScanner = document.getElementById('upload-scanner');
    
    // OCR Tab Elements
    const ocrCanvas = document.getElementById('ocr-pad');
    const ocrCtx = ocrCanvas.getContext('2d');
    const btnOcrClear = document.getElementById('btn-ocr-clear');
    const btnOcrPredict = document.getElementById('btn-ocr-predict');
    const ocrTranscription = document.getElementById('ocr-transcription');
    const ocrSegmentsList = document.getElementById('ocr-segments-list');
    const ocrScanner = document.getElementById('ocr-scanner');
    
    // EDA Tab Elements
    const graphSubToggles = document.querySelectorAll('.sub-toggle');
    const graphCurvesPanel = document.getElementById('graph-curves-panel');
    const graphCmPanel = document.getElementById('graph-cm-panel');
    const imgAccCurve = document.getElementById('img-acc-curve');
    const imgLossCurve = document.getElementById('img-loss-curve');
    const imgCmHeatmap = document.getElementById('img-cm-heatmap');

    // HUD Elements
    const diagLatency = document.getElementById('diag-latency');
    const diagFps = document.getElementById('diag-fps');
    const diagModel = document.getElementById('diag-model');
    const diagBarFill = document.getElementById('diag-bar-fill');

    // Interactive CNN layers
    const cnnLayers = document.querySelectorAll('.layer-3d');
    const cnnInfoTitle = document.getElementById('cnn-info-title');
    const cnnInfoDesc = document.getElementById('cnn-info-desc');

    // --- Web Audio Synthesizer (Zero asset SFX) ---
    const playSynthSound = (type, freqStart, freqEnd, duration, gainStart) => {
        if (!audioToggle || !audioToggle.checked) return;
        try {
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();
            
            osc.type = type;
            osc.frequency.setValueAtTime(freqStart, audioCtx.currentTime);
            if (freqEnd !== freqStart) {
                osc.frequency.exponentialRampToValueAtTime(freqEnd, audioCtx.currentTime + duration);
            }
            
            gainNode.gain.setValueAtTime(gainStart, audioCtx.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
            
            osc.connect(gainNode);
            gainNode.connect(audioCtx.destination);
            
            osc.start();
            osc.stop(audioCtx.currentTime + duration);
        } catch (e) {
            console.error('Audio synthesis failed', e);
        }
    };

    const playHoverSound = () => playSynthSound('sine', 1600, 2400, 0.06, 0.015);
    const playClickSound = () => playSynthSound('triangle', 800, 1200, 0.1, 0.05);
    const playClearSound = () => playSynthSound('sine', 600, 150, 0.35, 0.04);
    const playSuccessSound = () => {
        if (!audioToggle || !audioToggle.checked) return;
        try {
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const now = audioCtx.currentTime;
            const freqs = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6 (Arpeggio)
            freqs.forEach((f, idx) => {
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(f, now + idx * 0.05);
                gain.gain.setValueAtTime(0.02, now + idx * 0.05);
                gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.05 + 0.3);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start(now + idx * 0.05);
                osc.stop(now + idx * 0.05 + 0.3);
            });
        } catch (e) {}
    };

    // Bind UI sound triggers
    navButtons.forEach(btn => {
        btn.addEventListener('mouseenter', playHoverSound);
        btn.addEventListener('click', playClickSound);
    });
    document.querySelectorAll('.btn, .custom-select, .sub-toggle, .sample-box, .switch').forEach(el => {
        el.addEventListener('mouseenter', playHoverSound);
        el.addEventListener('click', playClickSound);
    });

    // --- 3D Mouse Tilt Card Handler ---
    const init3DTilt = () => {
        const tiltCards = document.querySelectorAll('.tilt-card');
        tiltCards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                // Relative offsets from center
                const xc = rect.width / 2;
                const yc = rect.height / 2;
                
                // Limit maximum rotation angle to +/- 10 degrees
                const rotateX = -((y - yc) / yc) * 8; 
                const rotateY = ((x - xc) / xc) * 8;
                
                card.style.setProperty('--rx', rotateX);
                card.style.setProperty('--ry', rotateY);
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.setProperty('--rx', 0);
                card.style.setProperty('--ry', 0);
            });
        });
    };
    init3DTilt();

    // --- Interactive CNN Architecture stack logic ---
    cnnLayers.forEach(layer => {
        layer.addEventListener('mouseenter', () => {
            cnnLayers.forEach(l => l.classList.remove('active-layer'));
            layer.classList.add('active-layer');
            
            const title = layer.getAttribute('data-layer-title');
            const desc = layer.getAttribute('data-layer-desc');
            
            if (cnnInfoTitle && cnnInfoDesc) {
                cnnInfoTitle.innerText = title.toUpperCase();
                cnnInfoDesc.innerText = desc;
            }
            playHoverSound();
        });
    });

    // --- Three.js Background Particle Net ---
    let scene, camera, renderer, particleSystem, particleLines;
    const initThreeBackground = () => {
        const bgCanvas = document.getElementById('neural-bg-canvas');
        if (!bgCanvas) return;

        const width = window.innerWidth;
        const height = window.innerHeight;

        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(60, width / height, 1, 1000);
        camera.position.z = 250;

        renderer = new THREE.WebGLRenderer({ canvas: bgCanvas, alpha: true, antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Particle Count & Buffers
        const count = 180;
        const particlesGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        const velocities = new Float32Array(count * 3);

        for (let i = 0; i < count * 3; i += 3) {
            // Distribute particles in 3D box
            positions[i] = (Math.random() - 0.5) * 500;
            positions[i + 1] = (Math.random() - 0.5) * 500;
            positions[i + 2] = (Math.random() - 0.5) * 200;

            velocities[i] = (Math.random() - 0.5) * 0.4;     // vx
            velocities[i + 1] = (Math.random() - 0.5) * 0.4; // vy
            velocities[i + 2] = (Math.random() - 0.5) * 0.2; // vz
        }

        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Shaders/Material for glowing points
        const particleMaterial = new THREE.PointsMaterial({
            color: 0x00f2fe,
            size: 3,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });

        particleSystem = new THREE.Points(particlesGeometry, particleMaterial);
        scene.add(particleSystem);

        // Lines linking near points
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x7c3aed,
            transparent: true,
            opacity: 0.12,
            blending: THREE.AdditiveBlending
        });

        const lineGeometry = new THREE.BufferGeometry();
        particleLines = new THREE.LineSegments(lineGeometry, lineMaterial);
        scene.add(particleLines);

        // Mouse pointer coordinates projection
        let mouseX = 0, mouseY = 0;
        let targetMouseX = 0, targetMouseY = 0;

        window.addEventListener('mousemove', (e) => {
            targetMouseX = (e.clientX - width / 2) * 0.3;
            targetMouseY = -(e.clientY - height / 2) * 0.3;
        });

        // FPS tracking
        let lastTime = performance.now();
        let frameCount = 0;

        // Animation Loop
        const animate = () => {
            requestAnimationFrame(animate);

            // Calculate FPS
            frameCount++;
            const time = performance.now();
            if (time >= lastTime + 1000) {
                if (diagFps) {
                    diagFps.innerText = `${Math.round((frameCount * 1000) / (time - lastTime))} FPS`;
                }
                frameCount = 0;
                lastTime = time;
            }

            // Smooth mouse tracking
            mouseX += (targetMouseX - mouseX) * 0.05;
            mouseY += (targetMouseY - mouseY) * 0.05;

            // Rotate camera slightly
            camera.position.x = mouseX * 0.5;
            camera.position.y = mouseY * 0.5;
            camera.lookAt(scene.position);

            const posArr = particleSystem.geometry.attributes.position.array;
            
            // Move particles & compute line segments
            const linePositions = [];
            const maxDistance = 65;

            for (let i = 0; i < count; i++) {
                const idx = i * 3;
                
                // Update pos
                posArr[idx] += velocities[idx];
                posArr[idx + 1] += velocities[idx + 1];
                posArr[idx + 2] += velocities[idx + 2];

                // Boundaries bounce check
                if (Math.abs(posArr[idx]) > 250) velocities[idx] *= -1;
                if (Math.abs(posArr[idx + 1]) > 250) velocities[idx + 1] *= -1;
                if (Math.abs(posArr[idx + 2]) > 100) velocities[idx + 2] *= -1;

                // Mouse attraction logic if mouse is close
                const dx = mouseX - posArr[idx];
                const dy = mouseY - posArr[idx + 1];
                const distToMouse = Math.sqrt(dx * dx + dy * dy);
                if (distToMouse < 90) {
                    posArr[idx] += dx * 0.005;
                    posArr[idx + 1] += dy * 0.005;
                }

                // Check connections to other points
                for (let j = i + 1; j < count; j++) {
                    const jdx = j * 3;
                    const lx = posArr[idx] - posArr[jdx];
                    const ly = posArr[idx + 1] - posArr[jdx + 1];
                    const lz = posArr[idx + 2] - posArr[jdx + 2];
                    const distance = Math.sqrt(lx * lx + ly * ly + lz * lz);

                    if (distance < maxDistance) {
                        linePositions.push(posArr[idx], posArr[idx + 1], posArr[idx + 2]);
                        linePositions.push(posArr[jdx], posArr[jdx + 1], posArr[jdx + 2]);
                    }
                }
            }

            particleSystem.geometry.attributes.position.needsUpdate = true;

            // Rebuild connect lines geometry
            particleLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
            particleLines.geometry.attributes.position.needsUpdate = true;

            // Animate HUD system diag bar with small fluctuating load patterns
            if (diagBarFill) {
                const currentWidth = parseFloat(diagBarFill.style.width) || 15;
                const fluctuate = (Math.random() - 0.5) * 3;
                diagBarFill.style.width = `${Math.min(Math.max(currentWidth + fluctuate, 10), 40)}%`;
            }

            renderer.render(scene, camera);
        };

        animate();

        // Screen Resize event
        window.addEventListener('resize', () => {
            const w = window.innerWidth;
            const h = window.innerHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        });
    };
    initThreeBackground();

    // --- Initialize Canvases ---
    function initCanvas(c, context, brushWidth) {
        // Set canvas background to solid black
        context.fillStyle = '#000000';
        context.fillRect(0, 0, c.width, c.height);
        
        // Configure brush style
        context.strokeStyle = '#ffffff';
        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.lineWidth = brushWidth;
    }
    
    if (canvas) initCanvas(canvas, ctx, 16);
    if (ocrCanvas) initCanvas(ocrCanvas, ocrCtx, 10);

    // --- Tab Navigation ---
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            navButtons.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            activeTab = btn.getAttribute('data-tab');
            const targetPanel = document.getElementById(activeTab);
            if (targetPanel) targetPanel.classList.add('active');
        });
    });

    // --- Active Pipeline Select Change ---
    if (datasetSelect) {
        datasetSelect.addEventListener('change', () => {
            const dataset = datasetSelect.value;
            if (diagModel) diagModel.innerText = dataset.toUpperCase();
            // Reset prediction displays
            resetPredictionUI();
            // Update EDA graphs
            updateEdaGraphs(dataset);
            // Auto check model availability
            checkModelStatus();
        });
    }

    function resetPredictionUI() {
        if (canvasPredChar) canvasPredChar.innerText = '-';
        if (canvasPredLabel) canvasPredLabel.innerText = 'Predicting...';
        if (canvasPredScore) canvasPredScore.innerText = '0.00%';
        if (canvasTop5) canvasTop5.innerHTML = `<div class="bar-row placeholder"><span>Draw on canvas to see distribution</span></div>`;
        
        if (uploadPredChar) uploadPredChar.innerText = '-';
        if (uploadPredLabel) uploadPredLabel.innerText = 'Upload image to predict';
        if (uploadPredScore) uploadPredScore.innerText = '0.00%';
        if (uploadTop5) uploadTop5.innerHTML = `<div class="bar-row placeholder"><span>Upload an image to see class probabilities</span></div>`;
    }

    function updateEdaGraphs(dataset) {
        if (imgAccCurve) imgAccCurve.src = `/static/assets/${dataset}_accuracy.png?t=${Date.now()}`;
        if (imgLossCurve) imgLossCurve.src = `/static/assets/${dataset}_loss.png?t=${Date.now()}`;
        if (imgCmHeatmap) imgCmHeatmap.src = `/static/assets/${dataset}_confusion_matrix.png?t=${Date.now()}`;
    }

    // --- Drawing Pad Mechanics ---
    function startDrawing(e, c, context) {
        isDrawing = true;
        const rect = c.getBoundingClientRect();
        
        // Handle touch vs mouse coordinates
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        lastX = clientX - rect.left;
        lastY = clientY - rect.top;
        
        // Draw a tiny dot on click start
        context.beginPath();
        context.arc(lastX, lastY, context.lineWidth / 2, 0, Math.PI * 2);
        context.fillStyle = '#ffffff';
        context.fill();
        context.beginPath();
        context.moveTo(lastX, lastY);
    }

    function draw(e, c, context, onDrawCallback) {
        if (!isDrawing) return;
        if (e.cancelable && e.touches) e.preventDefault(); // Prevent touch scroll scrolling
        
        const rect = c.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        
        context.lineTo(x, y);
        context.stroke();
        
        lastX = x;
        lastY = y;
        
        if (onDrawCallback) onDrawCallback();
    }

    function stopDrawing() {
        isDrawing = false;
    }

    // Setup Main Drawing Pad Listeners
    if (canvas) {
        canvas.addEventListener('mousedown', (e) => startDrawing(e, canvas, ctx));
        canvas.addEventListener('mousemove', (e) => draw(e, canvas, ctx, handleCanvasDraw));
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseleave', stopDrawing);
        
        canvas.addEventListener('touchstart', (e) => startDrawing(e, canvas, ctx));
        canvas.addEventListener('touchmove', (e) => draw(e, canvas, ctx, handleCanvasDraw));
        canvas.addEventListener('touchend', stopDrawing);
    }

    // Setup OCR Pad Listeners
    if (ocrCanvas) {
        ocrCanvas.addEventListener('mousedown', (e) => startDrawing(e, ocrCanvas, ocrCtx));
        ocrCanvas.addEventListener('mousemove', (e) => draw(e, ocrCanvas, ocrCtx));
        ocrCanvas.addEventListener('mouseup', stopDrawing);
        ocrCanvas.addEventListener('mouseleave', stopDrawing);
        
        ocrCanvas.addEventListener('touchstart', (e) => startDrawing(e, ocrCanvas, ocrCtx));
        ocrCanvas.addEventListener('touchmove', (e) => draw(e, ocrCanvas, ocrCtx));
        ocrCanvas.addEventListener('touchend', stopDrawing);
    }

    // --- Realtime / Button Prediction Logic ---
    let debounceTimer;
    function handleCanvasDraw() {
        if (!realtimeMode || !realtimeMode.checked) return;
        
        // Debounce predictions to avoid spamming Flask server
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            sendCanvasPrediction();
        }, 220);
    }

    if (btnClear) {
        btnClear.addEventListener('click', () => {
            initCanvas(canvas, ctx, 16);
            resetPredictionUI();
            playClearSound();
        });
    }

    if (btnPredictCanvas) {
        btnPredictCanvas.addEventListener('click', () => {
            sendCanvasPrediction();
        });
    }

    function toggleScanner(scannerEl, activate) {
        if (!scannerEl) return;
        if (activate) {
            scannerEl.classList.add('scanning');
        } else {
            scannerEl.classList.remove('scanning');
        }
    }

    function sendCanvasPrediction() {
        const dataUrl = canvas.toDataURL('image/png');
        const dataset = datasetSelect ? datasetSelect.value : 'emnist';
        
        toggleScanner(canvasScanner, true);
        const startTime = performance.now();

        postPrediction(dataUrl, dataset, (err, res) => {
            toggleScanner(canvasScanner, false);
            // Measure latency
            const duration = Math.round(performance.now() - startTime);
            if (diagLatency) diagLatency.innerText = `${duration} ms`;

            if (err) {
                console.error(err);
                if (err.not_trained) {
                    showTrainingAlert(dataset);
                } else {
                    if (canvasPredLabel) canvasPredLabel.innerText = "Error";
                    if (canvasPredChar) canvasPredChar.innerText = "?";
                    if (canvasPredScore) canvasPredScore.innerText = "0.00%";
                }
                return;
            }
            hideTrainingAlert();
            renderPredictionResults(res, canvasPredChar, canvasPredLabel, canvasPredScore, canvasTop5);
            playSuccessSound();
        });
    }

    // --- Upload Center logic ---
    if (dropzone) {
        dropzone.addEventListener('click', () => fileInput && fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleUploadedFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleUploadedFile(fileInput.files[0]);
            }
        });
    }

    function handleUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (PNG, JPG, JPEG)');
            return;
        }
        
        if (fileBadge) fileBadge.innerText = file.name;
        
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            const dataUrl = reader.result;
            
            // Show original image preview
            if (uploadPreview) uploadPreview.src = dataUrl;
            if (uploadPreviewWrapper) uploadPreviewWrapper.classList.remove('hidden');
            
            // Send to backend
            const dataset = datasetSelect ? datasetSelect.value : 'emnist';
            toggleScanner(uploadScanner, true);
            const startTime = performance.now();

            postPrediction(dataUrl, dataset, (err, res) => {
                toggleScanner(uploadScanner, false);
                const duration = Math.round(performance.now() - startTime);
                if (diagLatency) diagLatency.innerText = `${duration} ms`;

                if (err) {
                    console.error(err);
                    if (err.not_trained) {
                        showTrainingAlert(dataset);
                    } else {
                        if (uploadPredLabel) uploadPredLabel.innerText = "Error";
                        if (uploadPredChar) uploadPredChar.innerText = "?";
                        if (uploadPredScore) uploadPredScore.innerText = "0.00%";
                    }
                    return;
                }
                hideTrainingAlert();
                renderPredictionResults(res, uploadPredChar, uploadPredLabel, uploadPredScore, uploadTop5);
                playSuccessSound();
            });
        };
    }

    // --- Word OCR Logic ---
    if (btnOcrClear) {
        btnOcrClear.addEventListener('click', () => {
            initCanvas(ocrCanvas, ocrCtx, 10);
            if (ocrTranscription) ocrTranscription.innerText = '[Draw word and press OCR to decode]';
            if (ocrSegmentsList) ocrSegmentsList.innerHTML = `<p class="placeholder-text">Draw text above to view character segment details</p>`;
            playClearSound();
        });
    }

    if (btnOcrPredict) {
        btnOcrPredict.addEventListener('click', () => {
            const dataUrl = ocrCanvas.toDataURL('image/png');
            const dataset = datasetSelect ? datasetSelect.value : 'emnist'; 
            
            if (ocrTranscription) ocrTranscription.innerHTML = '<span class="loading-dots">Scanning character segments...</span>';
            if (ocrSegmentsList) ocrSegmentsList.innerHTML = '';
            
            toggleScanner(ocrScanner, true);
            const startTime = performance.now();

            fetch('/predict_ocr', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl, dataset: dataset })
            })
            .then(response => response.json())
            .then(res => {
                toggleScanner(ocrScanner, false);
                const duration = Math.round(performance.now() - startTime);
                if (diagLatency) diagLatency.innerText = `${duration} ms`;

                if (res.error) {
                    if (res.not_trained) {
                        showTrainingAlert(dataset);
                        if (ocrTranscription) ocrTranscription.innerText = 'Model Not Trained';
                    } else {
                        if (ocrTranscription) ocrTranscription.innerText = 'Error performing OCR';
                    }
                    return;
                }
                hideTrainingAlert();
                
                // Render combined text
                if (res.text.trim() === '') {
                    if (ocrTranscription) ocrTranscription.innerText = '[No characters detected]';
                    if (ocrSegmentsList) ocrSegmentsList.innerHTML = `<p class="placeholder-text">No drawing contours detected. Try drawing thicker strokes.</p>`;
                    return;
                }
                
                if (ocrTranscription) ocrTranscription.innerText = res.text;
                playSuccessSound();
                
                // Render individual characters segmented list
                res.characters.forEach((charData, index) => {
                    const card = document.createElement('div');
                    card.className = `ocr-char-card ${charData.is_space ? 'is-space' : ''}`;
                    
                    if (charData.is_space) {
                        card.innerHTML = `
                            <h5>␣</h5>
                            <span>Space</span>
                        `;
                    } else {
                        card.innerHTML = `
                            <h5>${charData.char}</h5>
                            <span>Idx: ${index + 1}</span>
                            <span>${(charData.confidence * 100).toFixed(0)}% score</span>
                        `;
                    }
                    if (ocrSegmentsList) ocrSegmentsList.appendChild(card);
                });
            })
            .catch(err => {
                toggleScanner(ocrScanner, false);
                console.error(err);
                if (ocrTranscription) ocrTranscription.innerText = 'Failed to fetch OCR results';
            });
        });
    }

    // --- Helper AJAX & Prediction Renderer functions ---
    function postPrediction(dataUrl, dataset, callback) {
        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl, dataset: dataset })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => {
                    throw errData;
                });
            }
            return response.json();
        })
        .then(res => callback(null, res))
        .catch(err => callback(err, null));
    }

    function renderPredictionResults(res, charEl, labelEl, scoreEl, top5El) {
        if (!res.success) return;
        
        // Single character display
        if (charEl) charEl.innerText = res.prediction;
        if (labelEl) labelEl.innerText = `Prediction: Class '${res.prediction}'`;
        if (scoreEl) scoreEl.innerText = `${(res.confidence * 100).toFixed(2)}%`;
        
        // Top 5 volumetric 3D bar charts
        if (top5El) {
            top5El.innerHTML = '';
            res.top_5.forEach(item => {
                const barRow = document.createElement('div');
                barRow.className = 'bar-row';
                
                const pct = (item.confidence * 100).toFixed(1);
                
                barRow.innerHTML = `
                    <span class="class-label">'${item.class}'</span>
                    <div class="bar-track-3d">
                        <div class="bar-fill-3d" style="width: ${pct}%"></div>
                    </div>
                    <span class="confidence-text">${pct}%</span>
                `;
                
                top5El.appendChild(barRow);
            });
        }
    }

    // --- Model Availability Banner check ---
    function checkModelStatus() {
        const dataset = datasetSelect ? datasetSelect.value : 'emnist';
        fetch('/', { method: 'GET' })
        .then(() => {
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=', dataset: dataset })
            })
            .then(res => {
                if (res.status === 400) {
                    res.json().then(data => {
                        if (data.not_trained) showTrainingAlert(dataset);
                    });
                } else {
                    hideTrainingAlert();
                }
            })
            .catch(() => {});
        });
    }
    
    // Run status check initially
    checkModelStatus();

    function showTrainingAlert(dataset) {
        if (datasetPlaceholder) datasetPlaceholder.innerText = dataset;
        if (trainingAlert) trainingAlert.classList.remove('hidden');
    }

    function hideTrainingAlert() {
        if (trainingAlert) trainingAlert.classList.add('hidden');
    }

    // --- EDA Tab Graph Toggle ---
    graphSubToggles.forEach(toggle => {
        toggle.addEventListener('click', () => {
            graphSubToggles.forEach(t => t.classList.remove('active'));
            toggle.classList.add('active');
            
            const graphType = toggle.getAttribute('data-graph');
            if (graphType === 'curves') {
                if (graphCurvesPanel) graphCurvesPanel.classList.add('active');
                if (graphCmPanel) graphCmPanel.classList.remove('active');
            } else {
                if (graphCurvesPanel) graphCurvesPanel.classList.remove('active');
                if (graphCmPanel) graphCmPanel.classList.add('active');
            }
        });
    });
});
