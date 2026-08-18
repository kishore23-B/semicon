/**
 * Semiconductor AI Image Restoration Front-End Application Logic
 * Implements real-time sample loading, interactive 3-panel synchronization,
 * split slider comparison, 1D cross-section graph plotting, FFT spectrum, and custom upload.
 */

// Application State
const state = {
  currentSample: "000000.npy",
  currentSplit: "validation",
  sampleData: null,
  catalog: [],
  useTTA: true,
  sliceRow: 128,
  zoomLevel: 1.0,
  viewMode: "triplet",
  activeTab: "cross-section",
  filter: "all"
};

// DOM Elements
const elements = {
  sampleList: document.getElementById("sample-list"),
  sampleCountBadge: document.getElementById("sample-count-badge"),
  currentFilename: document.getElementById("current-filename"),
  currentTag: document.getElementById("current-tag"),
  devicePill: document.getElementById("device-pill"),
  statLatency: document.getElementById("stat-latency"),
  statPsnr: document.getElementById("stat-psnr"),
  statSsim: document.getElementById("stat-ssim"),
  statLpips: document.getElementById("stat-lpips"),
  statPsnrContainer: document.getElementById("stat-psnr-container"),
  statSsimContainer: document.getElementById("stat-ssim-container"),
  statLpipsContainer: document.getElementById("stat-lpips-container"),
  
  // Viewport
  tripletContainer: document.getElementById("triplet-container"),
  featureViewContainer: document.getElementById("feature-view-container"),
  customFeatureDisplay: document.getElementById("custom-feature-display"),
  sliderWrapper: document.getElementById("slider-wrapper"),
  sliderAfterContainer: document.getElementById("slider-after-container"),
  sliderHandle: document.getElementById("slider-handle"),
  sliderBefore: document.getElementById("slider-before"),
  sliderAfter: document.getElementById("slider-after"),
  featureImg: document.getElementById("feature-img"),
  featureCardTitle: document.getElementById("feature-card-title"),
  
  // Triplet Images
  imgGt: document.getElementById("img-gt"),
  imgLr: document.getElementById("img-lr"),
  imgRestored: document.getElementById("img-restored"),
  panelGt: document.getElementById("panel-gt"),
  gtDimTag: document.getElementById("gt-dim-tag"),
  lrDimTag: document.getElementById("lr-dim-tag"),
  restoredDimTag: document.getElementById("restored-dim-tag"),
  gtStatContrast: document.getElementById("gt-stat-contrast"),
  lrOvershootTag: document.getElementById("lr-overshoot-tag"),
  lrDynamicRange: document.getElementById("lr-dynamic-range"),
  restoredGainTag: document.getElementById("restored-gain-tag"),
  restoredStatusTag: document.getElementById("restored-status-tag"),
  
  // Guide lines
  guideGt: document.getElementById("guide-gt"),
  guideLr: document.getElementById("guide-lr"),
  guideRestored: document.getElementById("guide-restored"),
  
  // Canvas charts
  profileCanvas: document.getElementById("profileCanvas"),
  histCanvas: document.getElementById("histCanvas"),
  chartRowNum: document.getElementById("chart-row-num"),
  
  // Feature Images
  fftGtImg: document.getElementById("fft-gt-img"),
  fftLrImg: document.getElementById("fft-lr-img"),
  fftRestoredImg: document.getElementById("fft-restored-img"),
  edgeLrImg: document.getElementById("edge-lr-img"),
  edgeRestoredImg: document.getElementById("edge-restored-img"),
  
  // Metric Cards
  mcSsim: document.getElementById("mc-ssim"),
  mcSsimDelta: document.getElementById("mc-ssim-delta"),
  mcPsnr: document.getElementById("mc-psnr"),
  mcPsnrDelta: document.getElementById("mc-psnr-delta"),
  mcLpips: document.getElementById("mc-lpips"),
  mcMae: document.getElementById("mc-mae"),
  
  // Controls
  ttaToggle: document.getElementById("tta-toggle"),
  viewModeSelect: document.getElementById("view-mode-select"),
  sliceSlider: document.getElementById("slice-slider"),
  sliceVal: document.getElementById("slice-val"),
  btnZoomIn: document.getElementById("btn-zoom-in"),
  btnZoomOut: document.getElementById("btn-zoom-out"),
  btnZoomReset: document.getElementById("btn-zoom-reset"),
  zoomLevelSpan: document.getElementById("zoom-level"),
  btnDownloadNpy: document.getElementById("btn-download-npy"),
  btnDownloadPng: document.getElementById("btn-download-png"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input")
};

// Initialize Application
async function initApp() {
  setupEventListeners();
  await loadSystemInfo();
  await loadSampleCatalog();
  if (state.catalog.length > 0) {
    loadSample(state.catalog[0].filename, state.catalog[0].split);
  }
}

// Load System Information
async function loadSystemInfo() {
  try {
    const res = await fetch("/api/system_info");
    const data = await res.json();
    elements.devicePill.textContent = `GPU: ${data.device_name}`;
    document.getElementById("header-ssim").textContent = data.val_ssim.toFixed(4);
    document.getElementById("header-psnr").textContent = `${data.val_psnr.toFixed(2)} dB`;
  } catch (err) {
    console.warn("System info fetch fallback:", err);
  }
}

// Load Sample Catalog
async function loadSampleCatalog() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    state.catalog = data.samples;
    renderSampleList();
  } catch (err) {
    console.error("Failed to load sample catalog:", err);
  }
}

// Render Sample List Sidebar
function renderSampleList() {
  elements.sampleList.innerHTML = "";
  const filtered = state.catalog.filter(s => {
    if (state.filter === "all") return true;
    if (state.filter === "contacts") return s.category.toLowerCase().includes("contact");
    if (state.filter === "traces") return s.category.toLowerCase().includes("trace");
    if (state.filter === "outliers") return s.category.toLowerCase().includes("outlier");
    if (state.filter === "test") return s.split === "test";
    return true;
  });

  elements.sampleCountBadge.textContent = `${filtered.length} items`;

  filtered.forEach(s => {
    const isCur = (s.filename === state.currentSample && s.split === state.currentSplit);
    const item = document.createElement("div");
    item.className = `sample-item ${isCur ? "active" : ""}`;
    item.innerHTML = `
      <div class="sample-item-info">
        <span class="sample-name">${s.filename}</span>
        <span class="sample-tag">${s.category} • ${s.tag}</span>
      </div>
      <span class="sample-split-tag ${s.split === 'validation' ? 'val' : 'test'}">
        ${s.split === 'validation' ? 'VAL' : 'TEST'}
      </span>
    `;
    item.addEventListener("click", () => {
      document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("active"));
      item.classList.add("active");
      loadSample(s.filename, s.split);
    });
    elements.sampleList.appendChild(item);
  });
}

// Load and Restore Sample
async function loadSample(filename, split = "validation") {
  state.currentSample = filename;
  state.currentSplit = split;
  elements.currentFilename.textContent = `Sample: [${split.toUpperCase()}] ${filename}`;
  const catalogItem = state.catalog.find(s => s.filename === filename && s.split === split);
  elements.currentTag.textContent = catalogItem ? `${catalogItem.category} • ${catalogItem.tag}` : "Custom Image";

  try {
    const res = await fetch(`/api/load_sample?filename=${filename}&split=${split}&tta=${state.useTTA}&row=${state.sliceRow}`);
    const data = await res.json();
    state.sampleData = data;
    updateUIWithSampleData(data);
  } catch (err) {
    console.error(`Failed to load sample ${filename} (${split}):`, err);
  }
}

// Update UI with Sample Data
function updateUIWithSampleData(data) {
  // Update Header Stats
  elements.statLatency.textContent = `${data.latency_ms.toFixed(1)} ms`;
  
  if (data.metrics && data.metrics.has_gt) {
    elements.statPsnrContainer.style.display = "flex";
    elements.statSsimContainer.style.display = "flex";
    elements.statLpipsContainer.style.display = "flex";
    elements.statPsnr.textContent = `${data.metrics.psnr.toFixed(2)} dB`;
    elements.statSsim.textContent = data.metrics.ssim.toFixed(4);
    elements.statLpips.textContent = data.metrics.lpips.toFixed(4);

    // Metric cards
    elements.mcSsim.textContent = data.metrics.ssim.toFixed(4);
    elements.mcSsimDelta.textContent = `+${data.metrics.ssim_gain.toFixed(4)} vs Bicubic`;
    elements.mcPsnr.textContent = `${data.metrics.psnr.toFixed(2)} dB`;
    elements.mcPsnrDelta.textContent = `+${data.metrics.psnr_gain.toFixed(2)} dB vs Bicubic`;
    elements.mcLpips.textContent = data.metrics.lpips.toFixed(4);
    elements.mcMae.textContent = data.metrics.mae.toFixed(4);

    elements.panelGt.style.display = "flex";
    elements.imgGt.src = data.gt_img_b64;
    elements.gtDimTag.textContent = `${data.gt_shape[0]}×${data.gt_shape[1]} • High SNR`;
    elements.restoredGainTag.textContent = `Gain: +${data.metrics.psnr_gain.toFixed(2)} dB`;
  } else {
    elements.statPsnrContainer.style.display = "none";
    elements.statSsimContainer.style.display = "none";
    elements.statLpipsContainer.style.display = "none";
    elements.panelGt.style.display = "none";
    elements.restoredGainTag.textContent = `Blind Test Restoration`;
  }

  // Input LR
  elements.imgLr.src = data.lr_img_b64;
  elements.lrDimTag.textContent = `${data.lr_shape[0]}×${data.lr_shape[1]} • Speckle + Blur`;
  elements.lrOvershootTag.textContent = `Overshoot: ${data.overshoot_pct.toFixed(1)}%`;
  elements.lrDynamicRange.textContent = `Range: [${data.lr_min.toFixed(2)}, ${data.lr_max.toFixed(2)}]`;

  // Restored Output
  elements.imgRestored.src = data.restored_img_b64;
  elements.restoredDimTag.textContent = `${data.restored_shape[0]}×${data.restored_shape[1]} • Super-Resolved`;

  // Slider view images
  elements.sliderBefore.src = data.bicubic_img_b64;
  elements.sliderAfter.src = data.restored_img_b64;

  // FFT Images
  if (data.fft_gt_b64) elements.fftGtImg.src = data.fft_gt_b64;
  elements.fftLrImg.src = data.fft_lr_b64;
  elements.fftRestoredImg.src = data.fft_restored_b64;

  // Edge Images
  elements.edgeLrImg.src = data.edge_lr_b64;
  elements.edgeRestoredImg.src = data.edge_restored_b64;

  // Update Cross-Section Chart
  renderCrossSectionChart(data.cross_section);

  // Update Histogram Chart
  renderHistogramChart(data.histogram_lr, data.histogram_restored);

  // Update view mode display
  applyViewMode();
}

// Render 1D Cross-Section Intensity Chart
function renderCrossSectionChart(csData) {
  const canvas = elements.profileCanvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Draw Grid Lines
  ctx.strokeStyle = "#1a2234";
  ctx.lineWidth = 1;
  for (let y = 0; y <= h; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Helper to plot 1D array
  function plotLine(arr, color, lineWidth = 2) {
    if (!arr || arr.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    const step = w / (arr.length - 1);
    for (let i = 0; i < arr.length; i++) {
      const x = i * step;
      const y = h - (arr[i] * (h - 20) + 10);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Plot Ground Truth (Blue)
  if (csData.gt) {
    plotLine(csData.gt, "#3b82f6", 2);
  }
  // Plot Noisy LR (Orange)
  plotLine(csData.bicubic_lr, "#f59e0b", 1.5);
  // Plot AI Restored (Green)
  plotLine(csData.restored, "#10b981", 2.5);

  // Update Row Label
  elements.chartRowNum.textContent = csData.row_index;
}

// Render Histogram Chart
function renderHistogramChart(histLr, histRestored) {
  const canvas = elements.histCanvas;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  if (!histRestored || !histRestored.counts) return;

  const countsRestored = histRestored.counts;
  const countsLr = histLr ? histLr.counts : [];
  const maxCount = Math.max(...countsRestored, ...(countsLr || [1]));
  const barWidth = w / countsRestored.length;

  for (let i = 0; i < countsRestored.length; i++) {
    const x = i * barWidth;

    // Noisy bar (Orange outline)
    if (countsLr && countsLr[i]) {
      const bhLr = (countsLr[i] / maxCount) * (h - 20);
      ctx.fillStyle = "rgba(245, 158, 11, 0.3)";
      ctx.fillRect(x, h - bhLr - 10, barWidth - 1, bhLr);
    }

    // Restored bar (Green)
    const bh = (countsRestored[i] / maxCount) * (h - 20);
    ctx.fillStyle = "rgba(16, 185, 129, 0.6)";
    ctx.fillRect(x, h - bh - 10, barWidth - 1, bh);
  }
}

// View Mode Handler
function applyViewMode() {
  const mode = elements.viewModeSelect.value;
  state.viewMode = mode;

  if (mode === "triplet") {
    elements.tripletContainer.style.display = "grid";
    elements.featureViewContainer.style.display = "none";
  } else if (mode === "slider") {
    elements.tripletContainer.style.display = "none";
    elements.featureViewContainer.style.display = "flex";
    elements.sliderWrapper.style.display = "block";
    elements.customFeatureDisplay.style.display = "none";
  } else {
    elements.tripletContainer.style.display = "none";
    elements.featureViewContainer.style.display = "flex";
    elements.sliderWrapper.style.display = "none";
    elements.customFeatureDisplay.style.display = "flex";

    if (mode === "diff" && state.sampleData) {
      elements.featureCardTitle.textContent = "Residual Error Difference Heatmap (Absolute Error)";
      elements.featureImg.src = state.sampleData.diff_img_b64 || state.sampleData.restored_img_b64;
    } else if (mode === "edges" && state.sampleData) {
      elements.featureCardTitle.textContent = "Sobel Boundary Gradient Map";
      elements.featureImg.src = state.sampleData.edge_restored_b64;
    } else if (mode === "fft" && state.sampleData) {
      elements.featureCardTitle.textContent = "2D Fourier (FFT) Log-Magnitude Frequency Spectrum";
      elements.featureImg.src = state.sampleData.fft_restored_b64;
    }
  }
}

// Setup Event Listeners
function setupEventListeners() {
  // Filter pills
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.filter = btn.getAttribute("data-filter");
      renderSampleList();
    });
  });

  // TTA Toggle
  elements.ttaToggle.addEventListener("change", (e) => {
    state.useTTA = e.target.checked;
    if (state.currentSample) {
      loadSample(state.currentSample, state.currentSplit);
    }
  });

  // View Mode Select
  elements.viewModeSelect.addEventListener("change", applyViewMode);

  // Cross-Section Row Slider
  elements.sliceSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    state.sliceRow = val;
    elements.sliceVal.textContent = `Row: ${val} / 256`;
    
    // Update guide line positions
    const pct = (val / 256) * 100;
    elements.guideGt.style.top = `${pct}%`;
    elements.guideGt.style.display = "block";
    elements.guideLr.style.top = `${pct}%`;
    elements.guideLr.style.display = "block";
    elements.guideRestored.style.top = `${pct}%`;
    elements.guideRestored.style.display = "block";

    if (state.currentSample) {
      loadSample(state.currentSample, state.currentSplit);
    }
  });

  // Analytics Deck Tabs
  document.querySelectorAll(".deck-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".deck-tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const targetId = `pane-${tab.getAttribute("data-tab")}`;
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add("active");
    });
  });

  // Zoom Controls
  elements.btnZoomIn.addEventListener("click", () => {
    state.zoomLevel = Math.min(state.zoomLevel + 0.25, 3.0);
    applyZoom();
  });
  elements.btnZoomOut.addEventListener("click", () => {
    state.zoomLevel = Math.max(state.zoomLevel - 0.25, 0.5);
    applyZoom();
  });
  elements.btnZoomReset.addEventListener("click", () => {
    state.zoomLevel = 1.0;
    applyZoom();
  });

  function applyZoom() {
    elements.zoomLevelSpan.textContent = `${Math.round(state.zoomLevel * 100)}%`;
    [elements.imgGt, elements.imgLr, elements.imgRestored].forEach(img => {
      img.style.transform = `scale(${state.zoomLevel})`;
    });
  }

  // Interactive Split Slider Dragging
  let isDraggingSlider = false;
  elements.sliderHandle.addEventListener("mousedown", () => { isDraggingSlider = true; });
  window.addEventListener("mouseup", () => { isDraggingSlider = false; });
  window.addEventListener("mousemove", (e) => {
    if (!isDraggingSlider) return;
    const rect = elements.sliderWrapper.getBoundingClientRect();
    let x = e.clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const pct = (x / rect.width) * 100;
    elements.sliderAfterContainer.style.width = `${pct}%`;
    elements.sliderHandle.style.left = `${pct}%`;
  });

  // Custom File Upload
  elements.dropzone.addEventListener("click", () => {
    elements.fileInput.click();
  });

  elements.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    elements.dropzone.style.borderColor = "var(--accent-blue)";
  });

  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.style.borderColor = "";
  });

  elements.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    elements.dropzone.style.borderColor = "";
    if (e.dataTransfer.files.length > 0) {
      handleCustomFileUpload(e.dataTransfer.files[0]);
    }
  });

  elements.fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleCustomFileUpload(e.target.files[0]);
    }
  });

  // Export Buttons
  elements.btnDownloadNpy.addEventListener("click", () => {
    if (!state.currentSample) return;
    const link = document.createElement("a");
    link.href = `/restored_test/${state.currentSample}`;
    link.download = state.currentSample;
    link.click();
  });

  elements.btnDownloadPng.addEventListener("click", () => {
    if (!state.sampleData || !state.sampleData.restored_img_b64) return;
    const link = document.createElement("a");
    link.href = state.sampleData.restored_img_b64;
    link.download = `${state.currentSample.replace(/\.npy$/, '')}_restored.png`;
    link.click();
  });
}

// Handle Custom File Upload
async function handleCustomFileUpload(file) {
  const reader = new FileReader();
  reader.onload = async (e) => {
    const base64Data = e.target.result;
    elements.currentFilename.textContent = `Custom: ${file.name}`;
    elements.currentTag.textContent = `User Uploaded Inspection Image`;

    try {
      const res = await fetch("/api/restore_custom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_b64: base64Data,
          use_tta: state.useTTA
        })
      });
      const data = await res.json();
      state.sampleData = data;
      updateUIWithSampleData(data);
    } catch (err) {
      console.error("Failed to restore uploaded image:", err);
    }
  };
  reader.readAsDataURL(file);
}

// Start application when DOM is ready
document.addEventListener("DOMContentLoaded", initApp);
