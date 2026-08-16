/**
 * tools/image_filter/app.js
 * =========================
 * Client logic cho công cụ lọc ảnh Keyframe & tìm kiếm ảnh trùng lặp AIC2026.
 */

// Global Application State
const appState = {
  batches: [],
  currentBatch: null,
  currentVideoId: null,
  currentVideoImages: [],
  currentFilterView: "all", // "all" | "unfiltered" | "filtered"
  totalFilteredCount: 0,
  
  // Similar search modal state
  activeRefImage: null,
  similarResults: [],
  selectedSimilarKeys: new Set(),
};

// DOM Elements
const elements = {
  batchList: document.getElementById("batch-list"),
  batchSummary: document.getElementById("batch-summary"),
  videoSearchInput: document.getElementById("video-search-input"),
  clearVideoSearch: document.getElementById("clear-video-search"),
  
  statTotalImages: document.getElementById("stat-total-images"),
  statFilteredCount: document.getElementById("stat-filtered-count"),
  statCurrentVideo: document.getElementById("stat-current-video"),
  btnExportCount: document.getElementById("btn-export-count"),
  
  btnRefresh: document.getElementById("btn-refresh"),
  btnExportJson: document.getElementById("btn-export-json"),
  
  currentBatchBadge: document.getElementById("current-batch-badge"),
  currentVideoHeading: document.getElementById("current-video-heading"),
  btnPrevVideo: document.getElementById("btn-prev-video"),
  btnNextVideo: document.getElementById("btn-next-video"),
  
  filterTabs: document.querySelectorAll(".filter-tab"),
  cntAll: document.getElementById("cnt-all"),
  cntUnfiltered: document.getElementById("cnt-unfiltered"),
  cntFiltered: document.getElementById("cnt-filtered"),
  
  btnSelectAllVideo: document.getElementById("btn-select-all-video"),
  btnDeselectAllVideo: document.getElementById("btn-deselect-all-video"),
  
  emptyState: document.getElementById("empty-state"),
  keyframeGrid: document.getElementById("keyframe-grid"),
  
  // Similar Modal Elements
  similarModal: document.getElementById("similar-modal"),
  btnCloseSimilarModal: document.getElementById("btn-close-similar-modal"),
  btnCancelSimilar: document.getElementById("btn-cancel-similar"),
  refImagePreview: document.getElementById("ref-image-preview"),
  refSceneId: document.getElementById("ref-scene-id"),
  refVideoId: document.getElementById("ref-video-id"),
  refFrameId: document.getElementById("ref-frame-id"),
  chkKeepRefImage: document.getElementById("chk-keep-ref-image"),
  simThresholdSlider: document.getElementById("sim-threshold-slider"),
  thresholdDisplay: document.getElementById("threshold-display"),
  simZoomSlider: document.getElementById("sim-zoom-slider"),
  zoomDisplay: document.getElementById("zoom-display"),
  similarResultsGrid: document.getElementById("similar-results-grid"),
  simFoundCount: document.getElementById("sim-found-count"),
  simSelectedCount: document.getElementById("sim-selected-count"),
  btnApplySimilarCount: document.getElementById("btn-apply-similar-count"),
  btnApplySimilarFilter: document.getElementById("btn-apply-similar-filter"),
  btnQuick95: document.getElementById("btn-quick-95"),
  btnQuick90: document.getElementById("btn-quick-90"),
  btnQuickAll: document.getElementById("btn-quick-all"),
  btnQuickNone: document.getElementById("btn-quick-none"),
  
  // Lightbox
  lightboxModal: document.getElementById("lightbox-modal"),
  btnCloseLightbox: document.getElementById("btn-close-lightbox"),
  lightboxImg: document.getElementById("lightbox-img"),
  lightboxTitle: document.getElementById("lightbox-title"),
  lightboxDetails: document.getElementById("lightbox-details"),
  
  toastContainer: document.getElementById("toast-container"),
};


// =============================================================================
// API Service Calls
// =============================================================================

async function fetchBatches() {
  try {
    const res = await fetch("/api/batches");
    const data = await res.json();
    appState.batches = data.batches || [];
    appState.totalFilteredCount = data.total_filtered || 0;
    updateGlobalStats();
    renderBatchList();
  } catch (err) {
    showToast("Lỗi khi tải danh sách batch: " + err.message, "danger");
  }
}

async function fetchVideos(batch) {
  try {
    const res = await fetch(`/api/videos/${batch}`);
    const data = await res.json();
    return data.videos || [];
  } catch (err) {
    showToast(`Lỗi khi tải videos batch ${batch}: ` + err.message, "danger");
    return [];
  }
}

async function fetchImages(batch, videoId) {
  try {
    elements.keyframeGrid.innerHTML = '<div class="loading-spinner">Đang tải ảnh keyframe...</div>';
    elements.keyframeGrid.style.display = "grid";
    elements.emptyState.style.display = "none";
    
    const res = await fetch(`/api/images/${batch}/${videoId}`);
    const data = await res.json();
    appState.currentVideoImages = data.images || [];
    renderKeyframeGrid();
    updateVideoCounts();
  } catch (err) {
    showToast(`Lỗi tải ảnh ${batch}/${videoId}: ` + err.message, "danger");
  }
}

async function postFilterUpdate(action, items, reason = "manual") {
  try {
    const res = await fetch("/api/filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, items, reason }),
    });
    const data = await res.json();
    appState.totalFilteredCount = data.total_filtered;
    updateGlobalStats();
    return data;
  } catch (err) {
    showToast("Lỗi cập nhật filter: " + err.message, "danger");
    return null;
  }
}

async function fetchSimilarImages(sceneId, batch, videoId, frameId, minSim = 0.70) {
  try {
    const res = await fetch("/api/similar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene_id: sceneId,
        batch: batch,
        video_id: videoId,
        frame_id: frameId,
        top_k: 1000,
        min_similarity: minSim,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Search failed");
    }
    return await res.json();
  } catch (err) {
    showToast("Lỗi tìm ảnh tương tự: " + err.message, "danger");
    return null;
  }
}


// =============================================================================
// UI Rendering
// =============================================================================

function updateGlobalStats() {
  elements.statFilteredCount.textContent = appState.totalFilteredCount.toLocaleString();
  elements.btnExportCount.textContent = appState.totalFilteredCount.toLocaleString();
}

function renderBatchList() {
  elements.batchList.innerHTML = "";
  let totalVideos = 0;

  appState.batches.forEach((b) => {
    totalVideos += b.video_count;
    const group = document.createElement("div");
    group.className = "batch-group";
    group.dataset.batch = b.batch;

    const hasFiltered = b.filtered_count > 0;
    const badgeClass = hasFiltered ? "batch-badge has-filtered" : "batch-badge";
    const badgeText = hasFiltered ? `❌ ${b.filtered_count} / ${b.video_count} vids` : `${b.video_count} vids`;

    group.innerHTML = `
      <div class="batch-header">
        <span class="batch-title">
          <span class="arrow-icon">▶</span>
          <span>📁 ${b.batch}</span>
        </span>
        <span class="${badgeClass}">${badgeText}</span>
      </div>
      <div class="video-list" id="vlist-${b.batch}">
        <div class="loading-spinner" style="padding: 10px;">Đang nạp video...</div>
      </div>
    `;

    const header = group.querySelector(".batch-header");
    header.addEventListener("click", async () => {
      const isOpen = group.classList.contains("open");
      // Close all other groups if desired, or toggle
      group.classList.toggle("open");
      
      if (!isOpen && !group.dataset.loaded) {
        const vListEl = document.getElementById(`vlist-${b.batch}`);
        const videos = await fetchVideos(b.batch);
        group.dataset.loaded = "true";
        group.dataset.videos = JSON.stringify(videos);
        renderVideoItems(b.batch, videos, vListEl);
      }
    });

    elements.batchList.appendChild(group);
  });

  elements.batchSummary.textContent = `${appState.batches.length} batches • ${totalVideos} videos tổng cộng`;
}

function renderVideoItems(batch, videos, container) {
  container.innerHTML = "";
  if (videos.length === 0) {
    container.innerHTML = '<div style="padding: 8px; color: var(--text-dim);">Không có video</div>';
    return;
  }

  videos.forEach((v) => {
    const item = document.createElement("div");
    item.className = "video-item";
    item.dataset.batch = batch;
    item.dataset.videoId = v.video_id;

    if (appState.currentBatch === batch && appState.currentVideoId === v.video_id) {
      item.classList.add("active");
    }

    const filteredTag = v.filtered_count > 0 ? `<span class="v-filtered-tag">❌${v.filtered_count}</span>` : "";

    item.innerHTML = `
      <span class="v-id">${v.video_id}</span>
      <span class="v-count">${v.image_count} imgs${filteredTag}</span>
    `;

    item.addEventListener("click", (e) => {
      e.stopPropagation();
      selectVideo(batch, v.video_id);
    });

    container.appendChild(item);
  });
}

function selectVideo(batch, videoId) {
  appState.currentBatch = batch;
  appState.currentVideoId = videoId;

  // Highlight active in sidebar
  document.querySelectorAll(".video-item").forEach((el) => {
    if (el.dataset.batch === batch && el.dataset.videoId === videoId) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });

  elements.currentBatchBadge.textContent = batch;
  elements.currentVideoHeading.textContent = `${batch}_${videoId}`;
  elements.statCurrentVideo.textContent = `${batch}_${videoId}`;

  fetchImages(batch, videoId);
}

function updateVideoCounts() {
  const total = appState.currentVideoImages.length;
  const filtered = appState.currentVideoImages.filter((img) => img.is_filtered).length;
  const unfiltered = total - filtered;

  elements.cntAll.textContent = total;
  elements.cntUnfiltered.textContent = unfiltered;
  elements.cntFiltered.textContent = filtered;
}

let lastClickedIndex = -1;

function renderKeyframeGrid() {
  elements.keyframeGrid.innerHTML = "";
  elements.keyframeGrid.style.display = "grid";
  elements.emptyState.style.display = "none";
  lastClickedIndex = -1;
  
  const filter = appState.currentFilterView;
  const filteredImages = appState.currentVideoImages.filter((img) => {
    if (filter === "filtered") return img.is_filtered;
    if (filter === "unfiltered") return !img.is_filtered;
    return true;
  });

  if (filteredImages.length === 0) {
    elements.keyframeGrid.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1; height: 300px;">
        <div class="empty-icon">🔍</div>
        <h3>Không có ảnh nào khớp bộ lọc</h3>
        <p>Thử chuyển sang tab "Tất cả" hoặc chọn video khác.</p>
      </div>
    `;
    return;
  }

  filteredImages.forEach((img, idx) => {
    const card = document.createElement("div");
    card.className = "keyframe-card" + (img.is_filtered ? " filtered" : "");
    card.dataset.key = img.keyframe_path;
    card.dataset.index = idx;

    card.innerHTML = `
      <div class="card-checkbox" title="Click để chọn/bỏ lọc ảnh này"></div>
      <div class="card-img-wrap">
        <img src="${img.image_url}" alt="Frame ${img.frame_id}" loading="lazy">
        <div class="filter-overlay">
          <div class="overlay-icon">✕</div>
          <div class="overlay-text">ĐÃ LỌC</div>
        </div>
      </div>
      <div class="card-info">
        <span class="frame-id">F#${img.frame_id}</span>
        <span class="frame-time">${img.timestamp_mid.toFixed(2)}s</span>
      </div>
      <div class="card-actions">
        <button class="action-btn-sm btn-similar" title="Tìm ảnh giống để lọc trùng lặp">🔍 Giống</button>
        <button class="action-btn-sm btn-preview" title="Xem phóng to">🔎</button>
      </div>
    `;

    // Click card (supports Shift + Click for range selection!)
    card.addEventListener("click", (e) => {
      if (e.target.closest(".card-actions")) return;
      
      if (e.shiftKey && lastClickedIndex >= 0 && lastClickedIndex !== idx) {
        const start = Math.min(lastClickedIndex, idx);
        const end = Math.max(lastClickedIndex, idx);
        const targetStatus = !img.is_filtered;
        
        const itemsToUpdate = [];
        for (let i = start; i <= end; i++) {
          const targetImg = filteredImages[i];
          if (targetImg && targetImg.is_filtered !== targetStatus) {
            targetImg.is_filtered = targetStatus;
            itemsToUpdate.push({
              batch: appState.currentBatch,
              video_id: appState.currentVideoId,
              filename: targetImg.filename,
              frame_id: targetImg.frame_id,
              keyframe_path: targetImg.keyframe_path,
              reason: "manual_range",
            });
            const c = document.querySelector(`.keyframe-card[data-key="${targetImg.keyframe_path}"]`);
            if (c) c.classList.toggle("filtered", targetStatus);
          }
        }
        updateVideoCounts();
        if (itemsToUpdate.length > 0) {
          postFilterUpdate(targetStatus ? "add" : "remove", itemsToUpdate);
          updateBatchSidebarCount(appState.currentBatch);
        }
      } else {
        toggleSingleImageFilter(img);
      }
      lastClickedIndex = idx;
    });

    // Similar button click
    const btnSim = card.querySelector(".btn-similar");
    btnSim.addEventListener("click", (e) => {
      e.stopPropagation();
      openSimilarModal(img);
    });

    // Preview button click
    const btnPrev = card.querySelector(".btn-preview");
    btnPrev.addEventListener("click", (e) => {
      e.stopPropagation();
      openLightbox(img);
    });

    elements.keyframeGrid.appendChild(card);
  });
}


// =============================================================================
// Filter Actions
// =============================================================================

async function toggleSingleImageFilter(img) {
  const newStatus = !img.is_filtered;
  img.is_filtered = newStatus;

  // Immediate UI update
  const card = document.querySelector(`.keyframe-card[data-key="${img.keyframe_path}"]`);
  if (card) {
    card.classList.toggle("filtered", newStatus);
  }
  updateVideoCounts();

  const action = newStatus ? "add" : "remove";
  const itemPayload = [{
    batch: appState.currentBatch,
    video_id: appState.currentVideoId,
    filename: img.filename,
    frame_id: img.frame_id,
    keyframe_path: img.keyframe_path,
    reason: "manual",
  }];

  await postFilterUpdate(action, itemPayload);
  updateBatchSidebarCount(appState.currentBatch);
}

async function bulkFilterCurrentVideo(action) {
  const isAdd = action === "add";
  const itemsToUpdate = [];

  appState.currentVideoImages.forEach((img) => {
    if (img.is_filtered !== isAdd) {
      img.is_filtered = isAdd;
      itemsToUpdate.push({
        batch: appState.currentBatch,
        video_id: appState.currentVideoId,
        filename: img.filename,
        frame_id: img.frame_id,
        keyframe_path: img.keyframe_path,
        reason: "manual_video_bulk",
      });
    }
  });

  if (itemsToUpdate.length === 0) return;

  renderKeyframeGrid();
  updateVideoCounts();

  await postFilterUpdate(action, itemsToUpdate);
  updateBatchSidebarCount(appState.currentBatch);
  showToast(`Đã ${isAdd ? "đánh dấu lọc" : "bỏ lọc"} toàn bộ video ${appState.currentBatch}_${appState.currentVideoId}`, "success");
}

function updateBatchSidebarCount(batch) {
  // Update badge in sidebar
  const vItem = document.querySelector(`.video-item[data-batch="${batch}"][data-video-id="${appState.currentVideoId}"]`);
  if (vItem) {
    const fCount = appState.currentVideoImages.filter((img) => img.is_filtered).length;
    const tag = vItem.querySelector(".v-filtered-tag");
    if (fCount > 0) {
      if (tag) tag.textContent = `❌${fCount}`;
      else {
        const span = document.createElement("span");
        span.className = "v-filtered-tag";
        span.textContent = `❌${fCount}`;
        vItem.querySelector(".v-count").appendChild(span);
      }
    } else if (tag) {
      tag.remove();
    }
  }
}


// =============================================================================
// Similar / Duplicate Search Modal
// =============================================================================

async function openSimilarModal(img) {
  appState.activeRefImage = img;
  elements.refImagePreview.src = img.image_url;
  elements.refSceneId.textContent = img.scene_id || "--";
  elements.refVideoId.textContent = `${appState.currentBatch}_${appState.currentVideoId}`;
  elements.refFrameId.textContent = `#${img.frame_id}`;
  
  const cols = elements.simZoomSlider ? elements.simZoomSlider.value : "4";
  elements.similarResultsGrid.className = `similar-grid cols-${cols}`;
  elements.similarResultsGrid.innerHTML = '<div class="loading-spinner">Đang quét vector cosine similarity...</div>';
  elements.similarModal.classList.add("open");

  const minSim = parseFloat(elements.simThresholdSlider.value) / 100.0;
  const res = await fetchSimilarImages(img.scene_id, appState.currentBatch, appState.currentVideoId, img.frame_id, 0.70);

  if (res && res.results) {
    appState.similarResults = res.results;
    elements.simFoundCount.textContent = res.results.length;
    autoSelectByThreshold(minSim);
    renderSimilarResults();
  } else {
    elements.similarResultsGrid.innerHTML = '<div class="loading-spinner">Không tìm thấy ảnh tương đồng nào.</div>';
  }
}

function autoSelectByThreshold(threshold) {
  appState.selectedSimilarKeys.clear();
  const keepRef = elements.chkKeepRefImage.checked;

  appState.similarResults.forEach((r) => {
    // If it is reference image and user chooses to keep it, do not select
    if (r.is_query_image && keepRef) {
      return;
    }
    if (r.similarity >= threshold) {
      appState.selectedSimilarKeys.add(r.keyframe_path);
    }
  });

  updateSimilarSelectionCounts();
}

function updateSimilarSelectionCounts() {
  const cnt = appState.selectedSimilarKeys.size;
  elements.simSelectedCount.textContent = cnt;
  elements.btnApplySimilarCount.textContent = cnt;
}

function renderSimilarResults() {
  elements.similarResultsGrid.innerHTML = "";

  if (appState.similarResults.length === 0) {
    elements.similarResultsGrid.innerHTML = '<div class="loading-spinner">Không có kết quả nào.</div>';
    return;
  }

  appState.similarResults.forEach((r) => {
    const card = document.createElement("div");
    const isSelected = appState.selectedSimilarKeys.has(r.keyframe_path);
    card.className = "similar-card" + (isSelected ? " selected" : "");
    card.dataset.key = r.keyframe_path;

    const refTag = r.is_query_image ? '<span style="color:var(--electric-cyan); font-weight:bold;">[GỐC] </span>' : "";

    card.innerHTML = `
      <div class="sim-img-wrap">
        <img src="${r.image_url}" alt="${r.scene_id}" loading="lazy">
        <span class="sim-score-badge">${r.similarity_percent}%</span>
      </div>
      <div class="sim-card-info">
        <span class="v-tag">${refTag}${r.batch}_${r.video_id}</span>
        <span>#${r.frame_id}</span>
      </div>
    `;

    card.addEventListener("click", () => {
      if (appState.selectedSimilarKeys.has(r.keyframe_path)) {
        appState.selectedSimilarKeys.delete(r.keyframe_path);
        card.classList.remove("selected");
      } else {
        appState.selectedSimilarKeys.add(r.keyframe_path);
        card.classList.add("selected");
      }
      updateSimilarSelectionCounts();
    });

    elements.similarResultsGrid.appendChild(card);
  });
}

async function applySimilarFilter() {
  const selectedKeys = Array.from(appState.selectedSimilarKeys);
  if (selectedKeys.length === 0) {
    showToast("Chưa chọn ảnh nào để lọc!", "danger");
    return;
  }

  const items = [];
  selectedKeys.forEach((key) => {
    const target = appState.similarResults.find((r) => r.keyframe_path === key);
    if (target) {
      items.push({
        batch: target.batch,
        video_id: target.video_id,
        filename: target.filename,
        frame_id: target.frame_id,
        keyframe_path: target.keyframe_path,
        reason: `visual_duplicate_sim_${target.similarity_percent}%`,
      });
    }
  });

  const res = await postFilterUpdate("add", items, "visual_duplicate");
  if (res) {
    showToast(`Đã thêm ${items.length} ảnh tương đồng vào danh sách lọc!`, "success");
    elements.similarModal.classList.remove("open");
    
    // Refresh all sidebar batch & video counts
    await refreshAllSidebarCounts();
    
    // Refresh current video if it was modified
    if (appState.currentBatch && appState.currentVideoId) {
      fetchImages(appState.currentBatch, appState.currentVideoId);
    }
  }
}

async function refreshAllSidebarCounts() {
  await fetchBatches();
  
  // For each open batch group, refresh its video list
  const openGroups = document.querySelectorAll(".batch-group.open");
  for (const group of openGroups) {
    const batch = group.dataset.batch;
    const vListEl = document.getElementById(`vlist-${batch}`);
    if (batch && vListEl) {
      const videos = await fetchVideos(batch);
      renderVideoItems(batch, videos, vListEl);
    }
  }
}


// =============================================================================
// Lightbox Preview
// =============================================================================

function openLightbox(img) {
  elements.lightboxImg.src = img.image_url;
  elements.lightboxTitle.textContent = `${appState.currentBatch}_${appState.currentVideoId} / Frame #${img.frame_id}`;
  elements.lightboxDetails.textContent = `Scene ID: ${img.scene_id} | Timestamp: ${img.timestamp_mid.toFixed(2)}s | File: ${img.filename}`;
  elements.lightboxModal.classList.add("open");
}


// =============================================================================
// Toast Helpers
// =============================================================================

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === "success" ? "✅" : "⚠️"}</span><span>${message}</span>`;
  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(50px)";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}


// =============================================================================
// Event Listeners & Initialization
// =============================================================================

function setupEventListeners() {
  // Refresh Button
  elements.btnRefresh.addEventListener("click", () => {
    fetchBatches();
    if (appState.currentBatch && appState.currentVideoId) {
      fetchImages(appState.currentBatch, appState.currentVideoId);
    }
  });

  // Export JSON
  elements.btnExportJson.addEventListener("click", () => {
    window.location.href = "/api/filter/export";
  });

  // Video Search in sidebar
  elements.videoSearchInput.addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    document.querySelectorAll(".video-item").forEach((el) => {
      const vid = el.dataset.videoId.toLowerCase();
      const b = el.dataset.batch.toLowerCase();
      const full = `${b}_${vid}`;
      if (vid.includes(q) || b.includes(q) || full.includes(q)) {
        el.style.display = "flex";
      } else {
        el.style.display = "none";
      }
    });
  });

  elements.clearVideoSearch.addEventListener("click", () => {
    elements.videoSearchInput.value = "";
    document.querySelectorAll(".video-item").forEach((el) => (el.style.display = "flex"));
  });

  // Filter Tabs
  elements.filterTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      elements.filterTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      appState.currentFilterView = tab.dataset.filter;
      renderKeyframeGrid();
    });
  });

  // Bulk video actions
  elements.btnSelectAllVideo.addEventListener("click", () => bulkFilterCurrentVideo("add"));
  elements.btnDeselectAllVideo.addEventListener("click", () => bulkFilterCurrentVideo("remove"));

  // Prev / Next Video buttons
  elements.btnPrevVideo.addEventListener("click", navigateVideoAdjacent.bind(null, -1));
  elements.btnNextVideo.addEventListener("click", navigateVideoAdjacent.bind(null, 1));

  // Similar Modal controls
  elements.btnCloseSimilarModal.addEventListener("click", () => elements.similarModal.classList.remove("open"));
  elements.btnCancelSimilar.addEventListener("click", () => elements.similarModal.classList.remove("open"));
  
  elements.simThresholdSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    elements.thresholdDisplay.textContent = `${val}%`;
    autoSelectByThreshold(parseFloat(val) / 100.0);
    renderSimilarResults();
  });

  if (elements.simZoomSlider) {
    elements.simZoomSlider.addEventListener("input", (e) => {
      const val = e.target.value;
      elements.zoomDisplay.textContent = `${val} cột`;
      elements.similarResultsGrid.className = `similar-grid cols-${val}`;
    });
  }

  elements.chkKeepRefImage.addEventListener("change", () => {
    const threshold = parseFloat(elements.simThresholdSlider.value) / 100.0;
    autoSelectByThreshold(threshold);
    renderSimilarResults();
  });

  elements.btnQuick95.addEventListener("click", () => {
    elements.simThresholdSlider.value = 95;
    elements.thresholdDisplay.textContent = "95%";
    autoSelectByThreshold(0.95);
    renderSimilarResults();
  });

  elements.btnQuick90.addEventListener("click", () => {
    elements.simThresholdSlider.value = 90;
    elements.thresholdDisplay.textContent = "90%";
    autoSelectByThreshold(0.90);
    renderSimilarResults();
  });

  elements.btnQuickAll.addEventListener("click", () => {
    appState.similarResults.forEach((r) => {
      if (r.is_query_image && elements.chkKeepRefImage.checked) return;
      appState.selectedSimilarKeys.add(r.keyframe_path);
    });
    updateSimilarSelectionCounts();
    renderSimilarResults();
  });

  elements.btnQuickNone.addEventListener("click", () => {
    appState.selectedSimilarKeys.clear();
    updateSimilarSelectionCounts();
    renderSimilarResults();
  });

  elements.btnApplySimilarFilter.addEventListener("click", applySimilarFilter);

  // Lightbox
  elements.btnCloseLightbox.addEventListener("click", () => elements.lightboxModal.classList.remove("open"));
  elements.lightboxModal.addEventListener("click", (e) => {
    if (e.target === elements.lightboxModal) elements.lightboxModal.classList.remove("open");
  });

  // Density Switcher
  const densityBtns = document.querySelectorAll(".btn-density");
  const savedCols = localStorage.getItem("aic_filter_grid_cols") || "6";
  setGridDensity(savedCols);

  densityBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const cols = btn.dataset.cols;
      setGridDensity(cols);
      localStorage.setItem("aic_filter_grid_cols", cols);
    });
  });

  // Global Keyboard Shortcuts
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      elements.similarModal.classList.remove("open");
      elements.lightboxModal.classList.remove("open");
    } else if (e.key === "ArrowLeft" && !e.target.matches("input, textarea")) {
      navigateVideoAdjacent(-1);
    } else if (e.key === "ArrowRight" && !e.target.matches("input, textarea")) {
      navigateVideoAdjacent(1);
    }
  });
}

function setGridDensity(cols) {
  document.querySelectorAll(".btn-density").forEach((b) => {
    b.classList.toggle("active", b.dataset.cols === cols);
  });
  elements.keyframeGrid.className = `keyframe-grid cols-${cols}`;
}

function navigateVideoAdjacent(direction) {
  const visibleItems = Array.from(document.querySelectorAll(".video-item")).filter(
    (el) => el.style.display !== "none"
  );
  if (visibleItems.length === 0) return;

  const currentIndex = visibleItems.findIndex(
    (el) => el.dataset.batch === appState.currentBatch && el.dataset.videoId === appState.currentVideoId
  );

  let nextIndex = currentIndex + direction;
  if (nextIndex < 0) nextIndex = visibleItems.length - 1;
  if (nextIndex >= visibleItems.length) nextIndex = 0;

  const nextEl = visibleItems[nextIndex];
  selectVideo(nextEl.dataset.batch, nextEl.dataset.videoId);
}

// Entry point
window.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  fetchBatches();
});
