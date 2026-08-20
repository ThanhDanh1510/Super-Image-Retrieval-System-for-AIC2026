/* =============================================================================
   🕷️ SPIDER-MAN THEME APP LOGIC (AIC2026 SUPER RETRIEVAL)
   ============================================================================= */

const API_BASE = "http://localhost:8000";

// App State
const state = {
  mode: "kis", // kis, vqa, trake
  searchMode: "all", // all, embedding, ocr, asr
  query: "",
  topK: 50,
  useViranker: true,
  useJinaM0: false,
  selectedItems: new Map(), // key -> CandidateResult or TrakeChain
  batchQueries: new Map(),  // queryId -> { queryId, mode, count, csvString, timestamp, items, queryText, searchMode, resultsCache }
  queryCache: new Map(),    // cacheKey -> { results, duration, count, timestamp }
  rawResults: [],
  currentResultsList: [],   // Flattened list for sequential Lightbox navigation
  currentLightboxIndex: -1, // Current index in Lightbox
  selectedFile: null,
  selectedImageBase64: null,
  hoveredCand: null,
};

// DOM Elements
const queryInput = document.getElementById("query-input");
const searchBtn = document.getElementById("search-btn");
const resultsGrid = document.getElementById("results-grid");
const statusBar = document.getElementById("status-text");
const speedTag = document.getElementById("speed-tag");
const selectedCount = document.getElementById("selected-count");
const selectedList = document.getElementById("selected-list");
const drawer = document.getElementById("submission-drawer");

// Initialize Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupControls();
  setupSearch();
  setupDrawer();
  setupVoice();
  setupKeyboardShortcuts();
  updateUIForMode();
});

// --- Tab Setup ---
function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      updateUIForMode();
      updateDrawer();
    });
  });
}

function updateUIForMode() {
  const numEventsGroup = document.getElementById("num-events-group");
  const vqaBulkBox = document.getElementById("vqa-bulk-box");

  if (state.mode === "trake") {
    queryInput.placeholder = "Gõ chuỗi sự kiện theo thứ tự... (VD: người đàn ông bước vào xe -> xe chạy trên đường -> dừng lại ở quán ăn)";
    if (numEventsGroup) numEventsGroup.style.display = "flex";
    if (vqaBulkBox) vqaBulkBox.style.display = "none";
  } else if (state.mode === "vqa") {
    queryInput.placeholder = "Gõ câu hỏi VQA... (VD: Người lái xe mặc áo màu gì? / Có bao nhiêu người trong quán?)";
    if (numEventsGroup) numEventsGroup.style.display = "none";
    if (vqaBulkBox) vqaBulkBox.style.display = "flex";
  } else {
    queryInput.placeholder = "Gõ câu miêu tả tìm kiếm... (VD: xe máy màu vàng, tiệm vàng Lê Tứ)";
    if (numEventsGroup) numEventsGroup.style.display = "none";
    if (vqaBulkBox) vqaBulkBox.style.display = "none";
  }
  updateQueryIdInputForMode();
}

// --- Search Controls Setup ---
function setupControls() {
  // Branch Buttons
  document.querySelectorAll(".branch-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".branch-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.searchMode = btn.dataset.branch;
      statusBar.textContent = `🎯 Đã đổi sang nhánh tìm kiếm: ${btn.textContent.trim()}`;
    });
  });

  // Top-K
  const topKInput = document.getElementById("top-k-input");
  if (topKInput) {
    state.topK = parseInt(topKInput.value, 10) || 50;
    topKInput.addEventListener("input", (e) => {
      state.topK = parseInt(e.target.value, 10) || 50;
    });
    topKInput.addEventListener("change", (e) => {
      state.topK = parseInt(e.target.value, 10) || 50;
    });
  }

  // Card Density Slider
  const densitySlider = document.getElementById("density-slider");
  const densityVal = document.getElementById("density-val");
  if (densitySlider) {
    const updateSliderFill = (slider) => {
      const min = slider.min || 140;
      const max = slider.max || 320;
      const val = slider.value;
      const percent = ((val - min) / (max - min)) * 100;
      slider.style.background = `linear-gradient(to right, #ffd700 ${percent}%, rgba(255, 255, 255, 0.1) ${percent}%)`;
    };

    updateSliderFill(densitySlider);

    densitySlider.addEventListener("input", (e) => {
      const pxVal = e.target.value;
      resultsGrid.style.setProperty("--card-min-width", `${pxVal}px`);
      if (densityVal) densityVal.textContent = `${pxVal}px`;
      updateSliderFill(e.target);
    });
  }

  // Image upload
  const imgInput = document.getElementById("image-upload-input");
  if (imgInput) {
    imgInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        state.selectedFile = file;
        const reader = new FileReader();
        reader.onload = (evt) => {
          state.selectedImageBase64 = evt.target.result.split(",")[1];
          document.getElementById("chip-filename").textContent = file.name;
          document.getElementById("image-preview-chip").style.display = "inline-flex";
          statusBar.textContent = `🖼️ Đã chọn ảnh: ${file.name}. Bấm TÌM KIẾM hoặc Enter để tìm ảnh tương đồng.`;
        };
        reader.readAsDataURL(file);
      }
    });
  }

  // Drag and drop image onto search panel
  const searchPanel = document.querySelector(".search-panel");
  if (searchPanel) {
    searchPanel.addEventListener("dragover", (e) => {
      e.preventDefault();
      searchPanel.classList.add("drag-over");
    });
    searchPanel.addEventListener("dragleave", () => {
      searchPanel.classList.remove("drag-over");
    });
    searchPanel.addEventListener("drop", (e) => {
      e.preventDefault();
      searchPanel.classList.remove("drag-over");
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith("image/")) {
          state.selectedFile = file;
          const reader = new FileReader();
          reader.onload = (evt) => {
            state.selectedImageBase64 = evt.target.result.split(",")[1];
            document.getElementById("chip-filename").textContent = file.name;
            document.getElementById("image-preview-chip").style.display = "inline-flex";
            statusBar.textContent = `🖼️ Đã thả ảnh: ${file.name}. Bấm TÌM KIẾM để tìm ảnh tương đồng.`;
          };
          reader.readAsDataURL(file);
        }
      }
    });
  }

  const removeImgBtn = document.getElementById("remove-img-btn");
  if (removeImgBtn) {
    removeImgBtn.addEventListener("click", () => {
      state.selectedFile = null;
      state.selectedImageBase64 = null;
      if (imgInput) imgInput.value = "";
      document.getElementById("image-preview-chip").style.display = "none";
      statusBar.textContent = "🗑️ Đã bỏ chọn ảnh.";
    });
  }

  // Reranker Checkbox Switches (ViRanker Text & Jina M0 Visual)
  function updateRerankerStatus() {
    const parts = [];
    if (state.useViranker) parts.push("ViRanker Text");
    if (state.useJinaM0) parts.push("Jina M0 Visual");
    if (parts.length === 0) {
      statusBar.textContent = "⚙️ Reranker: Đã TẮT (Chế độ tìm kiếm siêu tốc <50ms)";
    } else {
      statusBar.textContent = `⚙️ Reranker: Đang BẬT [${parts.join(" + ")}]`;
    }
  }

  const virankerCheckbox = document.getElementById("use-viranker-checkbox");
  if (virankerCheckbox) {
    virankerCheckbox.checked = state.useViranker;
    virankerCheckbox.addEventListener("change", (e) => {
      state.useViranker = e.target.checked;
      updateRerankerStatus();
    });
  }

  const jinaCheckbox = document.getElementById("use-jina-checkbox");
  if (jinaCheckbox) {
    jinaCheckbox.checked = state.useJinaM0;
    jinaCheckbox.addEventListener("change", (e) => {
      state.useJinaM0 = e.target.checked;
      updateRerankerStatus();
    });
  }
}

let isVideoInteractMode = false;

// --- Keyboard Shortcuts (Lightbox Navigation, Video Space Control & Copy on Hover) ---
function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    const modal = document.getElementById("lightbox-modal");
    const isModalOpen = modal && modal.style.display !== "none";

    if (isModalOpen) {
      // Escape: Đóng modal
      if (e.key === "Escape") {
        modal.style.display = "none";
        closeLightboxVideo();
        return;
      }

      // Phím Space: Toggle Play/Pause video và kích hoạt Chế độ Tương Tác Video (Ngăn chặn triệt để cuộn trang & kích hoạt nút)
      if (e.code === "Space" || e.key === " " || e.key === "Spacebar") {
        if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
        e.preventDefault();
        e.stopPropagation();

        // Blur bất kỳ nút bấm nào đang được focus để tránh re-activation
        if (document.activeElement && typeof document.activeElement.blur === "function") {
          document.activeElement.blur();
        }

        isVideoInteractMode = true;
        if (modalVideoEl.paused) {
          modalVideoEl.play().catch(() => {});
          showToast("▶️ Video đang phát (Phím ←/→ tua 3s)");
        } else {
          modalVideoEl.pause();
          showToast("⏸️ Video tạm dừng (Phím ←/→ tua 3s)");
        }
        return;
      }

      // Khi đã bấm Space (isVideoInteractMode === true): Left/Right Arrow tua 3 giây
      if (isVideoInteractMode) {
        if (e.key === "ArrowLeft" || e.code === "KeyA") {
          if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
          e.preventDefault();
          modalVideoEl.currentTime = Math.max(0, modalVideoEl.currentTime - 3);
          updateVideoInfoLabel();
          showToast(`⏪ Tua lùi 3s (f${currentVideoFrame()})`);
          return;
        }
        if (e.key === "ArrowRight" || e.code === "KeyD") {
          if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
          e.preventDefault();
          modalVideoEl.currentTime = Math.min(modalVideoEl.duration || 999999, modalVideoEl.currentTime + 3);
          updateVideoInfoLabel();
          showToast(`⏩ Tua tiến 3s (f${currentVideoFrame()})`);
          return;
        }
      } else {
        // Khi chưa bấm Space: Left/Right Arrow dùng để chuyển qua lại giữa các ảnh kết quả
        if (e.key === "ArrowLeft" || e.code === "KeyA") {
          if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
          e.preventDefault();
          showLightboxAt(state.currentLightboxIndex - 1);
          return;
        }
        if (e.key === "ArrowRight" || e.code === "KeyD") {
          if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
          e.preventDefault();
          showLightboxAt(state.currentLightboxIndex + 1);
          return;
        }
      }
    }

    if ((e.key === "c" || e.key === "C") && state.hoveredCand) {
      // Don't trigger if user is actively typing in a text field
      if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
      e.preventDefault();
      copyToClipboard(state.hoveredCand);
    }
  });
}

// --- Copy Helper & Toast Notification ---
function copyToClipboard(cand) {
  if (!cand) return;
  const str = `${cand.video_id},${cand.frame_id}`;
  navigator.clipboard.writeText(str).then(() => {
    showToast(`📋 Đã copy: ${str}`);
  }).catch(() => {
    // Fallback if clipboard API is restricted
    showToast(`📋 Đã copy: ${str}`);
  });
}

function showToast(text) {
  const oldToast = document.querySelector(".copy-toast");
  if (oldToast) oldToast.remove();
  const toast = document.createElement("div");
  toast.className = "copy-toast";
  toast.textContent = text;
  document.body.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 2000);
}

// --- Voice Search Setup ---
function setupVoice() {
  const voiceBtn = document.getElementById("voice-btn");
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'vi-VN';

    voiceBtn.addEventListener("click", () => {
      recognition.start();
      statusBar.textContent = "🎙️ Đang nghe giọng nói của bạn...";
    });

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      queryInput.value = transcript;
      statusBar.textContent = `🎙️ Nhận diện giọng nói: "${transcript}"`;
      triggerSearch();
    };
  } else if (voiceBtn) {
    voiceBtn.style.display = "none";
  }
}

// --- Search Handler ---
function setupSearch() {
  searchBtn.addEventListener("click", triggerSearch);
  queryInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") triggerSearch();
  });
}

async function triggerSearch() {
  const queryText = queryInput.value.trim();
  state.lastQuery = queryText;

  if (!queryText && !state.selectedImageBase64 && !state.selectedFile) {
    statusBar.textContent = "⚠️ Vui lòng nhập câu tìm kiếm hoặc tải ảnh lên!";
    return;
  }

  const startTime = performance.now();
  speedTag.style.display = "none";

  // CASE 1: Image-to-Image Search (Nếu có ảnh được tải lên / chọn)
  if (state.selectedFile || state.selectedImageBase64) {
    statusBar.textContent = "🕸️ Đang quét toàn bộ vector để tìm ảnh tương đồng (Image-to-Image Search)...";
    resultsGrid.innerHTML = `<div class="empty-state">🕷️ Đang mã hóa vector ảnh CLIP và truy vấn Qdrant...</div>`;

    try {
      const formData = new FormData();
      if (state.selectedFile) {
        formData.append("file", state.selectedFile, state.selectedFile.name);
      } else {
        // Build blob from base64 string
        const byteCharacters = atob(state.selectedImageBase64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: "image/webp" });
        formData.append("file", blob, "query_image.webp");
      }

      const resp = await fetch(`${API_BASE}/api/v1/similar-image?top_k=${state.topK}&use_viranker=${state.useViranker}&use_jina_m0=${state.useJinaM0}`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      resultsList = data.results || [];

      const duration = ((performance.now() - startTime) / 1000).toFixed(2);
      state.rawResults = resultsList;
      renderResults(state.rawResults);

      statusBar.textContent = `✅ Tìm thấy ${state.rawResults.length} keyframe tương đồng với ảnh mẫu!`;
      speedTag.textContent = `${duration}s`;
      speedTag.style.display = "inline-block";
      return;
    } catch (err) {
      statusBar.textContent = `❌ Lỗi khi tìm kiếm ảnh: ${err.message}`;
      resultsGrid.innerHTML = `<div class="empty-state">Lỗi kết nối server: ${err.message}</div>`;
      return;
    }
  }

  // CASE 2: Text / Multimodal KIS / VQA / TRAKE Search
  statusBar.textContent = `🕸️ Spidey đang tìm kiếm (Nhánh: ${state.searchMode.toUpperCase()})...`;
  resultsGrid.innerHTML = `<div class="empty-state">🕷️ Đang thực hiện tìm kiếm & Rerank đa phương thức...</div>`;

  try {
    let endpoint = `${API_BASE}/api/v1/${state.mode}`;
    let payload = {
      input_type: "text",
      query: queryText,
      search_mode: state.searchMode,
      top_k: state.topK,
      user_choices: [],
      use_viranker: state.useViranker,
      use_jina_m0: state.useJinaM0
    };

    if (state.mode === "vqa") {
      payload.question = queryText;
    }

    if (state.mode === "trake") {
      const numEvents = parseInt(document.getElementById("num-events-input")?.value || "3", 10);
      payload.num_events = Math.max(2, Math.min(10, numEvents));
    }

    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    resultsList = data.results || [];

    const duration = ((performance.now() - startTime) / 1000).toFixed(2);
    state.rawResults = resultsList;
    renderResults(state.rawResults);

    // Save to queryCache
    const cacheKey = `${state.mode}_${state.searchMode}_${queryText}`;
    state.queryCache.set(cacheKey, {
      results: [...state.rawResults],
      duration: duration,
      count: state.rawResults.length,
      timestamp: Date.now(),
    });

    statusBar.textContent = `✅ Tìm thấy ${state.rawResults.length} kết quả (Nhánh: ${state.searchMode.toUpperCase()})`;
    speedTag.textContent = `${duration}s`;
    speedTag.style.display = "inline-block";

  } catch (err) {
    statusBar.textContent = `❌ Lỗi khi tìm kiếm: ${err.message}`;
    resultsGrid.innerHTML = `<div class="empty-state">Lỗi kết nối server: ${err.message}</div>`;
  }
}

// --- Render High Density Grid Results ---
function renderResults(results) {
  if (!results || results.length === 0) {
    resultsGrid.innerHTML = `<div class="empty-state">Không tìm thấy kết quả nào phù hợp.</div>`;
    state.currentResultsList = [];
    return;
  }

  resultsGrid.innerHTML = "";

  // TRAKE mode: mỗi item là 1 chain (array) của candidates
  const isTrake = state.mode === "trake" && Array.isArray(results[0]);

  if (isTrake) {
    state.currentResultsList = results.flat().filter(Boolean);
    // TRAKE: render từng chain trên 1 hàng
    results.forEach((chain, chainIdx) => {
      if (!Array.isArray(chain) || chain.length === 0) return;
      renderTrakeChain(chain, chainIdx);
    });
  } else {
    state.currentResultsList = results || [];
    // KIS/VQA: render từng ảnh riêng lẻ
    results.forEach((candidate, idx) => {
      renderSingleCard(candidate);
    });
  }
}

// --- Render 1 TRAKE chain (chuỗi ảnh nối tiếp) ---
function renderTrakeChain(chain, chainIdx) {
  const avgScore = chain.reduce((sum, c) => sum + c.score, 0) / chain.length;
  const chainKey = "trake_" + chain[0].video_id + "_" + chain.map((c) => c.frame_id).join("_");
  const isChainSelected = state.selectedItems.has(chainKey);

  const chainContainer = document.createElement("div");
  chainContainer.className = "trake-chain";

  let imagesHtml = "";
  chain.forEach((candidate, eventIdx) => {
    const imgUrl = `${API_BASE}/api/v1/image/${candidate.video_id}/${candidate.frame_id}`;
    const itemKey = `${candidate.video_id}_${candidate.frame_id}`;

    imagesHtml += `
      <div class="trake-event" data-key="${itemKey}">
        <div class="trake-img-wrapper">
          <img src="${imgUrl}" alt="${candidate.scene_id}" loading="lazy" />
          <span class="event-badge">E${eventIdx + 1}</span>
        </div>
        <div class="trake-event-info">
          <span class="frame-label">f${candidate.frame_id}</span>
          <span class="video-label">${candidate.video_id}</span>
        </div>
      </div>
    `;

    // Thêm mũi tên giữa các events (trừ event cuối)
    if (eventIdx < chain.length - 1) {
      imagesHtml += `<div class="trake-arrow">→</div>`;
    }
  });

  chainContainer.innerHTML = `
    <div class="trake-chain-header">
      <span class="chain-index">#${chainIdx + 1}</span>
      <span class="chain-score">${(avgScore * 100).toFixed(0)}%</span>
      <span class="chain-video">${chain[0].video_id}</span>
      <button class="trake-chain-select-btn ${isChainSelected ? 'selected' : ''}" data-chain-key="${chainKey}">
        ${isChainSelected ? '✓ Đã chọn chuỗi này' : `➕ Chọn cả chuỗi (${chain.length} events)`}
      </button>
    </div>
    <div class="trake-chain-images">
      ${imagesHtml}
    </div>
  `;

  // Event listener cho nút chọn cả chuỗi TRAKE
  const chainSelectBtn = chainContainer.querySelector(".trake-chain-select-btn");
  if (chainSelectBtn) {
    chainSelectBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSelectTrakeChain(chain, chainKey, chainSelectBtn);
    });
  }

  // Event listeners cho từng event trong chain (Lightbox & Hover)
  chainContainer.querySelectorAll(".trake-event").forEach((eventEl) => {
    const key = eventEl.dataset.key;
    const candidate = chain.find((c) => `${c.video_id}_${c.frame_id}` === key);
    if (!candidate) return;

    const imgUrl = `${API_BASE}/api/v1/image/${candidate.video_id}/${candidate.frame_id}`;

    // Lightbox on click
    eventEl.querySelector("img").addEventListener("click", () => showLightbox(candidate, imgUrl));

    // Hover tracking
    eventEl.addEventListener("mouseenter", () => { state.hoveredCand = candidate; });
    eventEl.addEventListener("mouseleave", () => { state.hoveredCand = null; });
  });

  resultsGrid.appendChild(chainContainer);
}

// --- Chọn / Bỏ chọn toàn bộ 1 chuỗi TRAKE ---
function toggleSelectTrakeChain(chain, chainKey, btn) {
  if (state.selectedItems.has(chainKey)) {
    state.selectedItems.delete(chainKey);
    if (btn) {
      btn.classList.remove("selected");
      btn.textContent = `➕ Chọn cả chuỗi (${chain.length} events)`;
    }
  } else {
    state.selectedItems.set(chainKey, {
      type: "trake_chain",
      video_id: chain[0].video_id,
      frame_ids: chain.map((c) => c.frame_id),
      chain_events: chain.map((c) => ({ ...c })), // Deep copy để việc sửa đổi giỏ nộp bài không làm đổi kết quả tìm kiếm gốc
      score: chain.reduce((sum, c) => sum + c.score, 0) / chain.length,
    });
    if (btn) {
      btn.classList.add("selected");
      btn.textContent = "✓ Đã chọn chuỗi này";
    }
  }
  updateDrawer();
}

// --- Render 1 single card (KIS/VQA) ---
function renderSingleCard(candidate) {
  const itemKey = `${candidate.video_id}_${candidate.frame_id}`;
  const isSelected = state.selectedItems.has(itemKey);
  const imgUrl = `${API_BASE}/api/v1/image/${candidate.video_id}/${candidate.frame_id}`;
  const scorePct = (candidate.score * 100).toFixed(1);

  const card = document.createElement("div");
  card.className = "kf-card";

  card.innerHTML = `
    <div class="kf-img-wrapper">
      <img src="${imgUrl}" alt="${candidate.scene_id}" loading="lazy" />
      <span class="score-badge">${scorePct}%</span>
      <button class="select-checkbox-btn ${isSelected ? 'selected' : ''}" data-key="${itemKey}">
        ${isSelected ? '✓' : '+'}
      </button>
      <div class="card-overlay-actions">
        <button class="copy-btn-overlay" title="Copy <video_id,frame_id> (Ctrl+C)">📋 Copy</button>
        <button class="similar-btn-overlay" title="Tìm các keyframe tương đồng với ảnh này">🔍 Tương tự</button>
      </div>
    </div>
    <div class="kf-details">
      <div class="kf-title">
        <span>${candidate.video_id}</span>
        <span class="ts-label" title="${candidate.video_id},${candidate.frame_id}">f${candidate.frame_id}</span>
      </div>
      ${candidate.ocr_text ? `<div class="ocr-pill" title="${candidate.ocr_text}">🔤 ${highlightMatchedText(candidate.ocr_text, state.lastQuery || (queryInput ? queryInput.value : ''))}</div>` : ''}
    </div>
  `;

  // Mouseover tracking for hotkey C copy
  card.addEventListener("mouseenter", () => { state.hoveredCand = candidate; });
  card.addEventListener("mouseleave", () => { state.hoveredCand = null; });

  // Lightbox image preview on click
  card.querySelector("img").addEventListener("click", () => showLightbox(candidate, imgUrl));

  // Copy Button Overlay Click
  card.querySelector(".copy-btn-overlay").addEventListener("click", (e) => {
    e.stopPropagation();
    copyToClipboard(candidate);
  });

  // Similar Image Search Button Click
  const simBtn = card.querySelector(".similar-btn-overlay");
  if (simBtn) {
    simBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      searchSimilarByCandidate(candidate);
    });
  }

  // Selection Checkbox
  const selectBtn = card.querySelector(".select-checkbox-btn");
  selectBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleSelect(itemKey, candidate, selectBtn);
  });

  resultsGrid.appendChild(card);
}

// --- Quick 1-Click Similar Image Search ---
async function searchSimilarByCandidate(cand) {
  statusBar.textContent = `🖼️ Đang tìm ảnh tương tự với ${cand.video_id} #f${cand.frame_id}...`;
  try {
    const imgUrl = `${API_BASE}/api/v1/image/${cand.video_id}/${cand.frame_id}`;
    const res = await fetch(imgUrl);
    if (!res.ok) throw new Error("Không thể tải file ảnh keyframe mẫu");
    const blob = await res.blob();
    state.selectedFile = new File([blob], `${cand.video_id}_${cand.frame_id}.webp`, { type: blob.type || "image/webp" });
    state.selectedImageBase64 = null;

    const chipEl = document.getElementById("image-preview-chip");
    const chipNameEl = document.getElementById("chip-filename");
    if (chipNameEl) chipNameEl.textContent = `Ảnh mẫu: ${cand.video_id} #f${cand.frame_id}`;
    if (chipEl) chipEl.style.display = "inline-flex";

    // Tự động đóng Lightbox nếu đang mở và thực hiện tìm kiếm tức thì
    const modal = document.getElementById("lightbox-modal");
    if (modal && modal.style.display !== "none") {
      modal.style.display = "none";
      closeLightboxVideo();
    }

    triggerSearch();
  } catch (err) {
    statusBar.textContent = `❌ Lỗi nạp ảnh: ${err.message}`;
  }
}

// --- Selection Drawer & Collapsible Sidebar ---
function toggleSelect(itemKey, candidate, btn) {
  if (state.selectedItems.has(itemKey)) {
    state.selectedItems.delete(itemKey);
    btn.classList.remove("selected");
    btn.textContent = "+";
  } else {
    state.selectedItems.set(itemKey, candidate);
    btn.classList.add("selected");
    btn.textContent = "✓";
  }
  updateDrawer();
}

function updateDrawer() {
  selectedCount.textContent = state.selectedItems.size;

  if (state.selectedItems.size === 0) {
    selectedList.innerHTML = `<div class="empty-state">Chưa chọn bức ảnh nào. Click dấu (+) trên ảnh để chọn nộp.</div>`;
    return;
  }

  selectedList.innerHTML = "";
  state.selectedItems.forEach((cand, key) => {
    // 1. Chuỗi sự kiện TRAKE
    if (cand.type === "trake_chain") {
      const div = document.createElement("div");
      div.className = "trake-cart-item";
      let thumbsHtml = "";
      cand.chain_events.forEach((ev, idx) => {
        const imgUrl = ev.custom_thumb || `${API_BASE}/api/v1/image/${ev.video_id}/${ev.frame_id}`;
        thumbsHtml += `
          <div class="trake-cart-thumb" title="${ev.video_id} f${ev.frame_id}">
            <img src="${imgUrl}" alt="f${ev.frame_id}" onerror="this.src='/static/placeholder.webp'" />
            <span>f${ev.frame_id}</span>
          </div>
        `;
        if (idx < cand.chain_events.length - 1) {
          thumbsHtml += `<span class="trake-cart-arrow">→</span>`;
        }
      });
      div.innerHTML = `
        <div class="trake-cart-header">
          <span>🎞️ ${cand.video_id} (${cand.frame_ids.length} events)</span>
          <button onclick="removeItem('${key}')" style="background:none; border:none; color:var(--spider-red); cursor:pointer; font-weight:bold; font-size:0.9rem;" title="Xóa chuỗi này">✕</button>
        </div>
        <div class="trake-cart-thumbs">
          ${thumbsHtml}
        </div>
      `;
      selectedList.appendChild(div);
      return;
    }

    // 2. Khung hình đơn lẻ (KIS / VQA)
    const div = document.createElement("div");
    div.className = "selected-item";
    const imgUrl = cand.custom_thumb || `${API_BASE}/api/v1/image/${cand.video_id}/${cand.frame_id}`;
    
    div.innerHTML = `
      <img src="${imgUrl}" alt="${cand.scene_id || cand.video_id}" onerror="this.src='/static/placeholder.webp'" />
      <div style="flex:1; min-width: 0;">
        <div style="font-weight:bold; font-size:0.85rem;">${cand.video_id}</div>
        <div style="font-size:0.75rem; color:var(--spider-blue); font-family:var(--font-code); font-weight:600;">f${cand.frame_id}</div>
        ${state.mode === "vqa" ? `
          <div class="vqa-item-box">
            <input type="text" class="vqa-ans-input" data-key="${key}" placeholder="Đáp án riêng cho frame này..." value="${cand.user_answer != null ? cand.user_answer : (cand.answer || '')}" maxlength="100" />
          </div>
        ` : ''}
      </div>
      <button onclick="copyItemKey('${cand.video_id}', ${cand.frame_id})" style="background:none; border:none; color:var(--spider-blue); cursor:pointer; font-size:1rem; margin-right:4px;" title="Copy">📋</button>
      <button onclick="removeItem('${key}')" style="background:none; border:none; color:var(--spider-red); cursor:pointer; font-weight:bold;">✕</button>
    `;
    selectedList.appendChild(div);
  });

  // Attach event listener for VQA inputs
  document.querySelectorAll(".vqa-ans-input").forEach((input) => {
    input.addEventListener("input", (e) => {
      const k = e.target.dataset.key;
      const item = state.selectedItems.get(k);
      if (item) {
        item.user_answer = e.target.value;
      }
    });
  });
}

window.copyItemKey = function(video_id, frame_id) {
  copyToClipboard({ video_id, frame_id });
};

window.removeItem = function(key) {
  state.selectedItems.delete(key);
  updateDrawer();
  renderResults(state.rawResults);
};

function setupDrawer() {
  const sidebar = document.getElementById("submission-sidebar");
  const toggleBtn = document.getElementById("toggle-sidebar-btn");
  const arrowSpan = document.getElementById("basket-arrow");

  function toggleSidebar(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    if (!sidebar) return;
    const isCollapsed = sidebar.classList.toggle("collapsed");
    if (arrowSpan) {
      arrowSpan.textContent = isCollapsed ? "▸" : "▾";
    }
  }

  if (toggleBtn) {
    toggleBtn.onclick = toggleSidebar;
  }

  // VQA Bulk Answer Apply
  const vqaApplyAllBtn = document.getElementById("vqa-apply-all-btn");
  if (vqaApplyAllBtn) {
    vqaApplyAllBtn.onclick = applyVqaAnswerToAll;
  }

  // Query ID input sync
  const queryIdInput = document.getElementById("query-id-input");
  const btnQueryIdLabel = document.getElementById("btn-query-id-label");
  if (queryIdInput) {
    queryIdInput.addEventListener("input", (e) => {
      let val = e.target.value.trim();
      if (!val) val = getDefaultQueryId();
      if (btnQueryIdLabel) {
        btnQueryIdLabel.textContent = `${val}.csv`;
      }
    });
  }

  // Button actions
  const exportBtn = document.getElementById("export-csv-btn");
  if (exportBtn) {
    exportBtn.onclick = exportSingleCSV;
  }

  const addToBatchBtn = document.getElementById("add-to-batch-btn");
  if (addToBatchBtn) {
    addToBatchBtn.onclick = addCurrentCartToBatch;
  }

  const packageZipBtn = document.getElementById("package-zip-btn");
  if (packageZipBtn) {
    packageZipBtn.onclick = packageSubmissionZip;
  }

  const clearBatchBtn = document.getElementById("clear-batch-btn");
  if (clearBatchBtn) {
    clearBatchBtn.onclick = clearAllBatchQueries;
  }
}

// --- Áp dụng 1 câu trả lời chung cho toàn bộ ảnh trong giỏ VQA ---
function applyVqaAnswerToAll() {
  const bulkInput = document.getElementById("vqa-bulk-answer-input");
  const val = (bulkInput ? bulkInput.value.trim() : "").substring(0, 100);
  if (!val) {
    alert("⚠️ Vui lòng nhập nội dung đáp án chung trước khi bấm 'Áp dụng tất cả'!");
    return;
  }
  if (state.selectedItems.size === 0) {
    alert("⚠️ Giỏ nộp bài đang trống! Hãy chọn ảnh trước khi điền đáp án.");
    return;
  }

  state.selectedItems.forEach((cand) => {
    cand.user_answer = val;
  });
  updateDrawer();
}

// --- Query ID Utilities ---
function getDefaultQueryId() {
  const suffix = state.mode === "vqa" ? "qa" : state.mode;
  return `query-${state.batchQueries.size + 1}-${suffix}`;
}

function updateQueryIdInputForMode() {
  const queryIdInput = document.getElementById("query-id-input");
  const btnQueryIdLabel = document.getElementById("btn-query-id-label");
  if (queryIdInput) {
    const currentVal = queryIdInput.value.trim();
    const suffix = state.mode === "vqa" ? "qa" : state.mode;
    // Replace trailing suffix or set default
    const match = currentVal.match(/^(query-\d+)-(kis|qa|trake)$/);
    if (match) {
      queryIdInput.value = `${match[1]}-${suffix}`;
    } else if (!currentVal || currentVal.startsWith("query-")) {
      queryIdInput.value = `query-${state.batchQueries.size + 1}-${suffix}`;
    }
    if (btnQueryIdLabel) {
      btnQueryIdLabel.textContent = `${queryIdInput.value.trim()}.csv`;
    }
  }
}

// --- CSV Generator per 100% BTC Official Specification ---
// Quy chuẩn sotuyenaic.oj.io.vn/rules/:
// 1. Encoding: UTF-8, Delimiter: dấu phẩy (,)
// 2. TUYỆT ĐỐI KHÔNG CÓ DÒNG HEADER (Bắt đầu trực tiếp bằng dữ liệu)
// 3. Tối đa 100 dòng cho mỗi file CSV
// 4. KIS: <video_name>, <frame_id>
// 5. Q&A: <video_name>, <frame_id>, "<answer>" (tối đa 100 ký tự)
// 6. TRAKE: <video_name>, <frame_1>, <frame_2>, ..., <frame_N>
// Lưu ý: Chỉ xuất những dòng có keyframe hợp lệ; answer trống thì để ""
function buildCSVContent(mode, items) {
  const safeItems = Array.from(items).slice(0, 100);
  let lines = [];

  if (mode === "vqa") {
    safeItems.forEach((cand) => {
      // Chỉ xuất khi có keyframe hợp lệ
      if (!cand || !cand.video_id || cand.frame_id == null || cand.frame_id === "") return;
      let ans = (cand.user_answer != null ? cand.user_answer : (cand.answer || "")).trim();
      if (ans.length > 100) ans = ans.substring(0, 100);
      // Double quote escaping for RFC 4180 / BTC rule
      const escapedAns = ans.replace(/"/g, '""');
      lines.push(`${cand.video_id}, ${cand.frame_id}, "${escapedAns}"`);
    });
  } else if (mode === "trake") {
    safeItems.forEach((cand) => {
      if (!cand || !cand.video_id) return;
      if (cand.type === "trake_chain" && Array.isArray(cand.frame_ids) && cand.frame_ids.length > 0) {
        lines.push(`${cand.video_id}, ${cand.frame_ids.join(", ")}`);
      } else if (cand.events && Array.isArray(cand.events) && cand.events.length > 0) {
        const frames = cand.events.map((e) => e.frame_id || cand.frame_id);
        lines.push(`${cand.video_id}, ${frames.join(", ")}`);
      } else if (cand.event_frames && Array.isArray(cand.event_frames) && cand.event_frames.length > 0) {
        lines.push(`${cand.video_id}, ${cand.event_frames.join(", ")}`);
      } else if (cand.frame_id != null && cand.frame_id !== "") {
        lines.push(`${cand.video_id}, ${cand.frame_id}`);
      }
    });
  } else {
    // KIS Format
    safeItems.forEach((cand) => {
      if (!cand || !cand.video_id || cand.frame_id == null || cand.frame_id === "") return;
      lines.push(`${cand.video_id}, ${cand.frame_id}`);
    });
  }

  return lines.join("\r\n") + (lines.length > 0 ? "\r\n" : "");
}

// --- Xuất lẻ 1 file CSV cho câu hỏi hiện tại ---
function exportSingleCSV() {
  if (state.selectedItems.size === 0) {
    alert("⚠️ Chưa có bức ảnh nào trong Giỏ Nộp Bài! Hãy click dấu (+) trên ảnh để chọn.");
    return;
  }

  const queryIdInput = document.getElementById("query-id-input");
  let queryId = (queryIdInput ? queryIdInput.value.trim() : "") || getDefaultQueryId();
  // Strip .csv if user accidentally typed it
  queryId = queryId.replace(/\.csv$/i, "");

  const csvContent = buildCSVContent(state.mode, state.selectedItems.values());
  const blob = new Blob(["\uFEFF" + csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `${queryId}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// --- Lưu câu hiện tại vào Gói Nộp Bài (Batch Package) ---
function addCurrentCartToBatch() {
  if (state.selectedItems.size === 0) {
    alert("⚠️ Giỏ nộp bài đang trống! Hãy chọn ít nhất 1 ảnh trước khi lưu vào gói.");
    return;
  }

  const queryIdInput = document.getElementById("query-id-input");
  let queryId = (queryIdInput ? queryIdInput.value.trim() : "") || getDefaultQueryId();
  queryId = queryId.replace(/\.csv$/i, "");

  const itemsArr = Array.from(state.selectedItems.values());
  const csvContent = buildCSVContent(state.mode, itemsArr);

  state.batchQueries.set(queryId, {
    queryId: queryId,
    mode: state.mode,
    count: itemsArr.length,
    csvString: csvContent,
    items: Array.from(state.selectedItems.entries()),
    queryText: state.lastQuery || (queryInput ? queryInput.value.trim() : ""),
    searchMode: state.searchMode,
    resultsCache: [...state.rawResults],
    timestamp: new Date().toLocaleTimeString(),
  });

  renderBatchManagerUI();

  // Suggest next query ID (e.g. query-1-kis -> query-2-kis)
  const match = queryId.match(/^(query-)(\d+)(-[a-z]+)$/i);
  if (match && queryIdInput) {
    const nextNum = parseInt(match[2], 10) + 1;
    queryIdInput.value = `${match[1]}${nextNum}${match[3]}`;
    const btnLabel = document.getElementById("btn-query-id-label");
    if (btnLabel) btnLabel.textContent = `${queryIdInput.value}.csv`;
  }

  // Clear current cart for next question
  state.selectedItems.clear();
  updateDrawer();
  renderResults(state.rawResults);
  showToast(`💾 Đã lưu câu '${queryId}.csv' vào Gói Nộp Bài!`);
}

// --- Render Gói Đề Thi UI ---
function renderBatchManagerUI() {
  const batchListEl = document.getElementById("batch-query-list");
  const batchCountEl = document.getElementById("batch-count");
  if (batchCountEl) batchCountEl.textContent = state.batchQueries.size;

  if (!batchListEl) return;

  if (state.batchQueries.size === 0) {
    batchListEl.innerHTML = `<span class="batch-empty-hint">Chưa lưu câu nào vào gói. Bấm '➕ Lưu Gói' sau khi chọn ảnh.</span>`;
    return;
  }

  batchListEl.innerHTML = "";
  state.batchQueries.forEach((qData, qid) => {
    const itemDiv = document.createElement("div");
    itemDiv.className = "batch-query-item";
    itemDiv.innerHTML = `
      <div class="batch-query-info">
        <span class="batch-query-name">📄 ${qid}.csv</span>
        <span class="batch-query-meta">${qData.count} dòng • ${qData.mode.toUpperCase()}${qData.queryText ? ` • "${qData.queryText.substring(0, 15)}..."` : ''}</span>
      </div>
      <div class="batch-item-actions">
        <button class="batch-load-btn" onclick="loadBatchQueryToCart('${qid}')" title="Nạp lại câu này và kết quả tìm kiếm gốc">👁️ Nạp lại</button>
        <button class="batch-remove-item-btn" onclick="removeBatchQuery('${qid}')" title="Xóa câu này khỏi gói">✕</button>
      </div>
    `;
    batchListEl.appendChild(itemDiv);
  });
}

// --- Nạp lại câu hỏi từ Gói Nộp Bài vào Giỏ & Tải lại kết quả tìm kiếm gốc từ Cache ---
window.loadBatchQueryToCart = function(qid) {
  const qData = state.batchQueries.get(qid);
  if (!qData) return;

  // 1. Chuyển tab sang mode tương ứng
  state.mode = qData.mode;
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === qData.mode);
  });
  updateUIForMode();

  // 2. Set query id input
  const queryIdInput = document.getElementById("query-id-input");
  const btnLabel = document.getElementById("btn-query-id-label");
  if (queryIdInput) queryIdInput.value = qid;
  if (btnLabel) btnLabel.textContent = `${qid}.csv`;

  // 3. Phục hồi câu query gốc vào thanh tìm kiếm
  if (queryInput) {
    queryInput.value = qData.queryText || "";
  }
  state.lastQuery = qData.queryText || "";

  // 4. Phục hồi chế độ tìm kiếm (searchMode)
  if (qData.searchMode) {
    state.searchMode = qData.searchMode;
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.searchMode === qData.searchMode);
    });
  }

  // 5. Nạp lại selectedItems vào Giỏ Nộp Bài
  state.selectedItems.clear();
  if (qData.items && Array.isArray(qData.items)) {
    qData.items.forEach(([k, v]) => {
      state.selectedItems.set(k, v);
    });
  }
  updateDrawer();

  // 6. Hiển thị lại kết quả tìm kiếm tức thì từ Cache (0ms)
  if (qData.resultsCache && qData.resultsCache.length > 0) {
    state.rawResults = qData.resultsCache;
    renderResults(state.rawResults);
    statusBar.textContent = `⚡ Đã nạp lại ${state.rawResults.length} kết quả từ Query Cache cho "${qData.queryText || qid}"`;
  } else {
    // Fallback: Tìm trong queryCache toàn cục hoặc gọi API
    const cacheKey = `${qData.mode}_${qData.searchMode || 'all'}_${qData.queryText || ''}`;
    if (state.queryCache && state.queryCache.has(cacheKey)) {
      const cached = state.queryCache.get(cacheKey);
      state.rawResults = cached.results;
      renderResults(state.rawResults);
      statusBar.textContent = `⚡ Đã nạp lại ${state.rawResults.length} kết quả từ Query Cache cho "${qData.queryText || qid}"`;
    } else if (qData.queryText && qData.queryText.trim()) {
      triggerSearch();
    } else {
      renderResults(state.rawResults);
    }
  }

  showToast(`📂 Đã nạp lại câu '${qid}' và kết quả tìm kiếm gốc!`);
};

window.removeBatchQuery = function(qid) {
  state.batchQueries.delete(qid);
  renderBatchManagerUI();
};

function clearAllBatchQueries() {
  if (state.batchQueries.size === 0) return;
  if (confirm("Bạn có chắc muốn xóa toàn bộ các câu trong gói đề thi hiện tại?")) {
    state.batchQueries.clear();
    renderBatchManagerUI();
  }
}

// --- ĐÓNG GÓI SUBMISSION.ZIP (CHUẨN 100% BTC SƠ TUYỂN AIC) ---
// Quy tắc BTC: File .zip BẮT BUỘC chứa thư mục con 'submission/' bên trong.
async function packageSubmissionZip() {
  if (typeof JSZip === "undefined") {
    alert("⚠️ Lỗi: Không thể nạp thư viện JSZip. Vui lòng kiểm tra kết nối internet hoặc thử lại.");
    return;
  }

  // Nếu trong batch chưa có câu nào nhưng giỏ đang có ảnh, tự động thêm giỏ vào batch
  if (state.batchQueries.size === 0) {
    if (state.selectedItems.size > 0) {
      addCurrentCartToBatch();
    } else {
      alert("⚠️ Gói nộp bài đang trống! Hãy chọn ảnh và bấm '➕ Lưu Gói' cho từng câu hỏi trước khi đóng gói .zip.");
      return;
    }
  }

  const zip = new JSZip();
  // BẮT BUỘC TẠO THƯ MỤC 'submission' THEO QUY CHẾ BTC
  const submissionFolder = zip.folder("submission");

  let totalFiles = 0;
  state.batchQueries.forEach((qData, qid) => {
    const filename = qid.endsWith(".csv") ? qid : `${qid}.csv`;
    submissionFolder.file(filename, qData.csvString);
    totalFiles++;
  });

  try {
    const zipBlob = await zip.generateAsync({
      type: "blob",
      compression: "DEFLATE",
      compressionOptions: { level: 9 },
    });

    const url = URL.createObjectURL(zipBlob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    const teamName = "team_submission";
    link.setAttribute("download", `${teamName}_${Date.now()}.zip`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    alert(`🎉 ĐÓNG GÓI THÀNH CÔNG!\n\nĐã xuất file .zip chứa thư mục 'submission/' với ${totalFiles} file CSV chuẩn quy chế BTC sơ tuyển AIC 2026.\nBạn có thể nộp trực tiếp file zip này lên hệ thống chấm thi.`);
  } catch (err) {
    console.error("Lỗi đóng gói zip:", err);
    alert("❌ Đã xảy ra lỗi khi tạo file zip: " + err.message);
  }
}

// Global window functions for direct access
window.addCurrentCartToBatch = addCurrentCartToBatch;
window.applyVqaAnswerToAll = applyVqaAnswerToAll;
window.exportSingleCSV = exportSingleCSV;
window.packageSubmissionZip = packageSubmissionZip;
window.clearAllBatchQueries = clearAllBatchQueries;

// Lightbox Modal — phát video tương tác ngay trong khung ảnh
let currentLightboxCand = null;

const modalImgEl = document.getElementById("modal-img");
const modalVideoEl = document.getElementById("modal-video");
const videoFrameBarEl = document.getElementById("video-frame-bar");
const videoErrorMsgEl = document.getElementById("video-error-msg");
const videoInfoLabelEl = document.getElementById("video-info-label");

let currentVideoId = null;
let currentVideoFps = 25.0; // fallback, được ghi đè bằng fps thật từ API ngay khi mở modal

function currentVideoFrame() {
  return Math.round(modalVideoEl.currentTime * currentVideoFps);
}

function updateVideoInfoLabel() {
  if (!currentVideoId) return;
  videoInfoLabelEl.textContent = `${currentVideoId},${currentVideoFrame()}`;
}

// --- Bulletproof Vietnamese Diacritics & Unicode Normalized Highlighter ---
function removeVietnameseDiacritics(str) {
  if (!str) return "";
  const normalized = str.normalize("NFC").toLowerCase();
  const accents = {
    '[àáảãạăằắẳẵặâầấẩẫậ]': 'a',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y',
    'đ': 'd'
  };
  let res = normalized;
  for (const [pattern, repl] of Object.entries(accents)) {
    res = res.replace(new RegExp(pattern, 'g'), repl);
  }
  return res;
}

function buildVietnameseRegexPattern(term) {
  const charMap = {
    'a': '[aàáảãạăằắẳẵặâầấẩẫậAÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]',
    'd': '[dđDĐ]',
    'e': '[eèéẻẽẹêềếểễệEÈÉẺẼẸÊỀẾỂỄỆ]',
    'i': '[iìíỉĩịIÌÍỈĨỊ]',
    'o': '[oòóỏõọôồốổỗộơờớởỡợOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ]',
    'u': '[uùúủũụưừứửữựUÙÚỦŨỤƯỪỨỬỮỰ]',
    'y': '[yỳýỷỹỵYỲÝỶỸỴ]'
  };

  const rootTerm = removeVietnameseDiacritics(term);
  let pat = "";
  for (const ch of rootTerm) {
    if (charMap[ch]) {
      pat += charMap[ch];
    } else if (/\s/.test(ch)) {
      pat += '\\s+';
    } else {
      pat += ch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
  }
  return pat;
}

function highlightMatchedText(fullText, query) {
  if (!fullText || fullText === "None" || fullText === "-") return fullText || "None";
  if (!query || typeof query !== "string" || !query.trim()) return escapeHtml(fullText);

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  const normalizedText = fullText.normalize("NFC");
  const rawTerms = query
    .normalize("NFC")
    .trim()
    .split(/\s+/)
    .filter((t) => t.length >= 2);

  // Sắp xếp các cụm từ dài trước, từ ngắn sau để regex match tham lam chính xác
  const uniqueTerms = Array.from(new Set([query.normalize("NFC").trim(), ...rawTerms])).sort((a, b) => b.length - a.length);

  const patterns = uniqueTerms.map(buildVietnameseRegexPattern).filter(Boolean);
  if (patterns.length === 0) return escapeHtml(normalizedText);

  const regex = new RegExp(`(${patterns.join("|")})`, "gi");

  return escapeHtml(normalizedText).replace(regex, (match) => `<mark class="highlight-match">${match}</mark>`);
}

function showLightbox(cand, imgUrl) {
  currentLightboxCand = cand;
  if (state.currentResultsList && state.currentResultsList.length > 0) {
    const foundIdx = state.currentResultsList.findIndex(
      (c) => c && c.video_id === cand.video_id && c.frame_id === cand.frame_id
    );
    if (foundIdx !== -1) {
      state.currentLightboxIndex = foundIdx;
    }
  }

  const modal = document.getElementById("lightbox-modal");
  document.getElementById("modal-title").textContent = `${cand.video_id} — Scene: ${cand.scene_id || ''} (Frame ${cand.frame_id})`;

  const descBox = document.getElementById("modal-desc");
  const ocrBox = document.getElementById("modal-ocr");
  const asrBox = document.getElementById("modal-asr");

  const queryForHighlight = state.lastQuery || (queryInput ? queryInput.value.trim() : "");

  descBox.innerHTML = `<strong>Image Description:</strong> ${highlightMatchedText(cand.description || 'None', queryForHighlight)}`;
  ocrBox.innerHTML = `<strong>OCR:</strong> ${highlightMatchedText(cand.ocr_text || 'None', queryForHighlight)}`;
  asrBox.innerHTML = `<strong>ASR:</strong> ${highlightMatchedText(cand.asr_text || 'None', queryForHighlight)}`;

  // Default scroll position at top
  descBox.scrollTop = 0;
  ocrBox.scrollTop = 0;
  asrBox.scrollTop = 0;

  loadLightboxVideo(cand, imgUrl);

  modal.style.display = "flex";
}

// --- Điều hướng sang ảnh kết quả tại vị trí index trong modal ---
function showLightboxAt(index) {
  if (!state.currentResultsList || state.currentResultsList.length === 0) return;

  // Wrap around vòng lặp
  if (index < 0) {
    index = state.currentResultsList.length - 1;
  } else if (index >= state.currentResultsList.length) {
    index = 0;
  }

  state.currentLightboxIndex = index;
  const cand = state.currentResultsList[index];
  if (!cand) return;

  const imgUrl = `${API_BASE}/api/v1/image/${cand.video_id}/${cand.frame_id}`;
  showLightbox(cand, imgUrl);
}

async function loadLightboxVideo(cand, imgUrl) {
  currentVideoId = cand.video_id;
  currentLightboxCand = cand;
  isVideoInteractMode = false;
  videoErrorMsgEl.style.display = "none";

  // Mặc định thử phát video ngay trong khung ảnh; nếu lỗi sẽ fallback về ảnh tĩnh.
  modalImgEl.style.display = "none";
  modalVideoEl.style.display = "block";
  modalVideoEl.volume = 0.25; // 25% initial volume per requirement
  videoFrameBarEl.style.display = "flex";
  videoInfoLabelEl.textContent = `${cand.video_id},${cand.frame_id}`;

  const repMenu = document.getElementById("video-replace-menu");
  if (repMenu) repMenu.style.display = "none";

  // Lấy fps thật của video này để tính thời gian chính xác 100%
  let fps = 25.0;
  try {
    const res = await fetch(`${API_BASE}/api/v1/video/${cand.video_id}/fps`);
    if (res.ok) {
      const data = await res.json();
      if (data.fps) {
        fps = Number(data.fps);
      }
    }
  } catch (err) {
    // Giữ fallback fps nếu API lỗi
  }

  currentVideoFps = fps;

  // Tính chính xác giây tua tới dựa trên frame_id và fps thật của video
  let seekTo = 0;
  if (cand.frame_id !== undefined && cand.frame_id !== null && fps > 0) {
    seekTo = cand.frame_id / fps;
  } else if (cand.timestamp_mid !== undefined && cand.timestamp_mid !== null) {
    seekTo = cand.timestamp_mid;
  }

  modalVideoEl.src = `${API_BASE}/api/v1/video/${cand.video_id}`;

  const onLoadedMeta = () => {
    modalVideoEl.currentTime = seekTo;
    modalVideoEl.volume = 0.25;
    updateVideoInfoLabel();
    modalVideoEl.removeEventListener("loadedmetadata", onLoadedMeta);
  };
  modalVideoEl.addEventListener("loadedmetadata", onLoadedMeta);

  modalVideoEl.onerror = () => {
    // Không tìm thấy video trên server -> fallback hiển thị ảnh keyframe tĩnh
    modalVideoEl.style.display = "none";
    videoFrameBarEl.style.display = "none";
    modalImgEl.style.display = "block";
    modalImgEl.src = imgUrl;
    videoErrorMsgEl.style.display = "block";
  };

  // Mở video ở chế độ dừng (pause) sẵn tại đúng frame, người dùng tự bấm play khi muốn xem.
  modalVideoEl.pause();
  updateVideoInfoLabel();
}

function closeLightboxVideo() {
  modalVideoEl.pause();
  modalVideoEl.removeAttribute("src");
  modalVideoEl.load();
  modalVideoEl.onerror = null;
  currentVideoId = null;
  isVideoInteractMode = false;
  const repMenu = document.getElementById("video-replace-menu");
  if (repMenu) repMenu.style.display = "none";
}

const modalEl = document.getElementById("lightbox-modal");
if (modalEl) {
  document.getElementById("close-modal-btn").addEventListener("click", () => {
    modalEl.style.display = "none";
    closeLightboxVideo();
  });

  const prevModalBtn = document.getElementById("modal-prev-btn");
  if (prevModalBtn) {
    prevModalBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showLightboxAt(state.currentLightboxIndex - 1);
    });
  }

  const nextModalBtn = document.getElementById("modal-next-btn");
  if (nextModalBtn) {
    nextModalBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showLightboxAt(state.currentLightboxIndex + 1);
    });
  }

  modalEl.addEventListener("click", (e) => {
    if (e.target === modalEl) {
      modalEl.style.display = "none";
      closeLightboxVideo();
    }
  });

  // Cập nhật "video_id,frame" realtime: đang phát, tua timeline, hoặc dừng lại.
  modalVideoEl.addEventListener("timeupdate", updateVideoInfoLabel);
  modalVideoEl.addEventListener("seeked", updateVideoInfoLabel);
  modalVideoEl.addEventListener("pause", updateVideoInfoLabel);
  modalVideoEl.addEventListener("play", updateVideoInfoLabel);

  // Về Frame Chính: quay lại đúng thời điểm frame gốc đã tìm được
  document.getElementById("video-goto-main-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    if (currentLightboxCand) {
      if (currentLightboxCand.frame_id !== undefined && currentVideoFps > 0) {
        modalVideoEl.currentTime = currentLightboxCand.frame_id / currentVideoFps;
      } else {
        modalVideoEl.currentTime = currentLightboxCand.timestamp_mid || 0;
      }
      updateVideoInfoLabel();
    }
  });

  // Copy Frame: copy <video_id,frame_id> của khung hình đang dừng/phát tại thời điểm hiện tại
  document.getElementById("video-copy-frame-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    if (!currentVideoId) return;
    copyToClipboard({ video_id: currentVideoId, frame_id: currentVideoFrame() });
  });

  // Thay Frame Trong Giỏ (Realignment)
  const replaceFrameBtn = document.getElementById("video-replace-frame-btn");
  if (replaceFrameBtn) {
    replaceFrameBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleReplaceFrameInCart();
    });
  }

  // Tìm ảnh tương tự từ Lightbox
  const videoSearchSimBtn = document.getElementById("video-search-similar-btn");
  if (videoSearchSimBtn) {
    const syncVideoSearchButton = () => {
      videoSearchSimBtn.disabled = !currentVideoId || !modalVideoEl.paused;
    };
    modalVideoEl.addEventListener("loadedmetadata", syncVideoSearchButton);
    modalVideoEl.addEventListener("play", syncVideoSearchButton);
    modalVideoEl.addEventListener("pause", syncVideoSearchButton);
    syncVideoSearchButton();

    videoSearchSimBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!currentVideoId || !modalVideoEl.paused) return;
      await searchSimilarToCurrentFrame();
    });
  }

  // Thêm vào giỏ: thêm khung hình đang dừng tại thời điểm hiện tại vào Giỏ Nộp Bài
  document.getElementById("video-add-frame-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    if (!currentVideoId) return;

    const frameId = currentVideoFrame();
    const ts = modalVideoEl.currentTime;
    const base = currentLightboxCand || {};
    const snap = captureCurrentVideoThumbnail();
    const newCand = {
      video_id: currentVideoId,
      frame_id: frameId,
      custom_thumb: snap,
      scene_id: base.scene_id || "",
      timestamp_start: ts,
      timestamp_end: ts,
      timestamp_mid: ts,
      keyframe_path: "",
      ocr_text: "",
      asr_text: "",
      description: "",
      score: 1.0,
    };

    const itemKey = `${newCand.video_id}_${newCand.frame_id}`;
    if (state.selectedItems.has(itemKey)) {
      showToast(`⚠️ Frame ${newCand.video_id},${newCand.frame_id} đã có trong giỏ`);
      return;
    }
    state.selectedItems.set(itemKey, newCand);
    updateDrawer();
    showToast(`➕ Đã thêm ${newCand.video_id},${newCand.frame_id} vào giỏ`);
  });
}

// --- Chụp ảnh snapshot hiện tại từ thẻ video để hiển thị thumbnail tức thì ---
function captureCurrentVideoThumbnail() {
  if (!modalVideoEl || !modalVideoEl.videoWidth || !modalVideoEl.videoHeight) return null;
  try {
    const canvas = document.createElement("canvas");
    canvas.width = modalVideoEl.videoWidth;
    canvas.height = modalVideoEl.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(modalVideoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  } catch (err) {
    return null;
  }
}

// --- Căn chỉnh & Thay thế Frame trong giỏ cho Video đang xem (Giữ nguyên kết quả tìm kiếm gốc) ---
function handleReplaceFrameInCart() {
  if (!currentVideoId) return;
  const currentFrame = currentVideoFrame();
  const replaceMenu = document.getElementById("video-replace-menu");
  if (!replaceMenu) return;

  // Tìm các item trong giỏ thuộc video này
  const matchingEntries = Array.from(state.selectedItems.entries()).filter(([k, cand]) => {
    if (cand.type === "trake_chain") return cand.video_id === currentVideoId;
    return cand.video_id === currentVideoId;
  });

  if (matchingEntries.length === 0) {
    alert(`⚠️ Giỏ nộp bài chưa có ảnh hoặc chuỗi TRAKE nào thuộc video ${currentVideoId}! Hãy bấm '➕ Thêm vào giỏ' hoặc chọn kết quả trước.`);
    return;
  }

  // Lấy entry đầu tiên khớp video
  const [cartKey, cartCand] = matchingEntries[0];

  if (cartCand.type === "trake_chain") {
    let buttonsHtml = `<span>Chọn Event của chuỗi <strong>${currentVideoId}</strong> muốn thay bằng frame <strong>f${currentFrame}</strong>:</span>`;
    cartCand.frame_ids.forEach((fid, idx) => {
      buttonsHtml += `
        <button class="replace-event-btn" onclick="applyFrameReplacement('${cartKey}', ${idx}, ${currentFrame})">
          Thay vào E${idx + 1} (cũ: f${fid})
        </button>
      `;
    });
    buttonsHtml += `<button class="replace-cancel-btn" onclick="document.getElementById('video-replace-menu').style.display='none'">✕ Hủy</button>`;
    replaceMenu.innerHTML = buttonsHtml;
    replaceMenu.style.display = "flex";
  } else {
    // KIS / VQA single item
    const oldFrame = cartCand.frame_id;
    cartCand.frame_id = currentFrame;
    const snap = captureCurrentVideoThumbnail();
    if (snap) {
      cartCand.custom_thumb = snap;
    }
    // Cập nhật lại key trong Map
    state.selectedItems.delete(cartKey);
    const newKey = `${cartCand.video_id}_${currentFrame}`;
    state.selectedItems.set(newKey, cartCand);
    updateDrawer();
    // Giữ nguyên kết quả tìm kiếm gốc state.rawResults
    showToast(`🔄 Đã thay thế frame ${currentVideoId}: f${oldFrame} ➔ f${currentFrame}!`);
    replaceMenu.style.display = "none";
  }
}

window.applyFrameReplacement = function(cartKey, eventIdx, newFrameId) {
  const item = state.selectedItems.get(cartKey);
  if (!item || !item.frame_ids) return;

  const oldFid = item.frame_ids[eventIdx];
  item.frame_ids[eventIdx] = newFrameId;
  if (item.chain_events && item.chain_events[eventIdx]) {
    item.chain_events[eventIdx].frame_id = newFrameId;
    const snap = captureCurrentVideoThumbnail();
    if (snap) {
      item.chain_events[eventIdx].custom_thumb = snap;
    }
  }

  updateDrawer();
  // Chỉ cập nhật giỏ nộp bài, hoàn toàn KHÔNG đụng tới kết quả tìm kiếm gốc
  const repMenu = document.getElementById("video-replace-menu");
  if (repMenu) repMenu.style.display = "none";
  showToast(`🔄 Đã thay thế Event ${eventIdx + 1} (${item.video_id}): f${oldFid} ➔ f${newFrameId}!`);
};

// Chụp khung hình đang dừng trong video-modal rồi gọi API image-to-image search
async function searchSimilarToCurrentFrame() {
  if (!modalVideoEl.videoWidth || !modalVideoEl.videoHeight) {
    showToast("⚠️ Chưa sẵn sàng khung hình để tìm kiếm");
    return;
  }

  const button = document.getElementById("video-search-similar-btn");
  const prevLabel = button ? button.textContent : "";
  if (button) {
    button.disabled = true;
    button.textContent = "⏳ Đang tìm...";
  }
  statusBar.textContent = "🕸️ Spidey đang tìm ảnh tương đồng với khung hình đang dừng...";

  try {
    const canvas = document.createElement("canvas");
    canvas.width = modalVideoEl.videoWidth;
    canvas.height = modalVideoEl.videoHeight;
    canvas.getContext("2d").drawImage(modalVideoEl, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) throw new Error("Không thể chụp khung hình từ video");

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    const startTime = performance.now();
    const resp = await fetch(`${API_BASE}/api/v1/similar-image?top_k=${state.topK}`, {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const duration = ((performance.now() - startTime) / 1000).toFixed(2);

    state.rawResults = data.results || [];
    renderResults(state.rawResults);

    modalEl.style.display = "none";
    closeLightboxVideo();

    statusBar.textContent = `✅ Tìm thấy ${state.rawResults.length} ảnh tương đồng với khung hình đã chọn`;
    speedTag.textContent = `${duration}s`;
    speedTag.style.display = "inline-block";
  } catch (err) {
    statusBar.textContent = `❌ Lỗi khi tìm ảnh tương đồng: ${err.message}`;
    showToast(`❌ Lỗi: ${err.message}`);
  } finally {
    if (button) {
      button.disabled = !currentVideoId || !modalVideoEl.paused;
      button.textContent = prevLabel;
    }
  }
}

// Text Zoom Modal Handling (Click any box to enlarge Image Description/OCR/ASR text)
const textZoomModal = document.getElementById("text-zoom-modal");
const textZoomBody = document.getElementById("text-zoom-body");
const textZoomTitle = document.getElementById("text-zoom-title");
let currentZoomedText = "";

function openTextZoom(typeLabel, textContent) {
  if (!textContent || textContent === 'None' || textContent === '-') return;
  currentZoomedText = textContent;
  textZoomTitle.textContent = `🔍 CHI TIẾT ${typeLabel.toUpperCase()}`;
  const queryForHighlight = queryInput ? queryInput.value.trim() : "";
  textZoomBody.innerHTML = highlightMatchedText(textContent, queryForHighlight);
  textZoomModal.style.display = "flex";
}

const descBoxEl = document.getElementById("modal-desc");
if (descBoxEl) {
  descBoxEl.addEventListener("click", (e) => {
    e.stopPropagation();
    const raw = descBoxEl.innerText || '';
    const cleanText = raw.replace(/^Image Description:\s*/i, '').trim();
    openTextZoom("Image Description", cleanText);
  });
}

const ocrBoxEl = document.getElementById("modal-ocr");
if (ocrBoxEl) {
  ocrBoxEl.addEventListener("click", (e) => {
    e.stopPropagation();
    const raw = ocrBoxEl.innerText || '';
    const cleanText = raw.replace(/^OCR:\s*/i, '').trim();
    openTextZoom("Văn Bản OCR", cleanText);
  });
}

const asrBoxEl = document.getElementById("modal-asr");
if (asrBoxEl) {
  asrBoxEl.addEventListener("click", (e) => {
    e.stopPropagation();
    const raw = asrBoxEl.innerText || '';
    const cleanText = raw.replace(/^ASR:\s*/i, '').trim();
    openTextZoom("Lời Thoại ASR", cleanText);
  });
}

const closeZoomBtn = document.getElementById("close-text-zoom-btn");
if (closeZoomBtn) {
  closeZoomBtn.addEventListener("click", () => {
    textZoomModal.style.display = "none";
  });
}

const copyZoomBtn = document.getElementById("copy-zoom-text-btn");
if (copyZoomBtn) {
  copyZoomBtn.addEventListener("click", () => {
    if (currentZoomedText) {
      navigator.clipboard.writeText(currentZoomedText).then(() => {
        showToast("📋 Đã copy toàn bộ văn bản!");
      });
    }
  });
}

if (textZoomModal) {
  textZoomModal.addEventListener("click", (e) => {
    if (e.target === textZoomModal) {
      textZoomModal.style.display = "none";
    }
  });
}
