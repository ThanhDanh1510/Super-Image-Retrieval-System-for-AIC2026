/* =============================================================================
   🕷️ SPIDER-MAN THEME APP LOGIC (AIC2026 SUPER RETRIEVAL)
   ============================================================================= */

const API_BASE = "http://localhost:8000";

// App State
const state = {
  mode: "kis", // kis, vqa, trake
  searchMode: "all", // all, embedding, ocr, asr
  query: "",
  topK: 30,
  useReranker: false,
  selectedItems: new Map(), // key -> CandidateResult
  rawResults: [],
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
  if (state.mode === "trake") {
    queryInput.placeholder = "Gõ chuỗi sự kiện theo thứ tự... (VD: người đàn ông bước vào xe -> xe chạy trên đường -> dừng lại ở quán ăn)";
    if (numEventsGroup) numEventsGroup.style.display = "flex";
  } else if (state.mode === "vqa") {
    queryInput.placeholder = "Gõ câu hỏi VQA... (VD: Người lái xe mặc áo màu gì? / Có bao nhiêu người trong quán?)";
    if (numEventsGroup) numEventsGroup.style.display = "none";
  } else {
    queryInput.placeholder = "Gõ câu miêu tả tìm kiếm... (VD: xe máy màu vàng, tiệm vàng Lê Tứ)";
    if (numEventsGroup) numEventsGroup.style.display = "none";
  }
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
    topKInput.addEventListener("change", (e) => {
      state.topK = parseInt(e.target.value, 10) || 30;
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

  // Reranker Checkbox Switch
  const rerankerCheckbox = document.getElementById("use-reranker-checkbox");
  if (rerankerCheckbox) {
    // Sync UI with state initial value
    rerankerCheckbox.checked = state.useReranker;
    rerankerCheckbox.addEventListener("change", (e) => {
      state.useReranker = e.target.checked;
      statusBar.textContent = `⚙️ Reranker đã được ${state.useReranker ? "BẬT" : "TẮT"}`;
    });
  }
}

// --- Keyboard Shortcuts (Copy on Hover) ---
function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
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

      const resp = await fetch(`${API_BASE}/api/v1/similar-image?top_k=${state.topK}&use_reranker=${state.useReranker}`, {
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
      use_reranker: state.useReranker
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
    return;
  }

  resultsGrid.innerHTML = "";

  // TRAKE mode: mỗi item là 1 chain (array) của candidates
  const isTrake = state.mode === "trake" && Array.isArray(results[0]);

  if (isTrake) {
    // TRAKE: render từng chain trên 1 hàng
    results.forEach((chain, chainIdx) => {
      if (!Array.isArray(chain) || chain.length === 0) return;
      renderTrakeChain(chain, chainIdx);
    });
  } else {
    // KIS/VQA: render từng ảnh riêng lẻ
    results.forEach((candidate, idx) => {
      renderSingleCard(candidate);
    });
  }
}

// --- Render 1 TRAKE chain (chuỗi ảnh nối tiếp) ---
function renderTrakeChain(chain, chainIdx) {
  const avgScore = chain.reduce((sum, c) => sum + c.score, 0) / chain.length;

  const chainContainer = document.createElement("div");
  chainContainer.className = "trake-chain";

  let imagesHtml = "";
  chain.forEach((candidate, eventIdx) => {
    const imgUrl = `${API_BASE}/api/v1/image/${candidate.video_id}/${candidate.frame_id}`;
    const itemKey = `${candidate.video_id}_${candidate.frame_id}`;
    const isSelected = state.selectedItems.has(itemKey);

    imagesHtml += `
      <div class="trake-event" data-key="${itemKey}">
        <div class="trake-img-wrapper">
          <img src="${imgUrl}" alt="${candidate.scene_id}" loading="lazy" />
          <span class="event-badge">E${eventIdx + 1}</span>
          <button class="select-checkbox-btn ${isSelected ? 'selected' : ''}" data-key="${itemKey}">
            ${isSelected ? '✓' : '+'}
          </button>
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
    </div>
    <div class="trake-chain-images">
      ${imagesHtml}
    </div>
  `;

  // Event listeners cho từng event trong chain
  chainContainer.querySelectorAll(".trake-event").forEach((eventEl) => {
    const key = eventEl.dataset.key;
    const candidate = chain.find(c => `${c.video_id}_${c.frame_id}` === key);
    if (!candidate) return;

    const imgUrl = `${API_BASE}/api/v1/image/${candidate.video_id}/${candidate.frame_id}`;

    // Lightbox on click
    eventEl.querySelector("img").addEventListener("click", () => showLightbox(candidate, imgUrl));

    // Selection
    const selectBtn = eventEl.querySelector(".select-checkbox-btn");
    selectBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSelect(key, candidate, selectBtn);
    });

    // Hover tracking
    eventEl.addEventListener("mouseenter", () => { state.hoveredCand = candidate; });
    eventEl.addEventListener("mouseleave", () => { state.hoveredCand = null; });
  });

  resultsGrid.appendChild(chainContainer);
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
  statusBar.textContent = `🖼️ Đang nạp ảnh mẫu ${cand.video_id} #f${cand.frame_id}...`;
  try {
    const imgUrl = `${API_BASE}/api/v1/image/${cand.video_id}/${cand.frame_id}`;
    const res = await fetch(imgUrl);
    if (!res.ok) throw new Error("Không thể tải file ảnh keyframe mẫu");
    const blob = await res.blob();
    state.selectedFile = new File([blob], `${cand.video_id}_${cand.frame_id}.webp`, { type: blob.type || "image/webp" });

    const reader = new FileReader();
    reader.onload = (evt) => {
      state.selectedImageBase64 = evt.target.result.split(",")[1];
      const chipEl = document.getElementById("image-preview-chip");
      const chipNameEl = document.getElementById("chip-filename");
      if (chipNameEl) chipNameEl.textContent = `Ảnh mẫu: ${cand.video_id} #f${cand.frame_id}`;
      if (chipEl) chipEl.style.display = "inline-flex";

      // Tự động đóng Lightbox nếu đang mở và thực hiện tìm kiếm
      const modal = document.getElementById("lightbox-modal");
      if (modal && modal.style.display !== "none") {
        modal.style.display = "none";
        closeLightboxVideo();
      }

      triggerSearch();
    };
    reader.readAsDataURL(blob);
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
    const div = document.createElement("div");
    div.className = "selected-item";
    const imgUrl = `${API_BASE}/api/v1/image/${cand.video_id}/${cand.frame_id}`;
    
    div.innerHTML = `
      <img src="${imgUrl}" alt="${cand.scene_id}" onerror="this.style.display='none'" />
      <div style="flex:1;">
        <div style="font-weight:bold; font-size:0.85rem;">${cand.video_id}</div>
        <div style="font-size:0.75rem; color:var(--spider-blue); font-family:var(--font-code); font-weight:600;">f${cand.frame_id}</div>
        ${state.mode === "vqa" ? `
          <input type="text" class="vqa-ans-input" data-key="${key}" placeholder="Nhập đáp án VQA..." value="${cand.user_answer || ''}" style="width:100%; margin-top:4px; font-size:0.75rem; padding:2px 4px; background:#000; border:1px solid var(--spider-blue); color:#fff; border-radius:4px;" />
        ` : ''}
      </div>
      <button onclick="copyItemKey('${cand.video_id}', ${cand.frame_id})" style="background:none; border:none; color:var(--spider-blue); cursor:pointer; font-size:1rem; margin-right:4px;" title="Copy">📋</button>
      <button onclick="removeItem('${key}')" style="background:none; border:none; color:var(--spider-red); cursor:pointer; font-weight:bold;">✕</button>
    `;
    selectedList.appendChild(div);
  });

  // Attach event listener for VQA inputs
  document.querySelectorAll(".vqa-ans-input").forEach(input => {
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

  // CSV Exporter
  const exportBtn = document.getElementById("export-csv-btn");
  if (exportBtn) {
    exportBtn.onclick = exportCSV;
  }
}

// --- CSV Exporter per Task Standard ---
function exportCSV() {
  if (state.selectedItems.size === 0) {
    alert("Chưa có item nào được chọn để xuất CSV!");
    return;
  }

  let csvContent = "";

  if (state.mode === "vqa") {
    // VQA Format: video_id,frame_id,answer
    csvContent = "video_id,frame_id,answer\n";
    state.selectedItems.forEach((cand) => {
      const ans = (cand.user_answer || '').replace(/"/g, '""');
      csvContent += `${cand.video_id},${cand.frame_id},"${ans}"\n`;
    });
  } else if (state.mode === "trake") {
    // TRAKE Format: video_id,frame_id_1,frame_id_2,frame_id_3,frame_id_4
    csvContent = "video_id,frame_id_1,frame_id_2,frame_id_3,frame_id_4\n";
    state.selectedItems.forEach((cand) => {
      csvContent += `${cand.video_id},${cand.frame_id},${cand.frame_id},${cand.frame_id},${cand.frame_id}\n`;
    });
  } else {
    // KIS Format: ONLY video_id,frame_id (Standard BTC)
    csvContent = "video_id,frame_id\n";
    state.selectedItems.forEach((cand) => {
      csvContent += `${cand.video_id},${cand.frame_id}\n`;
    });
  }

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `submission_${state.mode}_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

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
  const modal = document.getElementById("lightbox-modal");
  document.getElementById("modal-title").textContent = `${cand.video_id} — Scene: ${cand.scene_id} (Frame ${cand.frame_id})`;

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

async function loadLightboxVideo(cand, imgUrl) {
  currentVideoId = cand.video_id;
  currentLightboxCand = cand;
  videoErrorMsgEl.style.display = "none";

  // Mặc định thử phát video ngay trong khung ảnh; nếu lỗi sẽ fallback về ảnh tĩnh.
  modalImgEl.style.display = "none";
  modalVideoEl.style.display = "block";
  videoFrameBarEl.style.display = "flex";
  videoInfoLabelEl.textContent = `${cand.video_id},${cand.frame_id}`;

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
}

const modalEl = document.getElementById("lightbox-modal");
if (modalEl) {
  document.getElementById("close-modal-btn").addEventListener("click", () => {
    modalEl.style.display = "none";
    closeLightboxVideo();
  });

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

  // Tìm ảnh tương tự từ Lightbox
  const videoSearchSimBtn = document.getElementById("video-search-similar-btn");
  if (videoSearchSimBtn) {
    videoSearchSimBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!currentVideoId) return;
      const targetCand = {
        video_id: currentVideoId,
        frame_id: currentVideoFrame(),
      };
      searchSimilarByCandidate(targetCand);
    });
  }

  // Thêm vào giỏ: thêm khung hình đang dừng tại thời điểm hiện tại vào Giỏ Nộp Bài
  document.getElementById("video-add-frame-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    if (!currentVideoId) return;

    const frameId = currentVideoFrame();
    const ts = modalVideoEl.currentTime;
    const base = currentLightboxCand || {};
    const newCand = {
      video_id: currentVideoId,
      frame_id: frameId,
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
