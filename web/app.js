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
      // Hiện/ẩn num_events cho TRAKE
      const numEventsGroup = document.getElementById("num-events-group");
      if (numEventsGroup) {
        numEventsGroup.style.display = (state.mode === "trake") ? "flex" : "none";
      }
      updateDrawer();
    });
  });
}

// --- Controls Setup ---
function setupControls() {
  // Top-K number input
  const topKInput = document.getElementById("top-k-input");
  if (topKInput) {
    topKInput.addEventListener("input", (e) => {
      let val = parseInt(e.target.value);
      if (!isNaN(val) && val > 0) {
        state.topK = val;
      }
    });
  }

  // Branch mode buttons (Hybrid, Embedding, OCR, ASR)
  document.querySelectorAll(".branch-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".branch-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.searchMode = btn.dataset.branch;
      statusBar.textContent = `⚙️ Đã chuyển sang nhánh tìm kiếm: ${btn.textContent.trim()}`;
    });
  });

  // Dynamic Density Slider (adjusts card min-width)
  const densitySlider = document.getElementById("density-slider");
  const densityVal = document.getElementById("density-val");
  if (densitySlider) {
    densitySlider.addEventListener("input", (e) => {
      const pxVal = e.target.value;
      resultsGrid.style.setProperty("--card-min-width", `${pxVal}px`);
      if (densityVal) densityVal.textContent = `${pxVal}px`;
    });
  }

  // Image upload
  const imgInput = document.getElementById("image-upload-input");
  if (imgInput) {
    imgInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (evt) => {
          state.selectedImageBase64 = evt.target.result.split(",")[1];
          document.getElementById("chip-filename").textContent = file.name;
          document.getElementById("image-preview-chip").style.display = "inline-flex";
        };
        reader.readAsDataURL(file);
      }
    });
  }

  const removeImgBtn = document.getElementById("remove-img-btn");
  if (removeImgBtn) {
    removeImgBtn.addEventListener("click", () => {
      state.selectedImageBase64 = null;
      document.getElementById("image-preview-chip").style.display = "none";
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

  if (!queryText && !state.selectedImageBase64) {
    statusBar.textContent = "⚠️ Vui lòng nhập câu tìm kiếm hoặc tải ảnh lên!";
    return;
  }

  statusBar.textContent = `🕸️ Spidey đang tìm kiếm (Nhánh: ${state.searchMode.toUpperCase()})...`;
  speedTag.style.display = "none";
  resultsGrid.innerHTML = `<div class="empty-state">🕷️ Đang thực hiện tìm kiếm & Rerank đa phương thức...</div>`;

  const startTime = performance.now();

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
    // TRAKE: render từng chain như 1 chuỗi ảnh
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
  const chainKey = chain.map(c => `${c.video_id}_${c.frame_id}`).join("__");
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
          <span class="ts-label">${candidate.timestamp_mid.toFixed(1)}s</span>
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
      <button class="copy-btn-overlay" title="Copy <video_id,frame_id> (Ctrl+C)">📋 Copy</button>
    </div>
    <div class="kf-details">
      <div class="kf-title">
        <span>${candidate.video_id}</span>
        <span class="ts-label">${candidate.timestamp_mid.toFixed(1)}s</span>
      </div>
      ${candidate.ocr_text ? `<div class="ocr-pill" title="${candidate.ocr_text}">🔤 ${candidate.ocr_text}</div>` : ''}
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

  // Selection Checkbox
  const selectBtn = card.querySelector(".select-checkbox-btn");
  selectBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleSelect(itemKey, candidate, selectBtn);
  });

  resultsGrid.appendChild(card);
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
      <img src="${imgUrl}" alt="${cand.scene_id}" />
      <div style="flex:1;">
        <div style="font-weight:bold; font-size:0.85rem;">${cand.video_id} (Frame ${cand.frame_id})</div>
        <div style="font-size:0.75rem; color:var(--text-sub);">${cand.timestamp_mid.toFixed(1)}s</div>
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

// Lightbox Modal
function showLightbox(cand, imgUrl) {
  const modal = document.getElementById("lightbox-modal");
  document.getElementById("modal-img").src = imgUrl;
  document.getElementById("modal-title").textContent = `${cand.video_id} — Scene: ${cand.scene_id} (Frame ${cand.frame_id})`;

  const descBox = document.getElementById("modal-desc");
  const ocrBox = document.getElementById("modal-ocr");
  const asrBox = document.getElementById("modal-asr");

  descBox.innerHTML = `<strong>Image Description:</strong> ${cand.description || 'None'}`;
  ocrBox.innerHTML = `<strong>OCR:</strong> ${cand.ocr_text || 'None'}`;
  asrBox.innerHTML = `<strong>ASR:</strong> ${cand.asr_text || 'None'}`;

  // Default scroll position at top
  descBox.scrollTop = 0;
  ocrBox.scrollTop = 0;
  asrBox.scrollTop = 0;

  modal.style.display = "flex";
}

const modalEl = document.getElementById("lightbox-modal");
if (modalEl) {
  document.getElementById("close-modal-btn").addEventListener("click", () => {
    modalEl.style.display = "none";
  });

  modalEl.addEventListener("click", (e) => {
    if (e.target === modalEl) {
      modalEl.style.display = "none";
    }
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
  textZoomBody.textContent = textContent;
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
