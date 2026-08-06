/* =============================================================================
   🕷️ SPIDER-MAN THEME APP LOGIC (AIC2026 SUPER RETRIEVAL)
   ============================================================================= */

const API_BASE = "http://localhost:8000";

// App State
const state = {
  mode: "kis", // kis, vqa, trake
  query: "",
  topK: 30,
  selectedItems: new Map(), // key -> CandidateResult
  rawResults: [],
  selectedImageBase64: null,
};

// DOM Elements
const queryInput = document.getElementById("query-input");
const searchBtn = document.getElementById("search-btn");
const topKSlider = document.getElementById("top-k-slider");
const topKVal = document.getElementById("top-k-val");
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
});

// --- Tab Setup ---
function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;

      // Toggle TRAKE event inputs
      const trakeContainer = document.getElementById("trake-events-container");
      if (state.mode === "trake") {
        trakeContainer.style.display = "block";
      } else {
        trakeContainer.style.display = "none";
      }
    });
  });
}

// --- Controls Setup ---
function setupControls() {
  const topKInput = document.getElementById("top-k-input");
  topKInput.addEventListener("input", (e) => {
    let val = parseInt(e.target.value);
    if (!isNaN(val) && val > 0) {
      state.topK = val;
    }
  });

  // Dynamic Density Slider (adjusts card min-width)
  const densitySlider = document.getElementById("density-slider");
  const densityVal = document.getElementById("density-val");
  densitySlider.addEventListener("input", (e) => {
    const pxVal = e.target.value;
    resultsGrid.style.setProperty("--card-min-width", `${pxVal}px`);
    densityVal.textContent = `${pxVal}px`;
  });

  // Image upload
  const imgInput = document.getElementById("image-upload-input");
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

  document.getElementById("remove-img-btn").addEventListener("click", () => {
    state.selectedImageBase64 = null;
    document.getElementById("image-preview-chip").style.display = "none";
  });
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
  } else {
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

  statusBar.textContent = "🕸️ Spidey đang quét dữ liệu trong Qdrant Vector DB...";
  speedTag.style.display = "none";
  resultsGrid.innerHTML = `<div class="empty-state">🕷️ Đang thực hiện tìm kiếm & Rerank đa phương thức...</div>`;

  const startTime = performance.now();

  try {
    let endpoint = `${API_BASE}/api/v1/${state.mode}`;
    let payload = {};

    if (state.mode === "kis") {
      payload = {
        input_type: "text",
        query: queryText,
        top_k: state.topK,
        user_choices: []
      };
    } else if (state.mode === "vqa") {
      payload = {
        input_type: "text",
        query: queryText,
        question: queryText,
        top_k: state.topK,
        user_choices: []
      };
    } else if (state.mode === "trake") {
      const inputs = document.querySelectorAll(".trake-event-input");
      const subEvents = Array.from(inputs).map(i => i.value.trim()).filter(Boolean);
      payload = {
        input_type: "text",
        query: subEvents.join("\n") || queryText,
        num_events: subEvents.length || 2,
        top_k: state.topK,
        user_choices: []
      };
    }

    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    const duration = ((performance.now() - startTime) / 1000).toFixed(2);

    state.rawResults = data.results || [];
    renderResults(state.rawResults);

    statusBar.textContent = `✅ Đã tìm thấy ${state.rawResults.length} kết quả phù hợp cho query: "${data.query_text || queryText}"`;
    speedTag.textContent = `${duration}s`;
    speedTag.style.display = "inline-block";

  } catch (err) {
    statusBar.textContent = `❌ Lỗi khi tìm kiếm: ${err.message}`;
    resultsGrid.innerHTML = `<div class="empty-state">Không thể kết nối đến server API tại ${API_BASE}</div>`;
  }
}

// --- Render High Density Grid Results ---
function renderResults(results) {
  if (!results || results.length === 0) {
    resultsGrid.innerHTML = `<div class="empty-state">Không tìm thấy kết quả nào phù hợp.</div>`;
    return;
  }

  resultsGrid.innerHTML = "";

  results.forEach((item, idx) => {
    // Handle TRAKE tuple format or KIS single format
    const candidate = Array.isArray(item) ? item[0] : item;
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
      </div>
      <div class="kf-details">
        <div class="kf-title">
          <span>${candidate.video_id}</span>
          <span class="ts-label">${candidate.timestamp_mid.toFixed(1)}s</span>
        </div>
        ${candidate.ocr_text ? `<div class="ocr-pill" title="${candidate.ocr_text}">🔤 ${candidate.ocr_text}</div>` : ''}
      </div>
    `;

    // Lightbox image preview on click
    card.querySelector("img").addEventListener("click", () => showLightbox(candidate, imgUrl));

    // Selection Checkbox
    const selectBtn = card.querySelector(".select-checkbox-btn");
    selectBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSelect(itemKey, candidate, selectBtn);
    });

    resultsGrid.appendChild(card);
  });
}

// --- Selection Drawer ---
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
      </div>
      <button onclick="removeItem('${key}')" style="background:none; border:none; color:var(--spider-red); cursor:pointer; font-weight:bold;">✕</button>
    `;
    selectedList.appendChild(div);
  });
}

window.removeItem = function(key) {
  state.selectedItems.delete(key);
  updateDrawer();
  renderResults(state.rawResults);
};

function setupDrawer() {
  document.getElementById("toggle-drawer-btn").addEventListener("click", () => drawer.classList.add("open"));
  document.getElementById("close-drawer-btn").addEventListener("click", () => drawer.classList.remove("open"));

  // CSV Exporter
  document.getElementById("export-csv-btn").addEventListener("click", exportCSV);
}

function exportCSV() {
  if (state.selectedItems.size === 0) {
    alert("Chưa có item nào được chọn để xuất CSV!");
    return;
  }

  let csvContent = "video_id,frame_id,timestamp_mid,keyframe_path,ocr_text\n";
  state.selectedItems.forEach((cand) => {
    const ocrEscaped = `"${(cand.ocr_text || '').replace(/"/g, '""')}"`;
    csvContent += `${cand.video_id},${cand.frame_id},${cand.timestamp_mid},${cand.keyframe_path},${ocrEscaped}\n`;
  });

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `AIC2026_submission_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Lightbox
function showLightbox(cand, imgUrl) {
  const modal = document.getElementById("lightbox-modal");
  document.getElementById("modal-img").src = imgUrl;
  document.getElementById("modal-title").textContent = `${cand.video_id} — Scene: ${cand.scene_id} (Frame ${cand.frame_id})`;
  document.getElementById("modal-ocr").innerHTML = `<strong>OCR:</strong> ${cand.ocr_text || 'None'}`;
  document.getElementById("modal-asr").innerHTML = `<strong>ASR:</strong> ${cand.asr_text || 'None'}`;
  modal.style.display = "flex";
}

const modalEl = document.getElementById("lightbox-modal");
document.getElementById("close-modal-btn").addEventListener("click", () => {
  modalEl.style.display = "none";
});

modalEl.addEventListener("click", (e) => {
  if (e.target === modalEl) {
    modalEl.style.display = "none";
  }
});
