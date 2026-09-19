document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const queryForm = document.getElementById('query-form');
  const queryInput = document.getElementById('query-input');
  const btnSubmit = document.getElementById('btn-submit');
  const btnSendText = document.getElementById('btn-send-text');
  const btnSendIcon = document.getElementById('btn-send-icon');
  const btnSpinner = document.getElementById('btn-spinner');

  const modelSelector = document.getElementById('model-selector');
  const ragToggle = document.getElementById('rag-toggle');
  const ragToggleText = document.getElementById('rag-toggle-text');
  const btnToggleCompare = document.getElementById('btn-toggle-compare');
  const compareBtnText = document.getElementById('compare-button-text');

  const standardView = document.getElementById('standard-view');
  const comparisonView = document.getElementById('comparison-view');

  const resultModelBadge = document.getElementById('result-model-badge');
  const resultRagBadge = document.getElementById('result-rag-badge');
  const resultLatencyPill = document.getElementById('result-latency-pill');
  const answerContent = document.getElementById('answer-content');
  const citationsWrapper = document.getElementById('citations-wrapper');
  const citationsList = document.getElementById('citations-list');
  const btnCopyAnswer = document.getElementById('btn-copy-answer');

  // Comparison View Elements
  const compRagAnswer = document.getElementById('comp-rag-answer');
  const compRagLatency = document.getElementById('comp-rag-latency');
  const compRagCitations = document.getElementById('comp-rag-citations');
  const compDirectAnswer = document.getElementById('comp-direct-answer');
  const compDirectLatency = document.getElementById('comp-direct-latency');

  // Status Elements
  const ragStatusBadge = document.getElementById('rag-status-badge');
  const ragStatusLabel = document.getElementById('rag-status-label');
  const ollamaStatusBadge = document.getElementById('ollama-status-badge');
  const ollamaStatusLabel = document.getElementById('ollama-status-label');

  // Modal Elements
  const btnOpenKb = document.getElementById('btn-open-kb');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const kbModal = document.getElementById('kb-modal');
  const kbDocList = document.getElementById('kb-doc-list');
  const kbCountPill = document.getElementById('kb-count-pill');

  let isCompareMode = false;
  let currentRawAnswer = '';

  // 1. Initial Status Check
  checkHealth();
  loadDocuments();

  // 2. Mode & Toggle Handlers
  ragToggle.addEventListener('change', (e) => {
    const isChecked = e.target.checked;
    ragToggleText.textContent = isChecked ? 'RAG Enabled' : 'RAG Disabled';
    resultRagBadge.textContent = isChecked ? 'RAG Active' : 'Direct LLM';
    if (isChecked) {
      resultRagBadge.classList.remove('disabled');
    } else {
      resultRagBadge.classList.add('disabled');
    }
  });

  btnToggleCompare.addEventListener('click', () => {
    isCompareMode = !isCompareMode;
    if (isCompareMode) {
      btnToggleCompare.classList.add('active');
      compareBtnText.textContent = 'Single Query Mode';
      standardView.style.display = 'none';
      comparisonView.style.display = 'flex';
    } else {
      btnToggleCompare.classList.remove('active');
      compareBtnText.textContent = 'Compare Mode (RAG vs Direct)';
      standardView.style.display = 'block';
      comparisonView.style.display = 'none';
    }
  });

  // 3. Suggestion Chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-q');
      queryInput.value = q;
      queryForm.dispatchEvent(new Event('submit'));
    });
  });

  // 4. Form Submission
  queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    setLoading(true);

    try {
      if (isCompareMode) {
        await executeCompareQuery(query);
      } else {
        await executeStandardQuery(query);
      }
    } catch (err) {
      console.error(err);
      renderError(err.message || 'Error occurred while communicating with services');
    } finally {
      setLoading(false);
    }
  });

  // Standard Query Execution
  async function executeStandardQuery(query) {
    const model = modelSelector.value;
    const useRag = ragToggle.checked;

    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        model: model,
        use_rag: useRag,
        top_k: 3
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const data = await response.json();
    renderStandardResponse(data);
  }

  // Comparison Query Execution
  async function executeCompareQuery(query) {
    const model = modelSelector.value;

    compRagAnswer.innerHTML = '<div class="spinner"></div> Generating grounded answer...';
    compDirectAnswer.innerHTML = '<div class="spinner"></div> Generating parametric answer...';

    const response = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        model: model
      })
    });

    if (!response.ok) {
      throw new Error(`Comparison API returned status ${response.status}`);
    }

    const data = await response.json();
    renderCompareResponse(data);
  }

  // Render Standard Output
  function renderStandardResponse(data) {
    currentRawAnswer = data.answer;
    resultModelBadge.textContent = `Model: ${data.model}`;
    resultLatencyPill.textContent = `Latency: ${data.latency_ms} ms`;

    // Markdown conversion
    answerContent.innerHTML = formatMarkdown(data.answer);

    // Citations
    if (data.citations && data.citations.length > 0) {
      citationsWrapper.style.display = 'block';
      citationsList.innerHTML = data.citations.map(c => `
        <div class="citation-item">
          <div class="citation-header">
            <span class="citation-title">${escapeHtml(c.document_title)}</span>
            <span class="citation-score">Sim: ${(c.similarity_score * 100).toFixed(1)}%</span>
          </div>
          <div class="citation-section">${escapeHtml(c.section_title)} (${escapeHtml(c.source_file)})</div>
          <p class="citation-snippet">${escapeHtml(c.snippet)}</p>
        </div>
      `).join('');
    } else {
      citationsWrapper.style.display = 'none';
      citationsList.innerHTML = '';
    }
  }

  // Render Comparison Output
  function renderCompareResponse(data) {
    const rag = data.rag_response;
    const direct = data.direct_response;

    compRagLatency.textContent = `${rag.latency_ms} ms`;
    compDirectLatency.textContent = `${direct.latency_ms} ms`;

    compRagAnswer.innerHTML = formatMarkdown(rag.answer);
    compDirectAnswer.innerHTML = formatMarkdown(direct.answer);

    if (rag.citations && rag.citations.length > 0) {
      compRagCitations.innerHTML = rag.citations.map(c => `
        <div class="citation-item">
          <div class="citation-header">
            <span class="citation-title">${escapeHtml(c.document_title)}</span>
            <span class="citation-score">Match: ${(c.similarity_score * 100).toFixed(0)}%</span>
          </div>
          <div class="citation-section">${escapeHtml(c.section_title)}</div>
        </div>
      `).join('');
    } else {
      compRagCitations.innerHTML = '';
    }
  }

  function renderError(msg) {
    answerContent.innerHTML = `
      <div style="color: #fb7185; padding: 16px; background: rgba(244, 63, 94, 0.1); border-radius: 8px; border: 1px solid rgba(244, 63, 94, 0.3);">
        <strong>Error:</strong> ${escapeHtml(msg)}
      </div>
    `;
  }

  // Copy Answer to Clipboard
  btnCopyAnswer.addEventListener('click', () => {
    if (!currentRawAnswer) return;
    navigator.clipboard.writeText(currentRawAnswer).then(() => {
      const orig = btnCopyAnswer.innerHTML;
      btnCopyAnswer.innerHTML = '✓ Copied';
      setTimeout(() => { btnCopyAnswer.innerHTML = orig; }, 2000);
    });
  });

  // Modal Controls
  btnOpenKb.addEventListener('click', () => {
    kbModal.classList.add('active');
  });

  btnCloseModal.addEventListener('click', () => {
    kbModal.classList.remove('active');
  });

  kbModal.addEventListener('click', (e) => {
    if (e.target === kbModal) kbModal.classList.remove('active');
  });

  // Evaluation & Benchmark Modal Controls
  const btnOpenEval = document.getElementById('btn-open-eval');
  const btnCloseEvalModal = document.getElementById('btn-close-eval-modal');
  const evalModal = document.getElementById('eval-modal');

  if (btnOpenEval) {
    btnOpenEval.addEventListener('click', () => {
      evalModal.classList.add('active');
    });
  }

  if (btnCloseEvalModal) {
    btnCloseEvalModal.addEventListener('click', () => {
      evalModal.classList.remove('active');
    });
  }

  if (evalModal) {
    evalModal.addEventListener('click', (e) => {
      if (e.target === evalModal) evalModal.classList.remove('active');
    });
  }

  // Evaluation Modal Tabs Switching
  document.querySelectorAll('.eval-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.eval-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.eval-tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // Check Health Status
  async function checkHealth() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const s = await res.json();
        
        // RAG Service
        if (s.rag_service) {
          ragStatusLabel.textContent = 'Active';
          ragStatusBadge.querySelector('.pulse-dot').className = 'pulse-dot active';
        } else {
          ragStatusLabel.textContent = 'Offline';
          ragStatusBadge.querySelector('.pulse-dot').className = 'pulse-dot warning';
        }

        // Ollama Service
        if (s.ollama_service) {
          ollamaStatusLabel.textContent = 'Online';
          ollamaStatusBadge.querySelector('.pulse-dot').className = 'pulse-dot active';
        } else {
          ollamaStatusLabel.textContent = 'Standby (Simulation Active)';
          ollamaStatusBadge.querySelector('.pulse-dot').className = 'pulse-dot warning';
        }
      }
    } catch (e) {
      ragStatusLabel.textContent = 'Unreachable';
      ollamaStatusLabel.textContent = 'Unreachable';
    }
  }

  // Load Indexed Documents
  async function loadDocuments() {
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        const docs = data.documents || [];
        kbCountPill.textContent = `${docs.length} Docs`;

        if (docs.length === 0) {
          kbDocList.innerHTML = '<p class="modal-sub">No indexed documents found.</p>';
          return;
        }

        kbDocList.innerHTML = docs.map(d => `
          <div class="kb-doc-card">
            <div class="kb-doc-header">
              <span class="kb-doc-title">${escapeHtml(d.document_title)}</span>
              <span class="kb-chunk-tag">${d.chunks_count} Chunks</span>
            </div>
            <div class="kb-file-name">${escapeHtml(d.source_file)}</div>
            <div class="kb-sections-list">
              ${(d.sections || []).map(s => `<span class="kb-sec-tag">${escapeHtml(s)}</span>`).join('')}
            </div>
          </div>
        `).join('');
      }
    } catch (e) {
      kbDocList.innerHTML = '<p class="modal-sub">Failed to retrieve indexed documents.</p>';
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      btnSubmit.disabled = true;
      btnSendText.style.display = 'none';
      btnSendIcon.style.display = 'none';
      btnSpinner.style.display = 'block';
    } else {
      btnSubmit.disabled = false;
      btnSendText.style.display = 'inline';
      btnSendIcon.style.display = 'inline';
      btnSpinner.style.display = 'none';
    }
  }

  // Simple Markdown Formatter
  function formatMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italics
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Bullet points
    html = html.replace(/^•\s+(.*)$/gm, '<li>$1</li>');
    html = html.replace(/^-\s+(.*)$/gm, '<li>$1</li>');
    // Wrap lists
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    // Paragraphs
    html = html.split('\n\n').map(p => {
      if (p.startsWith('<ul>') || p.startsWith('<li>') || p.startsWith('<div')) return p;
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return html;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Periodic health ping every 10 seconds
  setInterval(checkHealth, 10000);
});
