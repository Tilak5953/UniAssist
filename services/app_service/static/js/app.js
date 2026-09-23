document.addEventListener('DOMContentLoaded', () => {
  // ============================================================
  // DOM Elements - Navigation & Modes
  // ============================================================
  const tabBtnEval = document.getElementById('tab-btn-eval');
  const tabBtnChat = document.getElementById('tab-btn-chat');
  const tabBtnLegacyComp = document.getElementById('tab-btn-legacy-comp');

  const modelEvalView = document.getElementById('model-eval-view');
  const standardView = document.getElementById('standard-view');
  const comparisonView = document.getElementById('comparison-view');

  const legacyControlPanel = document.getElementById('legacy-control-panel');
  const legacySuggestions = document.getElementById('legacy-suggestions');
  const chatQueryDock = document.getElementById('chat-query-dock');

  // ============================================================
  // DOM Elements - Live Model Evaluation Dashboard
  // ============================================================
  const evalForm = document.getElementById('eval-form');
  const evalQuestionInput = document.getElementById('eval-question-input');
  const evalModeRag = document.getElementById('eval-mode-rag');
  const evalModeDirect = document.getElementById('eval-mode-direct');
  const labelModeRag = document.getElementById('label-mode-rag');
  const labelModeDirect = document.getElementById('label-mode-direct');
  const evalPrioritySelect = document.getElementById('eval-priority-select');
  const priorityHintText = document.getElementById('priority-hint-text');

  const btnCompareModels = document.getElementById('btn-compare-models');
  const btnCompareText = document.getElementById('btn-compare-text');
  const evalBtnSpinner = document.getElementById('eval-btn-spinner');

  const evalProgressCard = document.getElementById('eval-progress-card');
  const evalProgressStep = document.getElementById('eval-progress-step');
  const evalProgressPct = document.getElementById('eval-progress-pct');
  const evalProgressFill = document.getElementById('eval-progress-fill');
  const evalProgressSub = document.getElementById('eval-progress-sub');

  const evalErrorCard = document.getElementById('eval-error-card');
  const evalErrorMessage = document.getElementById('eval-error-message');

  const evalResultsWrapper = document.getElementById('eval-results-wrapper');
  const recVerdictText = document.getElementById('rec-verdict-text');
  const recPriorityLabel = document.getElementById('rec-priority-label');
  const recSummaryText = document.getElementById('rec-summary-text');
  const recEvidenceText = document.getElementById('rec-evidence-text');
  const recTradeoffsText = document.getElementById('rec-tradeoffs-text');
  const evalTableTbody = document.getElementById('eval-table-tbody');
  const evalModelCardsGrid = document.getElementById('eval-model-cards-grid');

  const btnToggleExplainer = document.getElementById('btn-toggle-explainer');
  const metricsExplainerPanel = document.getElementById('metrics-explainer-panel');
  const explainerToggleText = document.getElementById('explainer-toggle-text');

  // ============================================================
  // DOM Elements - Chat & Legacy Compare
  // ============================================================
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

  const resultModelBadge = document.getElementById('result-model-badge');
  const resultRagBadge = document.getElementById('result-rag-badge');
  const resultLatencyPill = document.getElementById('result-latency-pill');
  const answerContent = document.getElementById('answer-content');
  const citationsWrapper = document.getElementById('citations-wrapper');
  const citationsList = document.getElementById('citations-list');
  const btnCopyAnswer = document.getElementById('btn-copy-answer');

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

  // Modals
  const btnOpenKb = document.getElementById('btn-open-kb');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const kbModal = document.getElementById('kb-modal');
  const kbDocList = document.getElementById('kb-doc-list');
  const kbCountPill = document.getElementById('kb-count-pill');

  const btnOpenEval = document.getElementById('btn-open-eval');
  const btnCloseEvalModal = document.getElementById('btn-close-eval-modal');
  const evalModal = document.getElementById('eval-modal');

  // State Cache
  let currentRawAnswer = '';
  let lastEvaluationData = null;
  let activeView = 'model-eval-view';

  // Priority Hints dictionary
  const priorityHints = {
    'balanced': 'Harmonizes accuracy, relevance, low hallucination, speed, and memory.',
    'fastest': 'Prioritizes lowest response latency and highest generation throughput.',
    'accuracy': 'Emphasizes exact factual adherence to official university regulations.',
    'memory': 'Prefers lightweight models that minimize RAM consumption (ideal for 1GB EC2).',
    'relevance': 'Prioritizes context grounding, retrieval overlap, and hallucination resistance.'
  };

  // 1. Initial Setup
  checkHealth();
  loadDocuments();

  // ============================================================
  // 2. Navigation View Switching
  // ============================================================
  function switchView(viewName) {
    activeView = viewName;

    // Update Tab Buttons
    [tabBtnEval, tabBtnChat, tabBtnLegacyComp].forEach(btn => {
      if (btn) btn.classList.remove('active');
    });

    // Hide all views
    if (modelEvalView) modelEvalView.style.display = 'none';
    if (standardView) standardView.style.display = 'none';
    if (comparisonView) comparisonView.style.display = 'none';

    if (viewName === 'model-eval-view') {
      if (tabBtnEval) tabBtnEval.classList.add('active');
      if (modelEvalView) modelEvalView.style.display = 'flex';
      if (legacyControlPanel) legacyControlPanel.style.display = 'none';
      if (legacySuggestions) legacySuggestions.style.display = 'none';
      if (chatQueryDock) chatQueryDock.style.display = 'none';
    } else if (viewName === 'standard-view') {
      if (tabBtnChat) tabBtnChat.classList.add('active');
      if (standardView) standardView.style.display = 'block';
      if (legacyControlPanel) legacyControlPanel.style.display = 'flex';
      if (legacySuggestions) legacySuggestions.style.display = 'flex';
      if (chatQueryDock) chatQueryDock.style.display = 'block';
    } else if (viewName === 'comparison-view') {
      if (tabBtnLegacyComp) tabBtnLegacyComp.classList.add('active');
      if (comparisonView) comparisonView.style.display = 'flex';
      if (legacyControlPanel) legacyControlPanel.style.display = 'flex';
      if (legacySuggestions) legacySuggestions.style.display = 'flex';
      if (chatQueryDock) chatQueryDock.style.display = 'block';
    }
  }

  if (tabBtnEval) tabBtnEval.addEventListener('click', () => switchView('model-eval-view'));
  if (tabBtnChat) tabBtnChat.addEventListener('click', () => switchView('standard-view'));
  if (tabBtnLegacyComp) tabBtnLegacyComp.addEventListener('click', () => switchView('comparison-view'));

  // ============================================================
  // 3. Live Model Evaluation Form Handlers
  // ============================================================

  // Mode radio toggle styling
  if (evalModeRag && evalModeDirect) {
    evalModeRag.addEventListener('change', () => {
      labelModeRag.classList.add('active');
      labelModeDirect.classList.remove('active');
    });
    evalModeDirect.addEventListener('change', () => {
      labelModeDirect.classList.add('active');
      labelModeRag.classList.remove('active');
    });
  }

  // Priority dropdown change hint
  if (evalPrioritySelect && priorityHintText) {
    evalPrioritySelect.addEventListener('change', (e) => {
      priorityHintText.textContent = priorityHints[e.target.value] || '';
    });
  }

  // Suggestion chips in evaluation view
  document.querySelectorAll('.eval-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-q');
      if (evalQuestionInput) {
        evalQuestionInput.value = q;
        evalForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  // "Explain These Metrics" Accordion Toggle
  if (btnToggleExplainer && metricsExplainerPanel) {
    btnToggleExplainer.addEventListener('click', () => {
      const isHidden = metricsExplainerPanel.style.display === 'none';
      metricsExplainerPanel.style.display = isHidden ? 'block' : 'none';
      explainerToggleText.textContent = isHidden ? 'Hide Metric Explanations' : 'Explain These Metrics';
    });
  }

  // Form Submit Handler
  if (evalForm) {
    evalForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const query = evalQuestionInput.value.trim();
      if (!query) return;

      // Read selected models
      const selectedModelCheckboxes = document.querySelectorAll('input[name="eval-models"]:checked');
      const selectedModels = Array.from(selectedModelCheckboxes).map(cb => cb.value);

      if (selectedModels.length === 0) {
        showEvalError('Please select at least one model to evaluate.');
        return;
      }

      const useRag = evalModeRag ? evalModeRag.checked : true;
      const priority = evalPrioritySelect ? evalPrioritySelect.value : 'balanced';

      startEvalLoading(selectedModels.length);

      try {
        const response = await fetch('/api/compare-models', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            models: selectedModels,
            use_rag: useRag,
            priority: priority,
            top_k: 3
          })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Evaluation server returned status ${response.status}`);
        }

        const data = await response.json();
        lastEvaluationData = data;
        renderEvaluationResults(data);
      } catch (err) {
        console.error('Model evaluation error:', err);
        showEvalError(err.message || 'Failed to complete model evaluation.');
      } finally {
        stopEvalLoading();
      }
    });
  }

  // Loading indicator with sequential progress animation
  let progressInterval = null;

  function startEvalLoading(numModels) {
    if (evalErrorCard) evalErrorCard.style.display = 'none';
    if (btnCompareModels) btnCompareModels.disabled = true;
    if (btnCompareText) btnCompareText.textContent = 'Evaluating Models...';
    if (evalBtnSpinner) evalBtnSpinner.style.display = 'block';
    if (evalProgressCard) evalProgressCard.style.display = 'flex';
    if (evalResultsWrapper) evalResultsWrapper.style.opacity = '0.35';

    let currentPct = 5;
    updateProgressUI(currentPct, 'Step 1/3: Retrieving university context from vector store...');

    // Pace ticks proportionally to number of models being evaluated
    const tickMs = Math.max(240, numModels * 110);

    progressInterval = setInterval(() => {
      if (currentPct < 30) {
        currentPct += 5;
        updateProgressUI(currentPct, 'Step 1/3: Retrieving university context from vector store...');
      } else if (currentPct < 70) {
        currentPct += 3;
        updateProgressUI(currentPct, `Step 2/3: Sequentially executing ${numModels} models on this query...`);
      } else if (currentPct < 88) {
        currentPct += 2;
        updateProgressUI(currentPct, 'Step 3/3: Evaluating factual accuracy & hallucination risk...');
      } else if (currentPct < 94) {
        currentPct += 1;
        updateProgressUI(currentPct, 'Synthesizing transparent evidence-based recommendation...');
      }
    }, tickMs);
  }

  function stopEvalLoading() {
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
    updateProgressUI(100, 'Evaluation complete!');
    if (btnCompareModels) btnCompareModels.disabled = false;
    if (btnCompareText) btnCompareText.textContent = 'Compare Models';
    if (evalBtnSpinner) evalBtnSpinner.style.display = 'none';
    if (evalResultsWrapper) evalResultsWrapper.style.opacity = '1';

    setTimeout(() => {
      if (evalProgressCard) evalProgressCard.style.display = 'none';
    }, 300);
  }

  function updateProgressUI(pct, text) {
    if (evalProgressPct) evalProgressPct.textContent = `${pct}%`;
    if (evalProgressFill) evalProgressFill.style.width = `${pct}%`;
    if (evalProgressStep) evalProgressStep.textContent = text;
  }

  function showEvalError(msg) {
    if (evalErrorCard) {
      evalErrorCard.style.display = 'flex';
      if (evalErrorMessage) evalErrorMessage.textContent = msg;
    }
    if (evalResultsWrapper) evalResultsWrapper.style.display = 'none';
  }

  // ============================================================
  // 4. Render Live Evaluation Results
  // ============================================================
  function renderEvaluationResults(data) {
    if (!data) return;

    if (evalResultsWrapper) {
      evalResultsWrapper.style.display = 'flex';
      // Smooth scroll into view
      evalResultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    renderRecommendation(data.recommendation, data.model_results, data.priority);
    renderComparisonTable(data.model_results, data.retrieval);
    renderModelCards(data.model_results, data.retrieval);
  }

  // Render Section D: Recommendation Banner
  function renderRecommendation(rec, models, activePriority) {
    if (!rec) return;

    if (recVerdictText) recVerdictText.textContent = rec.verdict || 'Model Recommendation';
    if (recPriorityLabel) {
      const pLabels = {
        'balanced': 'Priority: Balanced Performance',
        'fastest': 'Priority: Fastest Response',
        'accuracy': 'Priority: Most Accurate Answer',
        'memory': 'Priority: Lowest Memory Usage',
        'relevance': 'Priority: Most Relevant & Grounded'
      };
      recPriorityLabel.textContent = pLabels[activePriority] || `Priority: ${activePriority}`;
    }

    if (recSummaryText) recSummaryText.innerHTML = formatMarkdown(rec.summary);
    if (recEvidenceText) recEvidenceText.innerHTML = formatMarkdown(rec.evidence);
    if (recTradeoffsText) recTradeoffsText.innerHTML = formatMarkdown(rec.trade_offs);

    // Setup interactive priority pill buttons
    document.querySelectorAll('.priority-pill').forEach(pill => {
      const p = pill.getAttribute('data-priority');
      if (p === activePriority) {
        pill.classList.add('active');
      } else {
        pill.classList.remove('active');
      }

      // Remove old listeners by cloning
      const newPill = pill.cloneNode(true);
      pill.parentNode.replaceChild(newPill, pill);

      newPill.addEventListener('click', () => {
        const newPriority = newPill.getAttribute('data-priority');
        recalculateRecommendationClient(newPriority);
      });
    });
  }

  // Client-side instant recalculation of recommendation when priority pill is clicked
  function recalculateRecommendationClient(newPriority) {
    if (!lastEvaluationData || !lastEvaluationData.model_results) return;

    const successfulModels = lastEvaluationData.model_results.filter(m => m.status === 'success');
    if (successfulModels.length === 0) return;

    const minLat = Math.min(...successfulModels.map(m => m.latency_ms));
    const minRam = Math.min(...successfulModels.map(m => m.ram_usage_mb));
    const hasAccuracy = successfulModels.some(m => m.accuracy && m.accuracy.score !== null);

    const scores = {};
    successfulModels.forEach(m => {
      const speedScore = minLat / Math.max(1.0, m.latency_ms);
      const memScore = minRam / Math.max(1.0, m.ram_usage_mb);
      const relScore = (m.relevance ? m.relevance.score : 50.0) / 100.0;
      const halluRes = Math.max(0.0, 1.0 - ((m.hallucination ? m.hallucination.score : 50.0) / 100.0));
      const accScore = (m.accuracy && m.accuracy.score !== null) ? (m.accuracy.score / 100.0) : relScore;

      let total = 0.0;
      if (newPriority === 'fastest') {
        total = 0.60 * speedScore + 0.20 * relScore + 0.15 * halluRes + 0.05 * memScore;
      } else if (newPriority === 'accuracy') {
        total = hasAccuracy ? (0.50 * accScore + 0.30 * halluRes + 0.20 * relScore) : (0.45 * relScore + 0.45 * halluRes + 0.10 * speedScore);
      } else if (newPriority === 'memory') {
        total = 0.50 * memScore + 0.25 * speedScore + 0.15 * halluRes + 0.10 * relScore;
      } else if (newPriority === 'relevance') {
        total = 0.40 * relScore + 0.40 * halluRes + 0.20 * (hasAccuracy ? accScore : speedScore);
      } else { // balanced
        total = hasAccuracy ? (0.30 * accScore + 0.25 * relScore + 0.25 * halluRes + 0.10 * speedScore + 0.10 * memScore) : (0.35 * relScore + 0.35 * halluRes + 0.15 * speedScore + 0.15 * memScore);
      }
      scores[m.model_id] = total;
    });

    const sorted = [...successfulModels].sort((a, b) => scores[b.model_id] - scores[a.model_id]);
    const winner = sorted[0];
    const runnerUp = sorted[1] || sorted[0];
    const isTie = sorted.length > 1 && (scores[winner.model_id] - scores[runnerUp.model_id]) < 0.04;

    const pLabels = {
      'balanced': 'Balanced Performance',
      'fastest': 'Fastest Response',
      'accuracy': 'Most Accurate Answer',
      'memory': 'Lowest Memory Usage',
      'relevance': 'Most Relevant & Grounded'
    };

    const verdict = isTie ? `Near Tie: ${winner.model_name} & ${runnerUp.model_name}` : `Recommended: ${winner.model_name}`;
    const summary = isTie ?
      `Both **${winner.model_name}** and **${runnerUp.model_name}** perform virtually identically under the **${pLabels[newPriority]}** profile.` :
      `**${winner.model_name}** is the most suitable model for this question under the **${pLabels[newPriority]}** criteria.`;

    const evidenceParts = [];
    if (hasAccuracy && winner.accuracy && winner.accuracy.score !== null) {
      evidenceParts.push(`achieved ${winner.accuracy.score}% accuracy against official university ground truth`);
    } else {
      evidenceParts.push(`delivered ${winner.relevance.score}% topical relevance`);
    }
    evidenceParts.push(`responded in ${winner.latency_ms} ms`);
    evidenceParts.push(`demonstrated ${winner.hallucination.risk_level.toLowerCase()}`);

    const evidence = `Under the ${pLabels[newPriority]} priority, this model scored highest because it ${evidenceParts.join(', ')}.`;

    const fastestM = [...successfulModels].sort((a, b) => a.latency_ms - b.latency_ms)[0];
    const lightestM = [...successfulModels].sort((a, b) => a.ram_usage_mb - b.ram_usage_mb)[0];

    const tradeOffParts = [];
    if (winner.model_id !== fastestM.model_id) {
      tradeOffParts.push(`**${fastestM.model_name}** was faster by ${Math.round(winner.latency_ms - fastestM.latency_ms)} ms, but **${winner.model_name}** scored higher on quality.`);
    }
    if (winner.model_id !== lightestM.model_id) {
      tradeOffParts.push(`For strictly constrained 1GB AWS servers, **${lightestM.model_name}** remains the safest zero-OOM choice (${lightestM.ram_usage_mb} MB).`);
    } else {
      tradeOffParts.push(`**${winner.model_name}** also minimizes memory usage (${winner.ram_usage_mb} MB), safeguarding against cloud OOM terminations.`);
    }

    const updatedRec = {
      recommended_model_id: winner.model_id,
      recommended_model_name: winner.model_name,
      verdict: verdict,
      summary: summary,
      evidence: evidence,
      trade_offs: tradeOffParts.join(' ')
    };

    renderRecommendation(updatedRec, lastEvaluationData.model_results, newPriority);
    highlightWinnerCard(winner.model_id);
  }

  function highlightWinnerCard(winnerId) {
    document.querySelectorAll('.eval-model-card').forEach(c => {
      const mId = c.getAttribute('data-model-id');
      if (mId === winnerId) {
        c.classList.add('winner-card');
      } else {
        c.classList.remove('winner-card');
      }
    });
  }

  // Render Section B: Comparison Table
  function renderComparisonTable(models, retrieval) {
    if (!evalTableTbody || !models) return;

    evalTableTbody.innerHTML = models.map(m => {
      const isSuccess = m.status === 'success';

      // Latency badge
      const latClass = m.latency_ms < 300 ? 'metric-green' : (m.latency_ms < 600 ? 'metric-yellow' : 'metric-red');
      const latDisplay = isSuccess ? `${m.latency_ms} ms` : 'Failed';

      // Accuracy badge
      let accClass = 'metric-gray';
      let accDisplay = 'Unverified';
      if (m.accuracy) {
        if (m.accuracy.status === 'verified') {
          accClass = m.accuracy.score >= 70 ? 'metric-green' : (m.accuracy.score >= 40 ? 'metric-yellow' : 'metric-red');
          accDisplay = `${m.accuracy.score}%`;
        } else {
          accDisplay = 'Unverified';
        }
      }

      // Relevance badge
      let relClass = 'metric-gray';
      let relDisplay = '--';
      if (m.relevance) {
        relClass = m.relevance.score >= 70 ? 'metric-green' : (m.relevance.score >= 40 ? 'metric-yellow' : 'metric-red');
        relDisplay = `${m.relevance.score}%`;
      }

      // Hallucination badge
      let halluClass = 'metric-gray';
      let halluDisplay = '--';
      if (m.hallucination) {
        if (m.hallucination.risk_level === 'Low Risk') halluClass = 'metric-green';
        else if (m.hallucination.risk_level === 'Moderate Risk') halluClass = 'metric-yellow';
        else halluClass = 'metric-red';
        halluDisplay = m.hallucination.risk_level;
      }

      // RAM Usage
      const ramClass = m.ram_usage_mb <= 600 ? 'metric-green' : (m.ram_usage_mb <= 1000 ? 'metric-yellow' : 'metric-red');

      // Retrieval status
      let retDisplay = 'N/A (Direct)';
      let retClass = 'metric-gray';
      if (retrieval && retrieval.chunks_retrieved > 0) {
        if (retrieval.reference_available) {
          retDisplay = retrieval.retrieval_hit ? '✓ Hit (Top-3)' : '✗ Missed';
          retClass = retrieval.retrieval_hit ? 'metric-green' : 'metric-red';
        } else {
          retDisplay = `${retrieval.chunks_retrieved} Chunks`;
          retClass = 'metric-green';
        }
      }

      const snippet = m.answer ? escapeHtml(m.answer.replace(/\n+/g, ' ').slice(0, 80) + '...') : 'No output generated';

      return `
        <tr>
          <td>
            <div class="table-model-name">
              <span>${escapeHtml(m.model_name)}</span>
              <span class="table-model-badge">${escapeHtml(m.badge)}</span>
            </div>
          </td>
          <td>
            <div class="table-ans-snippet" title="${escapeHtml(m.answer)}">${snippet}</div>
          </td>
          <td>
            <span class="metric-pill-cell ${latClass}">⏱️ ${latDisplay}</span>
          </td>
          <td>
            <span class="metric-pill-cell ${accClass}" title="${escapeHtml(m.accuracy ? m.accuracy.display : '')}">🎯 ${accDisplay}</span>
          </td>
          <td>
            <span class="metric-pill-cell ${relClass}">📊 ${relDisplay}</span>
          </td>
          <td>
            <span class="metric-pill-cell ${halluClass}">🛡️ ${halluDisplay}</span>
          </td>
          <td>
            <span class="metric-pill-cell ${ramClass}">💾 ${m.ram_usage_mb} MB</span>
          </td>
          <td>
            <span class="metric-pill-cell ${retClass}">${retDisplay}</span>
          </td>
        </tr>
      `;
    }).join('');
  }

  // Render Section C: Side-by-Side Model Cards
  function renderModelCards(models, retrieval) {
    if (!evalModelCardsGrid || !models) return;

    const winnerId = lastEvaluationData && lastEvaluationData.recommendation ? lastEvaluationData.recommendation.recommended_model_id : null;

    evalModelCardsGrid.innerHTML = models.map(m => {
      const isWinner = m.model_id === winnerId;
      const winnerBadge = isWinner ? '<span class="winner-tag">🏆 Best Fit</span>' : '';

      // Latency pills
      const latPill = `<span class="metric-pill-cell metric-green">⏱️ ${m.latency_ms} ms (Gen: ${m.generation_latency_ms} ms)</span>`;

      // Accuracy pill
      let accPill = '';
      if (m.accuracy && m.accuracy.status === 'verified') {
        const cls = m.accuracy.score >= 70 ? 'metric-green' : 'metric-yellow';
        accPill = `<span class="metric-pill-cell ${cls}" title="Ground truth facts matched: ${m.accuracy.matched_facts.join(', ')}">🎯 ${m.accuracy.score}% Verified</span>`;
      } else {
        accPill = `<span class="metric-pill-cell metric-gray" title="No ground truth reference answer available for this question">🎯 Accuracy Unverified</span>`;
      }

      // Relevance pill
      const relCls = m.relevance && m.relevance.score >= 70 ? 'metric-green' : 'metric-yellow';
      const relPill = `<span class="metric-pill-cell ${relCls}">📊 Rel: ${m.relevance ? m.relevance.score : 0}%</span>`;

      // Hallucination pill
      let halluCls = 'metric-green';
      if (m.hallucination) {
        if (m.hallucination.risk_level === 'Moderate Risk') halluCls = 'metric-yellow';
        else if (m.hallucination.risk_level === 'High Risk') halluCls = 'metric-red';
      }
      const halluPill = `<span class="metric-pill-cell ${halluCls}">🛡️ ${m.hallucination ? m.hallucination.risk_level : 'Evaluated'}</span>`;

      // Warnings
      let warningsHtml = '';
      if (m.warnings && m.warnings.length > 0) {
        warningsHtml = `
          <div class="eval-card-warning">
            ⚠️ <strong>Notice:</strong> ${m.warnings.map(w => escapeHtml(w)).join(' • ')}
          </div>
        `;
      }

      // Citations / View Sources Drawer
      let citationsHtml = '';
      if (retrieval && retrieval.citations && retrieval.citations.length > 0) {
        const chunkItems = retrieval.citations.map((c, idx) => `
          <div class="citation-chunk-item">
            <div class="citation-chunk-header">
              <span>[#${idx + 1}] ${escapeHtml(c.document_title)} • ${escapeHtml(c.section_title)}</span>
              <span>Sim: ${(c.similarity_score * 100).toFixed(0)}%</span>
            </div>
            <div class="citation-chunk-snippet">${escapeHtml(c.snippet)}</div>
          </div>
        `).join('');

        citationsHtml = `
          <div class="eval-card-citations">
            <button type="button" class="citations-toggle-btn" onclick="toggleCardCitations('${m.model_id}')">
              <span>📚 View Retrieved Sources (${retrieval.chunks_retrieved} Chunks)</span>
              <span id="citation-arrow-${m.model_id}">▼</span>
            </button>
            <div class="citations-drawer-content" id="citations-drawer-${m.model_id}" style="display: none;">
              ${chunkItems}
            </div>
          </div>
        `;
      }

      return `
        <div class="eval-model-card ${isWinner ? 'winner-card' : ''}" data-model-id="${escapeHtml(m.model_id)}">
          <div class="eval-card-top-bar">
            <div class="eval-card-model-title">
              <h4>${escapeHtml(m.model_name)}</h4>
              <span>${escapeHtml(m.badge)}</span>
            </div>
            ${winnerBadge}
          </div>

          <div class="eval-card-metrics-bar">
            ${latPill}
            ${accPill}
            ${relPill}
            ${halluPill}
          </div>

          <div class="eval-card-body">
            ${warningsHtml}
            <div class="eval-card-answer">
              ${formatMarkdown(m.answer)}
            </div>
            ${citationsHtml}
          </div>

          <div class="eval-card-resources">
            <span>💻 <strong>Resource Scope:</strong> Model Footprint: ${m.ram_usage_mb} MB | Gateway RSS: ${m.ram_scope ? m.ram_scope.split('|')[0].trim() : 'Active'} | Shared Ollama Engine</span>
          </div>
        </div>
      `;
    }).join('');
  }

  // Global helper for toggling citations in model cards
  window.toggleCardCitations = function(modelId) {
    const drawer = document.getElementById(`citations-drawer-${modelId}`);
    const arrow = document.getElementById(`citation-arrow-${modelId}`);
    if (drawer) {
      const isHidden = drawer.style.display === 'none';
      drawer.style.display = isHidden ? 'flex' : 'none';
      if (arrow) arrow.textContent = isHidden ? '▲' : '▼';
    }
  };

  // ============================================================
  // 5. Standard Query Chat Handlers (Preserved)
  // ============================================================
  if (ragToggle) {
    ragToggle.addEventListener('change', (e) => {
      const isChecked = e.target.checked;
      if (ragToggleText) ragToggleText.textContent = isChecked ? 'RAG Enabled' : 'RAG Disabled';
      if (resultRagBadge) {
        resultRagBadge.textContent = isChecked ? 'RAG Active' : 'Direct LLM';
        if (isChecked) resultRagBadge.classList.remove('disabled');
        else resultRagBadge.classList.add('disabled');
      }
    });
  }

  if (btnToggleCompare) {
    btnToggleCompare.addEventListener('click', () => {
      if (activeView === 'comparison-view') {
        switchView('standard-view');
      } else {
        switchView('comparison-view');
      }
    });
  }

  // Suggestions in chat view
  document.querySelectorAll('#suggestion-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-q');
      if (queryInput) {
        queryInput.value = q;
        queryForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  if (queryForm) {
    queryForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const query = queryInput.value.trim();
      if (!query) return;

      setChatLoading(true);

      try {
        if (activeView === 'comparison-view') {
          await executeCompareQuery(query);
        } else {
          await executeStandardQuery(query);
        }
      } catch (err) {
        console.error(err);
        renderChatError(err.message || 'Error occurred while communicating with services');
      } finally {
        setChatLoading(false);
      }
    });
  }

  async function executeStandardQuery(query) {
    const model = modelSelector ? modelSelector.value : 'qwen2.5:0.5b';
    const useRag = ragToggle ? ragToggle.checked : true;

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

    if (!response.ok) throw new Error(`Server returned status ${response.status}`);
    const data = await response.json();
    renderStandardResponse(data);
  }

  async function executeCompareQuery(query) {
    const model = modelSelector ? modelSelector.value : 'qwen2.5:0.5b';

    if (compRagAnswer) compRagAnswer.innerHTML = '<div class="spinner"></div> Generating grounded answer...';
    if (compDirectAnswer) compDirectAnswer.innerHTML = '<div class="spinner"></div> Generating parametric answer...';

    const response = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        model: model
      })
    });

    if (!response.ok) throw new Error(`Comparison API returned status ${response.status}`);
    const data = await response.json();
    renderCompareResponse(data);
  }

  function renderStandardResponse(data) {
    currentRawAnswer = data.answer;
    if (resultModelBadge) resultModelBadge.textContent = `Model: ${data.model}`;
    if (resultLatencyPill) resultLatencyPill.textContent = `Latency: ${data.latency_ms} ms`;

    if (answerContent) answerContent.innerHTML = formatMarkdown(data.answer);

    if (data.citations && data.citations.length > 0 && citationsWrapper && citationsList) {
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
    } else if (citationsWrapper) {
      citationsWrapper.style.display = 'none';
    }
  }

  function renderCompareResponse(data) {
    const rag = data.rag_response;
    const direct = data.direct_response;

    if (compRagLatency) compRagLatency.textContent = `${rag.latency_ms} ms`;
    if (compDirectLatency) compDirectLatency.textContent = `${direct.latency_ms} ms`;

    if (compRagAnswer) compRagAnswer.innerHTML = formatMarkdown(rag.answer);
    if (compDirectAnswer) compDirectAnswer.innerHTML = formatMarkdown(direct.answer);

    if (compRagCitations) {
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
  }

  function renderChatError(msg) {
    if (answerContent) {
      answerContent.innerHTML = `
        <div style="color: #fb7185; padding: 16px; background: rgba(244, 63, 94, 0.1); border-radius: 8px; border: 1px solid rgba(244, 63, 94, 0.3);">
          <strong>Error:</strong> ${escapeHtml(msg)}
        </div>
      `;
    }
  }

  function setChatLoading(isLoading) {
    if (isLoading) {
      if (btnSubmit) btnSubmit.disabled = true;
      if (btnSendText) btnSendText.style.display = 'none';
      if (btnSendIcon) btnSendIcon.style.display = 'none';
      if (btnSpinner) btnSpinner.style.display = 'block';
    } else {
      if (btnSubmit) btnSubmit.disabled = false;
      if (btnSendText) btnSendText.style.display = 'inline';
      if (btnSendIcon) btnSendIcon.style.display = 'inline';
      if (btnSpinner) btnSpinner.style.display = 'none';
    }
  }

  // Copy Answer
  if (btnCopyAnswer) {
    btnCopyAnswer.addEventListener('click', () => {
      if (!currentRawAnswer) return;
      navigator.clipboard.writeText(currentRawAnswer).then(() => {
        const orig = btnCopyAnswer.innerHTML;
        btnCopyAnswer.innerHTML = '✓ Copied';
        setTimeout(() => { btnCopyAnswer.innerHTML = orig; }, 2000);
      });
    });
  }

  // ============================================================
  // 6. Modals Controls
  // ============================================================
  if (btnOpenKb && kbModal) {
    btnOpenKb.addEventListener('click', () => kbModal.classList.add('active'));
  }
  if (btnCloseModal && kbModal) {
    btnCloseModal.addEventListener('click', () => kbModal.classList.remove('active'));
  }
  if (kbModal) {
    kbModal.addEventListener('click', (e) => {
      if (e.target === kbModal) kbModal.classList.remove('active');
    });
  }

  async function loadBenchmarkMeta() {
    try {
      const res = await fetch('/api/benchmark-meta');
      if (res.ok) {
        const data = await res.json();
        const countSpan = document.getElementById('eval-dataset-count');
        if (countSpan && data.total_tasks) {
          countSpan.textContent = data.total_tasks;
        }
      }
    } catch (e) {
      console.warn('Could not load benchmark metadata:', e);
    }
  }

  if (btnOpenEval && evalModal) {
    btnOpenEval.addEventListener('click', () => {
      loadBenchmarkMeta();
      evalModal.classList.add('active');
    });
  }
  if (btnCloseEvalModal && evalModal) {
    btnCloseEvalModal.addEventListener('click', () => evalModal.classList.remove('active'));
  }
  if (evalModal) {
    evalModal.addEventListener('click', (e) => {
      if (e.target === evalModal) evalModal.classList.remove('active');
    });
  }

  // Modal Tabs Switching
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

  // ============================================================
  // 7. System Health & Knowledge Base Loading
  // ============================================================
  async function checkHealth() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const s = await res.json();
        if (ragStatusLabel && ragStatusBadge) {
          if (s.rag_service) {
            ragStatusLabel.textContent = 'Active';
            ragStatusBadge.querySelector('.pulse-dot').className = 'pulse-dot active';
          } else {
            ragStatusLabel.textContent = 'Offline';
            ragStatusBadge.querySelector('.pulse-dot').className = 'pulse-dot warning';
          }
        }
        if (ollamaStatusLabel && ollamaStatusBadge) {
          if (s.ollama_service) {
            ollamaStatusLabel.textContent = 'Online';
            ollamaStatusBadge.querySelector('.pulse-dot').className = 'pulse-dot active';
          } else {
            ollamaStatusLabel.textContent = 'Standby (Simulation Active)';
            ollamaStatusBadge.querySelector('.pulse-dot').className = 'pulse-dot warning';
          }
        }
        if (s.dataset_tasks) {
          const countSpan = document.getElementById('eval-dataset-count');
          if (countSpan) countSpan.textContent = s.dataset_tasks;
        }
      }
    } catch (e) {
      if (ragStatusLabel) ragStatusLabel.textContent = 'Unreachable';
      if (ollamaStatusLabel) ollamaStatusLabel.textContent = 'Unreachable';
    }
  }

  async function loadDocuments() {
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        const docs = data.documents || [];
        if (kbCountPill) kbCountPill.textContent = `${docs.length} Docs`;

        if (kbDocList) {
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
      }
    } catch (e) {
      if (kbDocList) kbDocList.innerHTML = '<p class="modal-sub">Failed to retrieve indexed documents.</p>';
    }
  }

  // ============================================================
  // 8. Utility Formatters
  // ============================================================
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
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Health ping interval
  setInterval(checkHealth, 10000);
});
