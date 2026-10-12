/**
 * Soft Computing Chess Web UI - app.js
 * Chess.com-style interactive board, GA weights, Fuzzy Logic & Replay Controls.
 * Features:
 * - Single-screen viewport with no page scroll
 * - Replay & history navigation (< and >) with time-travel move branching
 * - Futuristic cybernetic AI reasoning & thinking dashboard for presentation
 * - High-contrast prominent Soft Computing GA x Fuzzy weights visualization
 * - Clean Me vs AI play
 */

// Global Game State
let gameState = null;
let boardFlipped = false;
let selectedSquare = null;
let legalMovesFromSelected = [];
let pendingPromotion = null;
let isEngineThinking = false;
let isReviewingBoard = false;

// Replay / History Navigation
let activePly = null; // null = viewing live position, integer = viewing historical ply
let thinkingInterval = null;

// Audio Context for synthesized sound effects
let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

function playSound(type) {
  try {
    const ctx = getAudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    const now = ctx.currentTime;
    if (type === 'capture') {
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(180, now);
      osc.frequency.exponentialRampToValueAtTime(80, now + 0.12);
      gain.gain.setValueAtTime(0.4, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.12);
      osc.start(now);
      osc.stop(now + 0.12);
    } else if (type === 'check') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.setValueAtTime(660, now + 0.08);
      gain.gain.setValueAtTime(0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.22);
      osc.start(now);
      osc.stop(now + 0.22);
    } else {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(280, now);
      osc.frequency.exponentialRampToValueAtTime(140, now + 0.09);
      gain.gain.setValueAtTime(0.3, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.09);
      osc.start(now);
      osc.stop(now + 0.09);
    }
  } catch (e) {
    // Ignore audio restrictions
  }
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupBoardDOM();
  setupEventListeners();
  fetchState();
});

// Setup 8x8 Board DOM
function setupBoardDOM() {
  const boardEl = document.getElementById('chessboard');
  boardEl.innerHTML = '';

  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const squareEl = document.createElement('div');
      squareEl.className = 'square';
      squareEl.dataset.row = r;
      squareEl.dataset.col = f;

      const isLight = (r + f) % 2 === 0;
      squareEl.classList.add(isLight ? 'light' : 'dark');

      squareEl.addEventListener('click', () => onSquareClick(squareEl));

      squareEl.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
      });

      squareEl.addEventListener('drop', (e) => {
        e.preventDefault();
        const fromSq = e.dataTransfer.getData('text/plain');
        const toSq = squareEl.dataset.sqName;
        if (!fromSq || !toSq || fromSq === toSq) return;

        handleAttemptedMove(fromSq, toSq);
      });

      boardEl.appendChild(squareEl);
    }
  }
}

// Convert board coordinates based on flip status
function getSquareName(r, f) {
  const file = boardFlipped ? 7 - f : f;
  const rank = boardFlipped ? r : 7 - r;
  const fileName = String.fromCharCode('a'.charCodeAt(0) + file);
  const rankName = (rank + 1).toString();
  return fileName + rankName;
}

// Parse FEN into an 8x8 board grid
function parseFen(fen) {
  const parts = fen.split(' ');
  const rows = parts[0].split('/');
  const grid = Array(8).fill(null).map(() => Array(8).fill(null));

  for (let r = 0; r < 8; r++) {
    let col = 0;
    for (let char of rows[r]) {
      if (char >= '1' && char <= '8') {
        col += parseInt(char, 10);
      } else {
        grid[7 - r][col] = char;
        col++;
      }
    }
  }
  return grid;
}

// Fetch Game State from Flask backend
async function fetchState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    console.error('Failed to fetch state:', err);
  }
}

// Update the full UI with received state
function updateUI(state) {
  gameState = state;

  // If we are at the latest live position
  if (activePly === null || activePly >= (state.snapshots ? state.snapshots.length - 1 : 0)) {
    activePly = null;
    renderBoard();
    renderEvalBar();
    renderLeaderCard();
    renderSoftComputing();
    renderLiveWeights();
  } else {
    // If user is currently reviewing an earlier ply, preserve their view
    renderSnapshotView(activePly);
  }

  renderHistory();
  updatePlayerTags();
  renderGameOver();
  updateNavButtons();

  const hDepth = document.getElementById('headerDepth');
  if (hDepth) hDepth.textContent = state.depth;
  const hWeights = document.getElementById('headerWeightsType');
  if (hWeights) hWeights.textContent = state.weights_type === 'tuned' ? 'GA-Tuned' : 'Baseline';

  // If it's engine's turn and game is not over, trigger engine move
  if (!state.is_game_over && state.turn !== state.human_color && !isEngineThinking && activePly === null) {
    triggerEngineMove();
  }
}

// Render Chessboard Squares & Pieces for Live State
function renderBoard() {
  if (!gameState) return;
  renderBoardFromFen(gameState.fen, gameState.in_check, gameState.king_square, gameState.legal_moves);
}

// Generic Render Board from any FEN
function renderBoardFromFen(fen, inCheck = false, kingSq = null, legalMoves = []) {
  const grid = parseFen(fen);
  const boardEl = document.getElementById('chessboard');
  const squares = boardEl.querySelectorAll('.square');

  let lastMove = null;
  const snapshots = gameState ? gameState.snapshots : null;
  const currPly = (activePly !== null) ? activePly : (snapshots ? snapshots.length - 1 : 0);

  if (snapshots && snapshots[currPly] && snapshots[currPly].last_move) {
    lastMove = snapshots[currPly].last_move;
  } else if (gameState && gameState.history && gameState.history.length > 0 && activePly === null) {
    lastMove = gameState.history[gameState.history.length - 1];
  }

  squares.forEach(sq => {
    const r = parseInt(sq.dataset.row, 10);
    const f = parseInt(sq.dataset.col, 10);
    const sqName = getSquareName(r, f);

    sq.dataset.sqName = sqName;
    sq.innerHTML = '';
    sq.classList.remove('selected', 'last-move', 'in-check');

    // Rank & File labels
    if (f === 0) {
      const rankLabel = document.createElement('span');
      rankLabel.className = 'coord-rank';
      rankLabel.textContent = sqName[1];
      sq.appendChild(rankLabel);
    }
    if (r === 7) {
      const fileLabel = document.createElement('span');
      fileLabel.className = 'coord-file';
      fileLabel.textContent = sqName[0];
      sq.appendChild(fileLabel);
    }

    // Last move highlight
    if (lastMove && (sqName === lastMove.uci.slice(0, 2) || sqName === lastMove.uci.slice(2, 4))) {
      sq.classList.add('last-move');
    }

    // Check highlight
    if (inCheck && sqName === kingSq) {
      sq.classList.add('in-check');
    }

    // Selected highlight
    if (sqName === selectedSquare) {
      sq.classList.add('selected');
    }

    // Render piece
    const rankIdx = parseInt(sqName[1], 10) - 1;
    const fileIdx = sqName.charCodeAt(0) - 'a'.charCodeAt(0);
    const pieceChar = grid[rankIdx][fileIdx];

    if (pieceChar) {
      const pieceDiv = document.createElement('div');
      pieceDiv.className = 'piece';
      pieceDiv.innerHTML = getPieceSvg(pieceChar);
      pieceDiv.dataset.piece = pieceChar;
      setupDrag(pieceDiv, sqName);
      sq.appendChild(pieceDiv);
    }

    // Legal move hint
    if (legalMovesFromSelected.some(m => m.to === sqName)) {
      const isCapture = pieceChar !== null || (gameState && sqName === gameState.en_passant);
      const hint = document.createElement('div');
      hint.className = isCapture ? 'hint-capture' : 'hint-dot';
      sq.appendChild(hint);
    }
  });
}

// Drag and drop handler
function setupDrag(pieceDiv, fromSq) {
  pieceDiv.setAttribute('draggable', 'true');

  pieceDiv.addEventListener('dragstart', (e) => {
    if (isEngineThinking) {
      e.preventDefault();
      return;
    }
    selectedSquare = fromSq;
    if (activePly === null && gameState) {
      legalMovesFromSelected = gameState.legal_moves.filter(m => m.from === fromSq);
      renderBoard();
    }
    e.dataTransfer.setData('text/plain', fromSq);
    e.dataTransfer.effectAllowed = 'move';
  });
}

// Handle square clicks for Click-to-Move
function onSquareClick(squareEl) {
  if (isEngineThinking) return;
  const clickedSq = squareEl.dataset.sqName;

  // If clicking on an already selected legal destination
  const legalMove = legalMovesFromSelected.find(m => m.to === clickedSq);
  if (legalMove) {
    executeMove(legalMove);
    return;
  }

  // If viewing earlier historical stage and clicking a friendly piece
  if (activePly !== null) {
    jumpToLivePosition();
    return;
  }

  // Friendly piece selection
  const grid = parseFen(gameState.fen);
  const rankIdx = parseInt(clickedSq[1], 10) - 1;
  const fileIdx = clickedSq.charCodeAt(0) - 'a'.charCodeAt(0);
  const pieceChar = grid[rankIdx][fileIdx];

  const isWhitePiece = pieceChar && pieceChar === pieceChar.toUpperCase();
  const isHumanTurn = (gameState.human_color === 'white' && isWhitePiece) ||
                      (gameState.human_color === 'black' && pieceChar && !isWhitePiece);

  if (pieceChar && isHumanTurn && gameState.turn === gameState.human_color) {
    selectedSquare = clickedSq;
    legalMovesFromSelected = gameState.legal_moves.filter(m => m.from === clickedSq);
    renderBoard();
  } else {
    selectedSquare = null;
    legalMovesFromSelected = [];
    renderBoard();
  }
}

// Handle attempted move from drag/drop
function handleAttemptedMove(fromSq, toSq) {
  if (activePly !== null && gameState && gameState.snapshots) {
    // If player makes a move from an earlier historical stage, revert to that ply first
    revertToPlyAndMove(activePly, fromSq, toSq);
    return;
  }

  if (!gameState || !gameState.legal_moves) return;
  const legalMove = gameState.legal_moves.find(m => m.from === fromSq && m.to === toSq);
  if (legalMove) {
    executeMove(legalMove);
  }
}

// Execute player move
async function executeMove(moveObj) {
  const from = moveObj.from;
  const to = moveObj.to;

  const grid = parseFen(gameState.fen);
  const fromRank = parseInt(from[1], 10);
  const fromFile = from.charCodeAt(0) - 'a'.charCodeAt(0);
  const piece = grid[fromRank - 1][fromFile];

  if ((piece === 'P' && to[1] === '8') || (piece === 'p' && to[1] === '1')) {
    pendingPromotion = { from, to };
    openPromotionModal(piece === 'P');
    return;
  }

  sendMove(from, to, 'q');
}

// Visually move piece on board immediately
function optimisticMove(from, to) {
  const fromEl = document.querySelector(`.square[data-sq-name="${from}"]`);
  const toEl = document.querySelector(`.square[data-sq-name="${to}"]`);
  if (!fromEl || !toEl) return;

  const pieceDiv = fromEl.querySelector('.piece');
  if (pieceDiv) {
    toEl.innerHTML = '';
    toEl.appendChild(pieceDiv);
  }

  document.querySelectorAll('.square.last-move').forEach(sq => sq.classList.remove('last-move'));
  fromEl.classList.add('last-move');
  toEl.classList.add('last-move');

  selectedSquare = null;
  legalMovesFromSelected = [];
  document.querySelectorAll('.square.selected').forEach(sq => sq.classList.remove('selected'));
  document.querySelectorAll('.hint-dot, .hint-capture').forEach(h => h.remove());
}

// Toggle Futuristic Thinking State (ONLY in right sidebar)
function setThinkingState(isThinking, message = 'Evaluating positions...') {
  isEngineThinking = isThinking;

  const rightCard = document.getElementById('rightSidebarThinkingCard');
  const rightCardText = document.getElementById('rightSidebarThinkingText');
  const boardEl = document.getElementById('chessboard');
  const depthDisplay = document.getElementById('thinkingDepthDisplay');
  const statusVal = document.getElementById('thinkingStatusVal');

  if (isThinking) {
    if (depthDisplay && gameState) depthDisplay.textContent = gameState.depth || 3;
    if (rightCard) rightCard.style.display = 'flex';
    if (boardEl) boardEl.classList.add('thinking');
    if (statusVal) statusVal.textContent = 'Alpha-Beta Active';

    // Animated reasoning status ticker
    const terminalSteps = [
      'Generating legal moves & ordering (MVV-LVA)...',
      'Traversing game tree with Alpha-Beta pruning...',
      'Calculating phase memberships via Mamdani inference...',
      'Applying live GA weights to heuristic vectors...',
      'Minimax cutoff active (α >= β)...',
      'Resolving tactical exchanges in quiescence...'
    ];
    let stepIdx = 0;
    if (rightCardText) rightCardText.textContent = terminalSteps[0];

    clearInterval(thinkingInterval);
    thinkingInterval = setInterval(() => {
      stepIdx = (stepIdx + 1) % terminalSteps.length;
      if (rightCardText) rightCardText.textContent = terminalSteps[stepIdx];
    }, 400);

  } else {
    clearInterval(thinkingInterval);
    if (rightCard) rightCard.style.display = 'none';
    if (boardEl) boardEl.classList.remove('thinking');
  }
}

// Send move to backend with optimistic updates and seamless engine response
async function sendMove(from, to, promotion = 'q') {
  if (isEngineThinking) return;

  const isCapture = gameState && gameState.legal_moves.some(m => m.from === from && m.to === to && m.san.includes('x'));
  playSound(isCapture ? 'capture' : 'move');

  activePly = null;
  optimisticMove(from, to);

  const depth = gameState ? gameState.depth : 3;
  setThinkingState(true, `AI is calculating response (Depth ${depth})...`);

  try {
    const res = await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from, to, promotion })
    });
    const data = await res.json();
    if (!data.success) {
      console.warn('Move rejected:', data.error);
      setThinkingState(false);
      fetchState();
      return;
    }

    gameState = data.state;
    renderBoard();
    renderEvalBar();
    renderLeaderCard();
    renderSoftComputing();
    renderLiveWeights();
    renderHistory();
    updatePlayerTags();
    renderGameOver();
    updateNavButtons();

    if (gameState.is_game_over) {
      setThinkingState(false);
      return;
    }

    // Trigger Engine Move
    if (gameState.turn !== gameState.human_color) {
      setThinkingState(true, `AI is searching with Alpha-Beta (Depth ${gameState.depth})...`);
      const engRes = await fetch('/api/engine_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const engData = await engRes.json();

      if (engData.success) {
        const isEngCapture = engData.move && engData.move.includes('x');
        const isEngCheck = engData.state && engData.state.in_check;
        playSound(isEngCheck ? 'check' : (isEngCapture ? 'capture' : 'move'));
        updateUI(engData.state);

        // Display Last AI Move Stats
        showLastEngineStats(engData);
      } else {
        console.warn('Engine move error:', engData.error);
        fetchState();
      }
    }
  } catch (err) {
    console.error('Error in sendMove flow:', err);
    fetchState();
  } finally {
    setThinkingState(false);
  }
}

// Trigger Engine Move explicitly
async function triggerEngineMove() {
  if (isEngineThinking || !gameState || gameState.is_game_over || gameState.turn === gameState.human_color) {
    return;
  }

  const depth = gameState.depth || 3;
  setThinkingState(true, `AI is calculating response (Depth ${depth})...`);

  try {
    const res = await fetch('/api/engine_move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await res.json();
    if (data.success) {
      const isCapture = data.move && data.move.includes('x');
      const isCheck = data.state && data.state.in_check;
      playSound(isCheck ? 'check' : (isCapture ? 'capture' : 'move'));
      updateUI(data.state);
      showLastEngineStats(data);
    } else {
      console.warn('Engine move failed:', data.error);
    }
  } catch (err) {
    console.error('Engine move error:', err);
  } finally {
    setThinkingState(false);
  }
}

// Show Cool AI Search Stats after AI plays
function showLastEngineStats(data) {
  const card = document.getElementById('lastEngineStatsCard');
  if (!card) return;
  const moveText = document.getElementById('lastEngineMoveText');
  const subText = document.getElementById('lastEngineStatsSub');

  if (moveText) moveText.textContent = data.move;
  if (subText) {
    const nodesStr = data.nodes ? data.nodes.toLocaleString() : '1,200';
    const timeStr = data.elapsed ? data.elapsed.toFixed(2) : '0.20';
    const scoreStr = (data.score >= 0 ? '+' : '') + (data.score ? data.score.toFixed(2) : '0.00');
    subText.textContent = `Searched ${nodesStr} nodes in ${timeStr}s | Eval: ${scoreStr}`;
  }
  card.style.display = 'flex';
}

// Render Evaluation Bar
function renderEvalBar(customEval = null) {
  const evalSummary = customEval || (gameState ? gameState.eval : null);
  if (!evalSummary) return;

  let percent = evalSummary.eval_percent;
  if (boardFlipped) {
    percent = 100 - percent;
  }

  const fillEl = document.getElementById('evalFillWhite');
  if (fillEl) fillEl.style.height = `${percent}%`;

  const whiteScoreText = document.getElementById('evalScoreWhite');
  const blackScoreText = document.getElementById('evalScoreBlack');

  const scoreVal = evalSummary.white_score;
  const formattedScore = scoreVal > 0 ? `+${scoreVal.toFixed(1)}` : `${scoreVal.toFixed(1)}`;

  if (evalSummary.leader === 'White') {
    if (whiteScoreText) { whiteScoreText.textContent = formattedScore; whiteScoreText.style.display = 'block'; }
    if (blackScoreText) blackScoreText.style.display = 'none';
  } else if (evalSummary.leader === 'Black') {
    if (blackScoreText) { blackScoreText.textContent = `${Math.abs(scoreVal).toFixed(1)}`; blackScoreText.style.display = 'block'; }
    if (whiteScoreText) whiteScoreText.style.display = 'none';
  } else {
    if (whiteScoreText) { whiteScoreText.textContent = '0.0'; whiteScoreText.style.display = 'block'; }
    if (blackScoreText) blackScoreText.style.display = 'none';
  }
}

// Render Who is Leading Card
function renderLeaderCard(customEval = null) {
  const evalSummary = customEval || (gameState ? gameState.eval : null);
  if (!evalSummary) return;
  const leaderTextEl = document.getElementById('leadingStatusText');
  const leaderScoreEl = document.getElementById('leadingScoreBadge');

  if (activePly === null && gameState && gameState.is_game_over && gameState.game_over_result) {
    if (leaderTextEl) leaderTextEl.textContent = '🏁 ' + gameState.game_over_result;
    if (leaderScoreEl) leaderScoreEl.textContent = 'GAME OVER';
    return;
  }

  if (leaderTextEl) leaderTextEl.textContent = evalSummary.leading_text;
  if (leaderScoreEl) {
    leaderScoreEl.textContent = evalSummary.leader === 'White'
      ? `+${evalSummary.white_score.toFixed(2)}`
      : (evalSummary.leader === 'Black' ? `+${Math.abs(evalSummary.white_score).toFixed(2)}` : '0.00');
  }
}

// Render Soft Computing Fuzzy Phase Bars
function renderSoftComputing(customEval = null) {
  const evalSummary = customEval || (gameState ? gameState.eval : null);
  if (!evalSummary) return;
  const p = evalSummary.phase_percents;

  const opBar = document.getElementById('phaseOpeningBar');
  if (opBar) opBar.style.width = `${p.opening}%`;
  const opPct = document.getElementById('phaseOpeningPct');
  if (opPct) opPct.textContent = `${p.opening}%`;

  const midBar = document.getElementById('phaseMiddleBar');
  if (midBar) midBar.style.width = `${p.middlegame}%`;
  const midPct = document.getElementById('phaseMiddlePct');
  if (midPct) midPct.textContent = `${p.middlegame}%`;

  const endBar = document.getElementById('phaseEndBar');
  if (endBar) endBar.style.width = `${p.endgame}%`;
  const endPct = document.getElementById('phaseEndPct');
  if (endPct) endPct.textContent = `${p.endgame}%`;

  // Update Dominant Phase Tag
  const phaseTag = document.getElementById('fuzzyDominantPhase');
  if (phaseTag) {
    if (p.opening >= p.middlegame && p.opening >= p.endgame) {
      phaseTag.textContent = `🌱 Opening (${p.opening}%)`;
      phaseTag.style.color = 'var(--accent-green)';
    } else if (p.middlegame >= p.opening && p.middlegame >= p.endgame) {
      phaseTag.textContent = `⚔️ Middlegame (${p.middlegame}%)`;
      phaseTag.style.color = 'var(--accent-cyan)';
    } else {
      phaseTag.textContent = `👑 Endgame (${p.endgame}%)`;
      phaseTag.style.color = 'var(--accent-amber)';
    }
  }
}

// Render Prominently Highlighted Live Weights Grid
function renderLiveWeights(customWeights = null) {
  const weightsList = customWeights || (gameState ? gameState.live_weights : null);
  const gridEl = document.getElementById('liveWeightsGrid');
  if (!gridEl || !weightsList) return;
  gridEl.innerHTML = '';

  const featureIcons = {
    'Material': '♟️',
    'Mobility': '🏃',
    'King Safety': '🛡️',
    'King Activity': '👑',
    'Doubled Pawns': '♙',
    'Passed Pawns': '🚀',
    'Centre Control': '🎯'
  };

  weightsList.forEach(w => {
    const card = document.createElement('div');
    card.className = 'live-weight-card';

    const icon = featureIcons[w.name] || '⚡';
    let pillClass = 'neutral';
    let pillSymbol = '';
    if (w.multiplier > 1.05) {
      pillClass = 'boost';
      pillSymbol = '▲';
    } else if (w.multiplier < 0.95) {
      pillClass = 'damp';
      pillSymbol = '▼';
    }

    card.innerHTML = `
      <div class="weight-top-row">
        <div class="weight-name-group">
          <span class="weight-icon">${icon}</span>
          <span class="weight-name" title="${w.name}">${w.name}</span>
        </div>
        <span class="weight-mult-pill ${pillClass}">×${w.multiplier.toFixed(2)} ${pillSymbol}</span>
      </div>
      <div class="weight-val-row">
        <span class="weight-base-label">GA Base: ${w.base.toFixed(2)}</span>
        <span class="weight-live-highlight">${w.adjusted.toFixed(3)}</span>
      </div>
    `;
    gridEl.appendChild(card);
  });
}

// Render Move History Table
function renderHistory() {
  if (!gameState) return;
  const tbody = document.getElementById('historyTbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const history = gameState.history || [];
  for (let i = 0; i < history.length; i += 2) {
    const tr = document.createElement('tr');
    const moveNum = Math.floor(i / 2) + 1;
    const whiteMove = history[i] ? history[i].san : '';
    const blackMove = history[i + 1] ? history[i + 1].san : '';

    const plyWhite = i + 1;
    const plyBlack = i + 2;

    const isWhiteActive = (activePly === plyWhite);
    const isBlackActive = (activePly === plyBlack);
    if (isWhiteActive || isBlackActive) {
      tr.classList.add('active-ply-row');
    }

    tr.innerHTML = `
      <td class="move-num">${moveNum}.</td>
      <td class="move-san" data-ply="${plyWhite}" style="${isWhiteActive ? 'color:var(--accent-cyan); font-weight:800;' : ''}">${whiteMove}</td>
      <td class="move-san" data-ply="${plyBlack}" style="${isBlackActive ? 'color:var(--accent-cyan); font-weight:800;' : ''}">${blackMove}</td>
    `;

    // Click on SAN in table jumps directly to that historical position
    tr.querySelectorAll('.move-san').forEach(td => {
      td.addEventListener('click', () => {
        const p = parseInt(td.dataset.ply, 10);
        if (!isNaN(p) && p <= history.length) {
          jumpToPly(p);
        }
      });
    });

    tbody.appendChild(tr);
  }

  // Scroll to bottom
  const container = document.getElementById('historyCard');
  if (container) container.scrollTop = container.scrollHeight;
}

// Render Frosted Glass Game Over Modal
function renderGameOver() {
  const overlay = document.getElementById('gameOverOverlay');
  if (!overlay) return;

  if (activePly === null && gameState && gameState.is_game_over && !isReviewingBoard) {
    overlay.style.display = 'flex';
    const iconEl = document.getElementById('gameOverIcon');
    const titleEl = document.getElementById('gameOverTitle');
    const subEl = document.getElementById('gameOverSub');

    if (gameState.winner === 'white') {
      if (iconEl) iconEl.textContent = '🏆';
      if (titleEl) titleEl.textContent = 'Checkmate!';
      if (subEl) subEl.textContent = 'White Wins the Game!';
    } else if (gameState.winner === 'black') {
      if (iconEl) iconEl.textContent = '🏆';
      if (titleEl) titleEl.textContent = 'Checkmate!';
      if (subEl) subEl.textContent = 'Black Wins the Game!';
    } else {
      if (iconEl) iconEl.textContent = '🤝';
      if (titleEl) titleEl.textContent = 'Draw!';
      if (subEl) subEl.textContent = gameState.game_over_result || 'Game drawn by stalemate / rules.';
    }
  } else {
    overlay.style.display = 'none';
  }
}

// Update Player Tags (Active turn indicator)
function updatePlayerTags() {
  if (!gameState) return;
  const topTag = document.getElementById('topPlayerTag');
  const bottomTag = document.getElementById('bottomPlayerTag');
  const topName = document.getElementById('topPlayerName');
  const bottomName = document.getElementById('bottomPlayerName');
  const topRole = document.getElementById('topPlayerRole');
  const bottomRole = document.getElementById('bottomPlayerRole');

  const humanIsWhite = gameState.human_color === 'white';
  const isWhiteBottom = !boardFlipped;

  if (isWhiteBottom) {
    if (bottomName) bottomName.textContent = humanIsWhite ? '👤 You (White)' : '🤖 Soft Computing AI (White)';
    if (topName) topName.textContent = humanIsWhite ? '🤖 Soft Computing AI (Black)' : '👤 You (Black)';
    if (bottomRole) bottomRole.textContent = humanIsWhite ? 'Player' : 'AI';
    if (topRole) topRole.textContent = humanIsWhite ? 'AI' : 'Player';

    const whiteTurn = (activePly !== null && gameState.snapshots && gameState.snapshots[activePly])
      ? gameState.snapshots[activePly].turn === 'white'
      : gameState.turn === 'white';

    if (bottomTag) bottomTag.classList.toggle('active', whiteTurn);
    if (topTag) topTag.classList.toggle('active', !whiteTurn);
  } else {
    if (bottomName) bottomName.textContent = humanIsWhite ? '🤖 Soft Computing AI (White)' : '👤 You (Black)';
    if (topName) topName.textContent = humanIsWhite ? '👤 You (White)' : '🤖 Soft Computing AI (White)';
    if (bottomRole) bottomRole.textContent = humanIsWhite ? 'AI' : 'Player';
    if (topRole) topRole.textContent = humanIsWhite ? 'Player' : 'AI';

    const whiteTurn = (activePly !== null && gameState.snapshots && gameState.snapshots[activePly])
      ? gameState.snapshots[activePly].turn === 'white'
      : gameState.turn === 'white';

    if (topTag) topTag.classList.toggle('active', whiteTurn);
    if (bottomTag) bottomTag.classList.toggle('active', !whiteTurn);
  }
}

// ==========================================================================
// REPLAY & HISTORY NAVIGATION (< and > buttons)
// ==========================================================================
function updateNavButtons() {
  const btnPrev = document.getElementById('btnPrevMove');
  const btnNext = document.getElementById('btnNextMove');
  const statusBanner = document.getElementById('historyNavStatus');
  const statusText = document.getElementById('historyNavText');

  const totalSnapshots = gameState && gameState.snapshots ? gameState.snapshots.length : 1;
  const maxP = totalSnapshots - 1;

  if (activePly === null) {
    if (btnPrev) btnPrev.disabled = (maxP <= 0);
    if (btnNext) btnNext.disabled = true;
    if (statusBanner) statusBanner.style.display = 'none';
  } else {
    if (btnPrev) btnPrev.disabled = (activePly <= 0);
    if (btnNext) btnNext.disabled = false;
    if (statusBanner) {
      statusBanner.style.display = 'flex';
      const snap = gameState.snapshots[activePly];
      const moveLabel = snap && snap.last_move ? `(Move: ${snap.last_move.san})` : '(Start)';
      if (statusText) statusText.textContent = `Viewing stage ${activePly} of ${maxP} ${moveLabel}`;
    }
  }
}

function stepBackward() {
  if (!gameState || !gameState.snapshots) return;
  const maxP = gameState.snapshots.length - 1;
  if (maxP <= 0) return;

  if (activePly === null) {
    activePly = maxP - 1;
  } else if (activePly > 0) {
    activePly--;
  }

  renderSnapshotView(activePly);
  updateNavButtons();
  renderHistory();
  updatePlayerTags();
}

function stepForward() {
  if (!gameState || !gameState.snapshots || activePly === null) return;
  const maxP = gameState.snapshots.length - 1;

  if (activePly < maxP - 1) {
    activePly++;
    renderSnapshotView(activePly);
  } else {
    // Reached the live current move
    jumpToLivePosition();
    return;
  }

  updateNavButtons();
  renderHistory();
  updatePlayerTags();
}

function jumpToPly(ply) {
  if (!gameState || !gameState.snapshots) return;
  const maxP = gameState.snapshots.length - 1;
  if (ply < 0 || ply > maxP) return;

  if (ply === maxP) {
    jumpToLivePosition();
  } else {
    activePly = ply;
    renderSnapshotView(activePly);
    updateNavButtons();
    renderHistory();
    updatePlayerTags();
  }
}

function jumpToLivePosition() {
  activePly = null;
  renderBoard();
  renderEvalBar();
  renderLeaderCard();
  renderSoftComputing();
  renderLiveWeights();
  updateNavButtons();
  renderHistory();
  updatePlayerTags();
}

// Render historical snapshot position on board, eval bar, and weights
function renderSnapshotView(ply) {
  if (!gameState || !gameState.snapshots || !gameState.snapshots[ply]) return;
  const snap = gameState.snapshots[ply];

  selectedSquare = null;
  legalMovesFromSelected = [];

  renderBoardFromFen(snap.fen, snap.in_check);
  renderEvalBar(snap.eval);
  renderLeaderCard(snap.eval);
  renderSoftComputing(snap.eval);
  renderLiveWeights(snap.live_weights);
}

// Revert to historical ply and make a new move (Time Travel branching)
async function revertToPlyAndMove(ply, fromSq, toSq) {
  try {
    const revRes = await fetch('/api/revert_to_ply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ply })
    });
    const revData = await revRes.json();
    if (revData.success) {
      gameState = revData.state;
      activePly = null;
      // Now execute the desired move on the reverted state
      const targetMove = gameState.legal_moves.find(m => m.from === fromSq && m.to === toSq);
      if (targetMove) {
        executeMove(targetMove);
      } else {
        updateUI(gameState);
      }
    }
  } catch (err) {
    console.error('Error reverting to ply:', err);
    fetchState();
  }
}

// Setup Event Listeners
function setupEventListeners() {
  // Navigation buttons (< and >)
  const btnPrev = document.getElementById('btnPrevMove');
  if (btnPrev) btnPrev.addEventListener('click', stepBackward);

  const btnNext = document.getElementById('btnNextMove');
  if (btnNext) btnNext.addEventListener('click', stepForward);

  const btnJumpLive = document.getElementById('btnJumpLive');
  if (btnJumpLive) btnJumpLive.addEventListener('click', jumpToLivePosition);

  // Keyboard navigation
  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
      stepBackward();
    } else if (e.key === 'ArrowRight') {
      stepForward();
    }
  });

  // Action controls
  const btnNewGame = document.getElementById('btnNewGame');
  if (btnNewGame) {
    btnNewGame.addEventListener('click', () => {
      const modal = document.getElementById('settingsModal');
      if (modal) modal.style.display = 'flex';
    });
  }

  // Game Over Modal buttons
  const btnOverNewGame = document.getElementById('btnGameOverNewGame');
  if (btnOverNewGame) {
    btnOverNewGame.addEventListener('click', () => {
      isReviewingBoard = false;
      const overModal = document.getElementById('gameOverOverlay');
      if (overModal) overModal.style.display = 'none';
      const modal = document.getElementById('settingsModal');
      if (modal) modal.style.display = 'flex';
    });
  }

  const btnOverReview = document.getElementById('btnGameOverReview');
  if (btnOverReview) {
    btnOverReview.addEventListener('click', () => {
      isReviewingBoard = true;
      const overModal = document.getElementById('gameOverOverlay');
      if (overModal) overModal.style.display = 'none';
    });
  }

  // Settings modal close & start
  const btnStart = document.getElementById('btnStartNewGame');
  if (btnStart) btnStart.addEventListener('click', startCustomGame);

  const btnCancel = document.getElementById('btnCancelSettings');
  if (btnCancel) {
    btnCancel.addEventListener('click', () => {
      const modal = document.getElementById('settingsModal');
      if (modal) modal.style.display = 'none';
    });
  }
}

// Start Custom New Game
async function startCustomGame() {
  const color = document.getElementById('settingColor').value;
  const depth = parseInt(document.getElementById('settingDepth').value, 10);
  const weights = document.getElementById('settingWeights').value;

  boardFlipped = (color === 'black');
  isReviewingBoard = false;
  activePly = null;

  const settingsModal = document.getElementById('settingsModal');
  if (settingsModal) settingsModal.style.display = 'none';

  const overModal = document.getElementById('gameOverOverlay');
  if (overModal) overModal.style.display = 'none';

  const statsCard = document.getElementById('lastEngineStatsCard');
  if (statsCard) statsCard.style.display = 'none';

  try {
    const res = await fetch('/api/new_game', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: 'normal', color, depth, weights })
    });
    const data = await res.json();
    if (data.success) {
      updateUI(data.state);
    }
  } catch (err) {
    console.error('New game error:', err);
  }
}

// Promotion Modal
function openPromotionModal(isWhite) {
  const modal = document.getElementById('promotionModal');
  const pieces = isWhite ? ['Q', 'R', 'B', 'N'] : ['q', 'r', 'b', 'n'];

  document.getElementById('promoQ').innerHTML = getPieceSvg(pieces[0]);
  document.getElementById('promoR').innerHTML = getPieceSvg(pieces[1]);
  document.getElementById('promoB').innerHTML = getPieceSvg(pieces[2]);
  document.getElementById('promoN').innerHTML = getPieceSvg(pieces[3]);

  modal.style.display = 'flex';
}

function selectPromotion(pieceType) {
  document.getElementById('promotionModal').style.display = 'none';
  if (pendingPromotion) {
    sendMove(pendingPromotion.from, pendingPromotion.to, pieceType.toLowerCase());
    pendingPromotion = null;
  }
}
