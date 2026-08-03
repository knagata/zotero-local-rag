/* eslint-env browser */
/* globals graphology, Sigma, d3 */
/*
 * citation_graph/server.py (_build_sigma_html) から逐語的に切り出したもの。
 * THEME はこのファイルの直前に置かれるインライン <script> がサーバーから注入する。
 * このヘッダより下は、以前インライン化されていた内容とバイト単位で同一。
 */
(function () {
'use strict';

const THEME = window.__RAG_THEME__;
document.getElementById('loading').style.display = 'flex';

// 経過時間に応じてローディングメッセージを更新する
// キャッシュヒット時はレイアウト再計算が不要なのでシンプルなメッセージにする
var _loadingMsg = document.querySelector('#loading span');
var _loadingMsgsFull = [
  [0,   'グラフデータを構築中…'],
  [5,   'レイアウトを計算中…（初回やノード追加時は数分かかることがあります）'],
  [30,  'レイアウト計算中… しばらくお待ちください'],
  [90,  'もう少しです… FA2 レイアウト最適化中'],
];
var _loadingMsgsCache = [
  [0,   'グラフデータを読み込み中…'],
];
var _loadingMsgs = _loadingMsgsFull;  // キャッシュ状況が判明するまでフル版を使用
var _loadStart = Date.now();
var _loadTimer = setInterval(function() {
  var elapsed = (Date.now() - _loadStart) / 1000;
  var msg = _loadingMsgs[0][1];
  for (var i = 0; i < _loadingMsgs.length; i++) {
    if (elapsed >= _loadingMsgs[i][0]) msg = _loadingMsgs[i][1];
  }
  if (_loadingMsg) _loadingMsg.textContent = msg;
}, 1000);

// fetch('/api/graph') の.then/.catch双方から使うため、.then内ではなくここで
// 定義する。.then内に置くと.catchはその関数の外側なので参照できず、
// フェッチ失敗時のエラー表示自体が ReferenceError で失敗していた（2026-08-03）。
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

fetch('/api/graph')
  .then(function(r) { return r.json(); })
  .then(function(GRAPH_DATA) {
    clearInterval(_loadTimer);
    document.getElementById('loading').style.display = 'none';

    // ── ブラウザ閉じ検知（pagehide で close シグナルを送信）─────────────────
    // ページロード時に生存通知を送る（リフレッシュ時に close キャンセル）。
    fetch('/api/heartbeat', {method: 'POST'}).catch(function(){});
    // タブ・ウィンドウを閉じるときに sendBeacon で close シグナルを送信。
    // sendBeacon は pagehide で確実に届く（fetch/XHR は届かないことがある）。
    window.addEventListener('pagehide', function() {
      navigator.sendBeacon('/api/close');
    });

    // 凡例カウントをグラフデータから更新
    var _nAll   = GRAPH_DATA.nodes.length;
    var _nEdges = GRAPH_DATA.edges.length;
    var _nZ  = GRAPH_DATA.nodes.filter(function(n){return n.group==='zotero';}).length;
    var _nC  = GRAPH_DATA.nodes.filter(function(n){return n.group==='external';}).length;
    var _nR  = GRAPH_DATA.nodes.filter(function(n){return n.group==='reference';}).length;
    var _elStatNodes = document.getElementById('stat-nodes');
    var _elStatEdges = document.querySelector('#stat-nodes + *') || null;
    if (_elStatNodes) _elStatNodes.textContent = _nAll;
    var _elCiterVis = document.getElementById('stat-citer-vis');
    var _elRefVis   = document.getElementById('stat-ref-vis');
    if (_elCiterVis) { _elCiterVis.textContent = _nC; _elCiterVis.parentElement.lastChild.textContent = '('+_nC+')'; }
    if (_elRefVis)   { _elRefVis.textContent   = _nR; _elRefVis.parentElement.lastChild.textContent   = '('+_nR+')'; }

    // ── Semantic map state ─────────────────────────────────────────────────
    var _viewMode = 'citation';                     // 'citation' | 'semantic'
    var _semanticMethod = 'umap';                   // 選択中の次元削減手法
    var _nClusters = 8;                             // クラスタ数 (2-20)
    var _semanticPositions = {};                    // { node_id: [x, y] }
    var _citationNodePositions = {};                // 引用ネットワークのノード位置バックアップ
    var _semanticAbort = null;                      // 進行中の fetch をキャンセルするための AbortController
    var _semanticLayoutReady = false;
    var _semanticReqId = 0;                         // リクエスト識別子（古いレスポンスを破棄）
    var _clusterData = [];                          // クラスタ情報 [{id, label, item_keys, hull, color}, ...]
    var _selectedClusterId = -1;                    // -1 = 未選択
    var _clusterItemMap = {};                       // item_key → cluster_id

    // ── Save citation positions immediately (deep copy from graphology) ──
    // Wait for graph to be available, then save.  This runs after the graph is built.
    function _saveCitationPositionsNow() {
      _citationNodePositions = {};
      try {
        graph.forEachNode(function (id, attrs) {
          if (attrs.x != null) {
            _citationNodePositions[id] = [attrs.x, attrs.y];
          }
        });
        console.log('[RAG] saved citation positions: ' + Object.keys(_citationNodePositions).length + ' nodes');
      } catch(e) {
        // graph not ready yet — will retry on first use
        console.warn('[RAG] could not save citation positions (graph not ready):', e);
      }
    }

    function _loadSemanticLayout(method) {
      // 進行中のリクエストがあればキャンセル
      if (_semanticAbort) { _semanticAbort.abort(); }
      _semanticAbort = new AbortController();
      var signal = _semanticAbort.signal;
      var reqId = ++_semanticReqId;

      // badgePct の参照を壊さないよう、label テキストは data-label 属性で管理
      var badge = document.getElementById('layout-badge');
      var badgePctEl = document.getElementById('layout-pct');
      if (badge) {
        badge.style.display = 'block';
        badge.setAttribute('data-label', '意味マップ計算中 (' + method.toUpperCase() + ')');
        if (badgePctEl) badgePctEl.textContent = badge.getAttribute('data-label');
      }
      console.log('[RAG] loading semantic layout: method=' + method + ' n=' + _nClusters + ' reqId=' + reqId);
      fetch('/api/semantic-layout?method=' + encodeURIComponent(method) + '&n_clusters=' + _nClusters, { signal: signal })
        .then(function(r) { return r.json(); })
        .then(function(data) {
          // 古いリクエストの結果は無視
          if (reqId !== _semanticReqId) {
            console.log('[RAG] discarding stale response: reqId=' + reqId + ' current=' + _semanticReqId);
            return;
          }
          _semanticPositions = data.positions || {};
          _semanticLayoutReady = true;
          _semanticMethod = method;

          // クラスタデータ
          _clusterData = data.clusters || [];
          _clusterItemMap = {};
          _clusterData.forEach(function(c) {
            c.item_keys.forEach(function(ik) { _clusterItemMap[ik] = c.id; });
          });
          _buildClusterLegend();

          console.log('[RAG] semantic layout loaded: method=' + method +
                      ' totalPositions=' + Object.keys(_semanticPositions).length +
                      ' nItems=' + (data.n_items || 0) +
                      ' clusters=' + _clusterData.length);
          if (_viewMode === 'semantic') {
            _applySemanticPositions();
          }
          // Draw hulls AFTER positions are applied to graph nodes
          _drawClusterHulls();
          var _b = document.getElementById('layout-badge');
          if (_b) _b.style.display = 'none';
        })
        .catch(function(e) {
          if (e.name === 'AbortError') return;  // キャンセルは無視
          console.error('[RAG] semantic layout fetch failed:', e);
          var _b = document.getElementById('layout-badge');
          if (_b) _b.style.display = 'none';
        });
    }

    function _applySemanticPositions() {
      // Save citation positions on first switch (if not already saved)
      if (Object.keys(_citationNodePositions).length === 0) {
        _saveCitationPositionsNow();
      }

      // 物理シミュレーションを停止（動いたままだと座標が上書きされる）
      if (typeof simulation !== 'undefined' && simulation) {
        simulation.stop();
      }
      if (!layoutDone) {
        layoutDone = true;
        badge.style.display = 'none';
      }

      var pos = _semanticPositions;
      var count = 0;
      graph.forEachNode(function (id, attrs) {
        if (pos[id]) {
          graph.setNodeAttribute(id, 'x', pos[id][0]);
          graph.setNodeAttribute(id, 'y', pos[id][1]);
          count++;
        }
      });
      console.log('[RAG] applied semantic positions: ' + count + ' nodes updated');
      renderer.refresh();

      // ビューポートフィット
      if (selectedNode && typeof window._panToNode === 'function') {
        window._panToNode(selectedNode);
      } else if (typeof _fitView === 'function') {
        _fitView(true);
      }
    }

    function _restoreCitationPositions() {
      // Ensure we have saved positions
      if (Object.keys(_citationNodePositions).length === 0) {
        console.warn('[RAG] no citation positions saved, cannot restore');
        return;
      }
      var cpos = _citationNodePositions;
      var count = 0;
      graph.forEachNode(function (id, attrs) {
        if (cpos[id]) {
          graph.setNodeAttribute(id, 'x', cpos[id][0]);
          graph.setNodeAttribute(id, 'y', cpos[id][1]);
          count++;
        }
      });
      console.log('[RAG] restored citation positions: ' + count + ' nodes');
      renderer.refresh();

      // D3 シミュレーションを再開し、物理演算レイアウトを収束させる
      if (typeof simulation !== 'undefined' && simulation && typeof _addD3Nodes === 'function') {
        // バッジラベルを「再レイアウト中」に
        var _rb = document.getElementById('layout-badge');
        if (_rb) {
          _rb.setAttribute('data-label', '再レイアウト中');
          var _rp = document.getElementById('layout-pct');
          if (_rp) _rp.textContent = '再レイアウト中 0%';
        }
        // d3 state を全ノードで再構築（コレクションフィルタ後でも正しく動作）
        d3nodeById = {};
        d3nodes = [];
        activeNodeIds = new Set();
        graph.forEachNode(function (id) { _addD3Nodes([id]); });
        _rebuildLinks();
        simulation.nodes(d3nodes);
        simulation.force('link').links(d3links);
        layoutDone = false;
        simulation.alpha(0.3);
        var badge = document.getElementById('layout-badge');
        if (badge) badge.style.display = 'block';
        requestAnimationFrame(layoutStep);
      } else {
        if (selectedNode && typeof window._panToNode === 'function') {
          window._panToNode(selectedNode);
        } else if (typeof _fitView === 'function') {
          _fitView(true);
        }
      }
    }

    function switchViewMode(mode) {
      if (mode === _viewMode) return;
      _viewMode = mode;

      document.querySelectorAll('.view-tab').forEach(function(t) {
        t.classList.toggle('active', t.dataset.view === mode);
      });

      var methodWrap = document.getElementById('semantic-method-wrap');
      var clustersWrap = document.getElementById('semantic-clusters-wrap');
      var clusterLegend = document.getElementById('cluster-legend');
      if (mode === 'semantic') {
        // 物理シミュレーションを即座に停止（非同期ロード中も動き続けるのを防ぐ）
        if (typeof simulation !== 'undefined' && simulation) {
          simulation.stop();
        }
        if (!layoutDone) {
          layoutDone = true;
          badge.style.display = 'none';
        }

        // 意味マップでは外部論文が非表示のため、サイドバーもZoteroのみに強制
        var _zoEl = document.getElementById('sb-zotero-only');
        var _zlEl = document.getElementById('sb-zotero-only-label');
        if (_zoEl) {
          if (!('_savedZoteroOnly' in window)) window._savedZoteroOnly = _zoEl.checked;
          _zoEl.checked = true;
          _zoEl.disabled = true;
          if (_zlEl) _zlEl.style.opacity = '0.45';
        }
        if (typeof window._renderList === 'function') window._renderList();

        if (methodWrap) methodWrap.classList.add('show');
        if (clustersWrap) clustersWrap.classList.add('show');
        if (clusterLegend && _clusterData.length > 0) clusterLegend.classList.add('show');
        _drawClusterHulls();
        if (_semanticLayoutReady) {
          _applySemanticPositions();
        } else {
          _loadSemanticLayout(_semanticMethod);
        }
      } else {
        // 引用ネットワークに戻すときはZotero-onlyフィルタを元に戻す
        var _czEl = document.getElementById('sb-zotero-only');
        var _czLbl = document.getElementById('sb-zotero-only-label');
        if (_czEl) {
          _czEl.disabled = false;
          _czEl.checked = window._savedZoteroOnly || false;
          if (_czLbl) _czLbl.style.opacity = '';
        }
        if (typeof window._renderList === 'function') window._renderList();

        if (methodWrap) methodWrap.classList.remove('show');
        if (clustersWrap) clustersWrap.classList.remove('show');
        if (clusterLegend) clusterLegend.classList.remove('show');
        // Clear overlay when switching to citation mode
        var canvas = document.getElementById('cluster-overlay');
        if (canvas) {
          var ggl = canvas.__gl;
          if (ggl) { ggl.clear(ggl.COLOR_BUFFER_BIT); }
          else { var cc = canvas.getContext('2d'); if (cc) cc.clearRect(0, 0, canvas.width, canvas.height); }
        }
        _selectedClusterId = -1;
        if (window._colItemKeys !== null) {
          // クラスタフィルタを解除（コレクションフィルタはそのまま）
          window._colItemKeys = null;
          if (typeof _applyColFilter === 'function') _applyColFilter();
        }
        _restoreCitationPositions();
      }
    }

    // ── Plain Voronoi overlay ──────────────────────────────────────────
    // Each cell is flat-filled with its cluster colour (no stroke).

    function _drawClusterHulls() {
      if (!_clusterData.length) return;

      var canvas = document.getElementById('cluster-overlay');
      if (!canvas) return;
      var dim = renderer.getDimensions();
      if (dim.width <= 0 || dim.height <= 0) return;
      canvas.width = dim.width; canvas.height = dim.height;
      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, dim.width, dim.height);

      // Collect screen-space points with cluster colour
      var pts = [];
      _clusterData.forEach(function(c) {
        if (!c.item_keys) return;
        c.item_keys.forEach(function(ik) {
          var pos = _semanticPositions['item:' + ik];
          if (pos) {
            var vp = renderer.graphToViewport({ x: pos[0], y: pos[1] });
            pts.push({ x: vp.x, y: vp.y, color: c.color });
          }
        });
      });
      if (pts.length < 3) return;

      // Delaunay → Voronoi
      var delaunay = d3.Delaunay.from(pts, function(d) { return d.x; }, function(d) { return d.y; });
      var voronoi = delaunay.voronoi([0, 0, dim.width, dim.height]);

      ctx.globalAlpha = 0.35;
      for (var i = 0; i < pts.length; i++) {
        var poly = voronoi.cellPolygon(i);
        if (!poly || poly.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(poly[0][0], poly[0][1]);
        for (var j = 1; j < poly.length; j++) ctx.lineTo(poly[j][0], poly[j][1]);
        ctx.closePath();
        ctx.fillStyle = pts[i].color;
        ctx.fill();
      }
      ctx.globalAlpha = 1.0;
    }

    // ── Cluster legend builder ─────────────────────────────────────────────
    function _buildClusterLegend() {
      var wrap = document.getElementById('cluster-legend');
      var list = document.getElementById('cluster-legend-list');
      if (!wrap || !list) return;
      if (_viewMode === 'semantic' && _clusterData.length > 0) {
        wrap.classList.add('show');
      } else {
        wrap.classList.remove('show');
        return;
      }
      list.innerHTML = '';
      _clusterData.forEach(function(c) {
        var kw = c.keywords && c.keywords.length ? c.keywords.join(', ') : '';

        var row = document.createElement('div');
        row.className = 'cl-row' + (c.id === _selectedClusterId ? ' active' : '');
        var dot = document.createElement('span');
        dot.className = 'cl-dot';
        dot.style.background = c.color;
        var label = document.createElement('span');
        label.className = 'cl-label';
        label.textContent = c.label;
        var count = document.createElement('span');
        count.className = 'cl-count';
        count.textContent = String(c.item_keys.length);
        row.append(dot, label, count);
        if (kw) {
          var keywords = document.createElement('span');
          keywords.className = 'cl-keywords';
          keywords.textContent = kw;
          row.appendChild(keywords);
        }
        row.addEventListener('click', function() {
          _selectCluster(c.id === _selectedClusterId ? -1 : c.id);
        });
        list.appendChild(row);
      });
    }

    function _selectCluster(clusterId) {
      _selectedClusterId = clusterId;
      _buildClusterLegend();
      _drawClusterHulls();

      if (clusterId >= 0) {
        var cluster = _clusterData.find(function(c) { return c.id === clusterId; });
        if (cluster && cluster.item_keys) {
          window._colItemKeys = new Set(cluster.item_keys);
        }
      } else {
        window._colItemKeys = null;
      }
      renderer.refresh();
      if (typeof window._renderList === 'function') window._renderList();
    }

    // ── Cluster click detection (via sigma clickStage) ──────────────────────
    // Canvas overlay は pointer-events: none のまま。パン/ズームは sigma が処理する。
    var _lastCanvasClick = { x: 0, y: 0 };
    document.getElementById('sigma-container').addEventListener('click', function(ev) {
      _lastCanvasClick.x = ev.offsetX;
      _lastCanvasClick.y = ev.offsetY;
    });

    // Camera-follow handler registered below, after sigma initialization.

    function _pointInPolygon(x, y, polygon) {
      var inside = false;
      for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        var xi = polygon[i][0], yi = polygon[i][1];
        var xj = polygon[j][0], yj = polygon[j][1];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
          inside = !inside;
        }
      }
      return inside;
    }

    // View tab click handlers
    document.querySelectorAll('.view-tab').forEach(function(tab) {
      tab.addEventListener('click', function() {
        switchViewMode(tab.dataset.view);
      });
    });

    // Semantic method dropdown handler
    var _semanticMethodSelect = document.getElementById('semantic-method-select');
    if (_semanticMethodSelect) {
      _semanticMethodSelect.addEventListener('change', function() {
        _semanticMethod = this.value;
        _semanticLayoutReady = false;  // キャッシュがなければ再計算
        _loadSemanticLayout(_semanticMethod);
      });
    }

    // ── Cluster count slider ──
    var _nClustersSlider = document.getElementById('nclusters-slider');
    var _nClustersLabel = document.getElementById('nclusters-label');
    var _nClustersTimer = null;
    if (_nClustersSlider && _nClustersLabel) {
      _nClustersSlider.addEventListener('input', function() {
        _nClusters = parseInt(this.value);
        _nClustersLabel.textContent = _nClusters;
        // Debounce: wait 400ms after last change before re-fetching
        if (_nClustersTimer) clearTimeout(_nClustersTimer);
        _nClustersTimer = setTimeout(function() {
          if (_viewMode === 'semantic') {
            // Re-fetch with new n_clusters — layout cache still applies,
            // only clustering is recomputed server-side
            _loadSemanticLayout(_semanticMethod);
          }
        }, 400);
      });
    }

/* ── 0. Layout geometry – computed first so sections 2 & 3 can use it.
        bboxR:    half-size of the fixed normalisation bounding box.
                  Must contain the D3 equilibrium spread (empirically ≈ 70√N).
                  Set bigger than expected spread to avoid clipping outliers.
        initMaxR: outer radius of the phyllotaxis seed spiral (≈ 65 % of bbox).
                  Nodes start spread out so the animation is visible from frame 1.
*/
var _N0   = GRAPH_DATA.nodes.length;
// bboxR is larger now to accommodate the collision-based cluster radii.
// With forceCollide, clusters expand until nodes physically don't overlap,
// which for 15 citers around a hub requires cluster radius ≈ 300-500 graph units.
var bboxR = _N0 < 50   ? 1800 :
            _N0 < 150  ? 3000 :
            _N0 < 400  ? 4500 :
            _N0 < 1000 ? 7500 : 12000;
var initMaxR = bboxR * 0.60;   // initial spread = 60 % of bbox → visible in viewport

/* ── 1. Build graphology graph ─────────────────────────────── */
const graph = new graphology.DirectedGraph();
GRAPH_DATA.nodes.forEach(function (n) { graph.addNode(n.id, n); });
GRAPH_DATA.edges.forEach(function (e) {
  try { graph.addEdgeWithKey(e.id, e.source, e.target, e); } catch (_) {}
});

/* ── 2. Initial positions ────────────────────────────────────
   If the Python backend embedded pre-computed FA2 positions (x/y on each node),
   use them directly — no browser-side physics needed.
   Otherwise fall back to phyllotaxis seeding for the D3 simulation. */
var _hasPrecomputedLayout = GRAPH_DATA.nodes.length > 0 &&
    GRAPH_DATA.nodes[0].x != null && GRAPH_DATA.nodes[0].x !== undefined;

if (_hasPrecomputedLayout) {
  // Positions already embedded — just copy them into graphology attributes.
  // bboxR is set from actual coordinate spread so sigma's normalisation fits.
  var _allX = [], _allY = [];
  graph.forEachNode(function (id, attrs) {
    if (attrs.x != null) { _allX.push(attrs.x); _allY.push(attrs.y); }
  });
  if (_allX.length) {
    var _span = Math.max(
      Math.max.apply(null, _allX) - Math.min.apply(null, _allX),
      Math.max.apply(null, _allY) - Math.min.apply(null, _allY)
    );
    bboxR = Math.ceil(_span * 0.65);  // bbox slightly larger than spread
  }
  console.log('[RAG] using pre-computed layout, bboxR=' + bboxR);
} else {
  // Fallback: phyllotaxis for Zotero items, anchored spread for external nodes.
  var _zoteroIds = [], _externalIds = [];
  graph.forEachNode(function (id, attrs) {
    if (attrs.group === 'zotero') _zoteroIds.push(id);
    else _externalIds.push(id);
  });
  var _NZ = _zoteroIds.length || 1;
  var _zoteroPos = {};
  _zoteroIds.forEach(function (id, i) {
    var r = initMaxR * (0.25 + 0.55 * i / Math.max(1, _NZ - 1));
    var theta = i * 2.39996;
    var x = r * Math.cos(theta), y = r * Math.sin(theta);
    graph.setNodeAttribute(id, 'x', x);
    graph.setNodeAttribute(id, 'y', y);
    _zoteroPos[id] = { x: x, y: y };
  });
  var _anchor = {};
  graph.forEachEdge(function (e, attrs, src, tgt) {
    if (!_anchor[src] && _zoteroPos[tgt]) _anchor[src] = tgt;
    if (!_anchor[tgt] && _zoteroPos[src]) _anchor[tgt] = src;
  });
  var _spreadR = initMaxR * 0.18;
  _externalIds.forEach(function (id) {
    var anch = _anchor[id];
    var base = anch ? _zoteroPos[anch] : { x: 0, y: 0 };
    var ang = Math.random() * 2 * Math.PI;
    var d = _spreadR * (0.4 + Math.random() * 0.8);
    graph.setNodeAttribute(id, 'x', base.x + d * Math.cos(ang));
    graph.setNodeAttribute(id, 'y', base.y + d * Math.sin(ang));
  });
  console.log('[RAG] using fallback phyllotaxis layout');
}

// Save initial FA2/phyllotaxis positions for citation-network restore
_saveCitationPositionsNow();

/* ── 3. Create sigma renderer ───────────────────────────── */
const SigmaClass = (typeof Sigma === 'function') ? Sigma
                 : (Sigma && typeof Sigma.Sigma === 'function') ? Sigma.Sigma
                 : null;
if (!SigmaClass) { console.error('[RAG] sigma.js not loaded'); return; }

// ── コレクションフィルタ状態（nodeReducer から参照）────────────────────────
window._colItemKeys = null;  // null = フィルタなし、Set = 表示対象 item_key

// ── エッジホバー状態 ─────────────────────────────────────────────────────────
var hoveredEdge = null;

// ── 翻訳トグル状態（エッジ切替後も保持）──────────────────────────────────────
var _translateToggle = false;

// ── CC filter thresholds (updated by legend inputs) ──────────────────────────
var filterCiterCC = 50;  // min citation count for citer nodes (group='external')
var filterRefCC   = 50;  // min citation count for ref nodes   (group='reference')

// Click-selection state – declared here so nodeReducer / edgeReducer can close
// over them.  Hovering only shows a label pill; clicking toggles the focus
// (selected node + 1st-degree neighbours highlighted, everything else dimmed).
var selectedNode           = null;
var selectedNeighbors      = new Set();  // all direct neighbours of selectedNode
var selectedChunkNodes     = new Set();  // chunk-type direct neighbours
var selectedChunkNeighbors = new Set();  // nodes connected through those chunks
// 選択ノードからみた関係方向（選択時のハイライト色分け用）
var selectedCiterNodes     = new Set();  // selected ← other (other が選択ノードを引用)
var selectedRefNodes       = new Set();  // selected → other (選択ノードが other を参照)
var _activeEdge            = null;       // クリックされたエッジ（コンテキスト表示中は強調維持）

// ── 選択トランジション ───────────────────────────────────────────────────────
// _selectionT: 0 = 選択直後（未適用）→ 1 = 完全適用
// nodeReducer でこの値を使い、非選択ノードの色を背景色へ補間する。
var _selectionT    = 0;
var _selectionAnimId = null;
var _SELECTION_DUR = 380;  // ms
var _BG_COLOR      = THEME.surface || '#141218';

// 解除アニメーション中に直前の選択状態を保持する変数
var _prevSelectedNode    = null;
var _prevSelectedNeighbors   = new Set();
var _prevCiterNodes      = new Set();
var _prevRefNodes        = new Set();

// ── 起動フェードイン ─────────────────────────────────────────────────────────
// _fadeInT: 0 → 1 (500ms) で nodeReducer/edgeReducer の色を背景色から補間する。
// renderer 作成後に _fadeInRefresh に renderer.refresh を差し込む。
var _fadeInT = 0;
var _fadeInRefresh = null;
setTimeout(function() {
  var _FADE_DUR = 380;
  var _start = Date.now();
  function _step() {
    var raw = Math.min(1, (Date.now() - _start) / _FADE_DUR);
    _fadeInT = raw * raw * (3 - 2 * raw); // smoothstep
    if (_fadeInRefresh) _fadeInRefresh();
    if (raw < 1) { requestAnimationFrame(_step); }
    else { _fadeInT = 1; }
  }
  requestAnimationFrame(_step);
}, 500);

function _hexLerp(a, b, t) {
  // '#rrggbb' 同士を補間
  var ar = parseInt(a.slice(1,3),16), ag = parseInt(a.slice(3,5),16), ab = parseInt(a.slice(5,7),16);
  var br = parseInt(b.slice(1,3),16), bg = parseInt(b.slice(3,5),16), bb = parseInt(b.slice(5,7),16);
  var r = (ar + (br-ar)*t|0).toString(16).padStart(2,'0');
  var g = (ag + (bg-ag)*t|0).toString(16).padStart(2,'0');
  var bl = (ab + (bb-ab)*t|0).toString(16).padStart(2,'0');
  return '#' + r + g + bl;
}

function _startSelectionAnim() {
  if (_selectionAnimId) cancelAnimationFrame(_selectionAnimId);
  _selectionT = 0;
  var start = Date.now();
  function step() {
    var raw = Math.min(1, (Date.now() - start) / _SELECTION_DUR);
    // quadraticOut easing
    _selectionT = 1 - (1 - raw) * (1 - raw);
    renderer.refresh();
    if (raw < 1) { _selectionAnimId = requestAnimationFrame(step); }
    else { _selectionAnimId = null; }
  }
  _selectionAnimId = requestAnimationFrame(step);
}

function recomputeSelection(node) {
  selectedNode           = node;
  selectedNeighbors      = new Set(graph.neighbors(node));
  selectedChunkNodes     = new Set();
  selectedChunkNeighbors = new Set();
  selectedCiterNodes     = new Set();
  selectedRefNodes       = new Set();

  // 辺の向きからciter/refを分類:
  //   other → selected : other は selected を引用している = citer
  //   selected → other : selected は other を参照している = ref
  graph.edges(node).forEach(function(edge) {
    var src = graph.source(edge), tgt = graph.target(edge);
    if (tgt === node) {
      selectedCiterNodes.add(src);
    } else {
      selectedRefNodes.add(tgt);
    }
  });

  selectedNeighbors.forEach(function(n) {
    try {
      if (graph.getNodeAttribute(n, 'group') === 'chunk') {
        selectedChunkNodes.add(n);
        graph.neighbors(n).forEach(function(nn) {
          if (nn !== node) selectedChunkNeighbors.add(nn);
        });
      }
    } catch(_) {}
  });
  _startSelectionAnim();
}

function clearSelection() {
  if (_selectionAnimId) { cancelAnimationFrame(_selectionAnimId); _selectionAnimId = null; }
  // _selectionT を 1→0 方向に戻しながら選択状態を解除する。
  // nodeReducer は selectedNode===null なら dimming を一切行わないので、
  // _selectionT を 0 へ向けて下げながら refresh すると非選択ノードが徐々に復帰する。
  var startT = _selectionT;
  var start  = Date.now();
  // 直前の選択状態を保存（解除アニメーション中に参照）
  _prevSelectedNode      = selectedNode;
  _prevSelectedNeighbors = new Set(selectedNeighbors);
  _prevCiterNodes        = new Set(selectedCiterNodes);
  _prevRefNodes          = new Set(selectedRefNodes);
  selectedNode           = null;
  selectedNeighbors      = new Set();
  selectedChunkNodes     = new Set();
  selectedChunkNeighbors = new Set();
  selectedCiterNodes     = new Set();
  selectedRefNodes       = new Set();
  // すでに完全解除済みなら即終了
  if (startT <= 0) { _selectionT = 0; _prevSelectedNode = null; return; }
  function step() {
    var raw = Math.min(1, (Date.now() - start) / (_SELECTION_DUR * 2));
    var eased = 1 - (1 - raw) * (1 - raw);
    _selectionT = startT * (1 - eased);
    renderer.refresh();
    if (raw < 1) { _selectionAnimId = requestAnimationFrame(step); }
    else { _selectionT = 0; _selectionAnimId = null; _prevSelectedNode = null; }
  }
  _selectionAnimId = requestAnimationFrame(step);
}

// ── Custom hover renderer ─────────────────────────────────────────────────────
// Subtle ring + white pill label on hover.  No dimming here – that is done on
// click via nodeReducer / edgeReducer.
function customDrawHover(context, data, settings) {
  var size   = settings.labelSize   || 12;
  var font   = settings.labelFont   || 'sans-serif';
  var weight = settings.labelWeight || '500';
  var nr     = data.size;

  // Soft halo
  context.beginPath();
  context.arc(data.x, data.y, nr + 9, 0, Math.PI * 2);
  context.closePath();
  context.fillStyle = 'rgba(255,255,255,0.07)';
  context.fill();

  // White border ring
  context.beginPath();
  context.arc(data.x, data.y, nr + 3, 0, Math.PI * 2);
  context.closePath();
  context.fillStyle = 'rgba(255,255,255,0.85)';
  context.fill();

  // Node colour fill (use original colour from graph attributes, not reducer)
  var origColor = graph.getNodeAttribute(data.key || '', 'color') || data.color;
  context.beginPath();
  context.arc(data.x, data.y, nr + 1, 0, Math.PI * 2);
  context.closePath();
  context.fillStyle = origColor;
  context.fill();

  if (!data.label) return;

  context.font = weight + ' ' + size + 'px ' + font;
  var tw  = context.measureText(data.label).width;
  var pad = 6, br = 6;
  var bx  = data.x + nr + 8;
  var by  = data.y - size / 2 - pad;
  var bw  = tw + pad * 2;
  var bh  = size + pad * 2;

  // White pill with shadow
  context.shadowOffsetX = 0; context.shadowOffsetY = 2;
  context.shadowBlur    = 10; context.shadowColor = 'rgba(0,0,0,0.4)';
  context.fillStyle = '#ffffff';
  context.beginPath();
  if (context.roundRect) { context.roundRect(bx, by, bw, bh, br); }
  else { context.rect(bx, by, bw, bh); }
  context.fill();
  context.shadowBlur = 0;

  context.fillStyle = '#0f172a';
  context.fillText(data.label, bx + pad, data.y + size / 3);
}

// ── nodeReducer: dim non-neighbours when a node is selected ──────────────────
//   • selected node                  → full colour, highlighted
//   • direct neighbour or chunk's neighbour → full colour, label visible
//   • everything else                → very dark (fades into bg), label hidden
// Also applies zoom-adaptive sizing: nodes scale up when zoomed out so the
// graph doesn't collapse into an undifferentiated blob at low zoom levels.
// Group base z-index: Zotero items always render in front of external nodes (sigma v3 zIndex).
var _Z_BASE = { zotero: 20, chunk: 20, external: 10, reference: 10 };

function nodeReducer(node, data) {
  var res = Object.assign({}, data);

  // ── Base z-index by group ─────────────────────────────────────────────────
  res.zIndex = _Z_BASE[data.group] || 10;

  // ── Collection filter ────────────────────────────────────────────────────
  if (window._colItemKeys !== null) {
    if (data.group === 'zotero' || data.group === 'chunk') {
      // Zotero アイテム：コレクション外なら非表示
      // itemKey フィールド優先、なければノードID "item:XXXXX" から抽出
      var nkey = data.itemKey || (node.indexOf(':') >= 0 ? node.split(':')[1] : node);
      if (!window._colItemKeys.has(nkey)) {
        res.hidden = true;
        return res;
      }
    } else {
      // 外部論文：フィルタ後の Zotero ノードに接続していなければ非表示
      var hasVisibleNeighbor = false;
      try {
        graph.forEachNeighbor(node, function(nid, ndata) {
          if (ndata.group === 'zotero' || ndata.group === 'chunk') {
            var nk = ndata.itemKey || nid;
            if (window._colItemKeys.has(nk)) { hasVisibleNeighbor = true; }
          }
        });
      } catch(_) {}
      if (!hasVisibleNeighbor) {
        res.hidden = true;
        return res;
      }
    }
  }

  // ── CC threshold filter ───────────────────────────────────────────────────
  if (data.group === 'external' && filterCiterCC > 0 && (data.cc || 0) < filterCiterCC) {
    res.hidden = true;
    return res;
  }
  if (data.group === 'reference' && filterRefCC > 0 && (data.cc || 0) < filterRefCC) {
    res.hidden = true;
    return res;
  }

  // ── Semantic map mode: hide external/reference nodes, color by cluster ────
  if (_viewMode === 'semantic') {
    if (data.group === 'external' || data.group === 'reference') {
      res.hidden = true;
      return res;
    }
    // In semantic mode, use same colors as citation mode
    if (data.group === 'zotero') {
      res.color = THEME.nodeZotero;
    }
  }

  // ── Zoom-adaptive size ────────────────────────────────────────────────────
  // sigma node sizes are fixed screen-pixels, unaffected by camera zoom.
  // When zoomed OUT (ratio > 1): shrink nodes aggressively to prevent blob.
  // No enlargement when zoomed in — that causes a sudden-size bug on pan.
  // camera may be null on the very first call during renderer construction.
  if (camera) {
    var ratio = camera.getState().ratio;
    if (ratio > 1) {
      // Zoomed out: shrink. ratio=2→×0.57, ratio=4→×0.33, ratio=8→×0.19
      res.size = Math.max(0.5, data.size / Math.pow(ratio, 0.8));
    }
    // Hide labels for external/reference nodes when zoomed out — only show Zotero
    // item labels (group='zotero') so the screen doesn't fill with tiny text.
    if (ratio > 2 && data.group !== 'zotero' && data.group !== 'chunk') {
      res.label = '';
    }
  }

  // ── Selection dimming + citer/ref color highlight ────────────────────────
  // 選択時: 選択ノード自身は highlighted、隣接ノードは辺の向きで色分け。
  // Inactive nodes are hidden (not just dimmed) so they cannot cover active
  // nodes regardless of WebGL draw order.
  // _selectionT > 0 なら選択中 or 解除トランジション中
  if (selectedNode !== null || _selectionT > 0) {
    if (selectedNode !== null) {
      // ── 選択中 ──────────────────────────────────────────────
      if (node === selectedNode) {
        res.highlighted = true;
        res.zIndex = (res.zIndex || 10) + 10;
      } else if (selectedNeighbors.has(node) || selectedChunkNeighbors.has(node)) {
        res.zIndex = (res.zIndex || 10) + 5;
        var grp = data.group;
        if (grp === 'external' || grp === 'reference') {
          if (selectedCiterNodes.has(node)) res.color = THEME.nodeCiter;
          else if (selectedRefNodes.has(node)) res.color = THEME.nodeRef;
        }
      } else {
        if (_selectionT >= 1) { res.hidden = true; }
        else {
          res.color = _hexLerp(data.color || res.color, _BG_COLOR, _selectionT);
          res.size  = res.size * (1 - _selectionT * 0.8);
        }
        res.label = '';
      }
    } else {
      // ── 解除アニメーション中（selectedNode === null, _selectionT > 0）──
      // 直前の選択ノード・隣接は表示したまま色だけ元に戻す
      if (node === _prevSelectedNode) {
        res.zIndex = (res.zIndex || 10) + 10;
        // highlighted色 → 通常色へ補間（_selectionT: 1→0）
      } else if (_prevSelectedNeighbors.has(node)) {
        res.zIndex = (res.zIndex || 10) + 5;
        var grp2 = data.group;
        if (grp2 === 'external' || grp2 === 'reference') {
          var fromColor = _prevCiterNodes.has(node) ? THEME.nodeCiter
                        : _prevRefNodes.has(node)   ? THEME.nodeRef
                        : data.color;
          res.color = _hexLerp(data.color || res.color, fromColor, _selectionT);
        }
      } else {
        // 非選択ノード: 背景色から通常色へフェードイン
        res.color = _hexLerp(data.color || res.color, _BG_COLOR, _selectionT);
        res.size  = res.size * (1 - _selectionT * 0.8);
        res.label = '';
      }
    }
  }
  // ── 起動フェードイン ────────────────────────────────────────────────────────
  if (_fadeInT < 1 && !res.hidden) {
    res.color = _hexLerp(_BG_COLOR, res.color || data.color, _fadeInT);
    res.size  = (res.size || data.size || 4) * _fadeInT;
    res.label = '';
  }
  return res;
}

// ── edgeReducer: show edges touching selected node OR any of its chunk nodes ──
// Chunk nodes are expanded children: their edges to citers/refs must be visible.
function edgeReducer(edge, data) {
  var res = Object.assign({}, data);
  // Hide edges to/from filtered-out nodes
  var src = graph.source(edge), tgt = graph.target(edge);
  try {
    var sa = graph.getNodeAttributes(src), ta = graph.getNodeAttributes(tgt);
    if ((sa.group === 'external'  && filterCiterCC > 0 && (sa.cc || 0) < filterCiterCC) ||
        (ta.group === 'external'  && filterCiterCC > 0 && (ta.cc || 0) < filterCiterCC) ||
        (sa.group === 'reference' && filterRefCC   > 0 && (sa.cc || 0) < filterRefCC)   ||
        (ta.group === 'reference' && filterRefCC   > 0 && (ta.cc || 0) < filterRefCC)) {
      res.hidden = true;
      return res;
    }
  } catch(_) {}

  // ── Semantic map mode: hide edges involving external nodes ────────────────
  if (_viewMode === 'semantic') {
    try {
      var _sga = graph.getNodeAttributes(src), _tga = graph.getNodeAttributes(tgt);
      if (_sga.group === 'external' || _sga.group === 'reference' ||
          _tga.group === 'external' || _tga.group === 'reference') {
        res.hidden = true;
        return res;
      }
    } catch(_) {}
  }

  if (selectedNode !== null) {
    // ── 選択中 ──────────────────────────────────────────────────────────────
    if (src === selectedNode || tgt === selectedNode ||
        selectedChunkNodes.has(src) || selectedChunkNodes.has(tgt)) {
      res.zIndex = 1;
      if (tgt === selectedNode || selectedChunkNodes.has(tgt)) {
        res.color = THEME.edgeCitation;
      } else if (src === selectedNode || selectedChunkNodes.has(src)) {
        res.color = THEME.edgeReference;
      }
      // アクティブエッジ（コンテキスト表示中）はシアンで太く強調
      if (edge === _activeEdge) {
        res.color  = '#00E5FF';
        res.size   = (data.size || 1) * 3.0;
        res.zIndex = 3;
      }
      // ホバー中エッジを白+太く強調（アクティブより優先）
      if (edge === hoveredEdge) {
        res.color  = '#ffffff';
        res.size   = (data.size || 1) * 3.5;
        res.zIndex = 4;
      }
    } else {
      res.hidden = true;
    }
  } else if (_prevSelectedNode !== null && _selectionT > 0) {
    // ── 解除アニメーション中: 直前の選択エッジは表示したまま色を戻す ──────
    if (src === _prevSelectedNode || tgt === _prevSelectedNode) {
      res.zIndex = 1;
      var fromColor = (tgt === _prevSelectedNode) ? THEME.edgeCitation
                    : (src === _prevSelectedNode)  ? THEME.edgeReference
                    : data.color;
      res.color = _hexLerp(data.color || res.color, fromColor, _selectionT);
    } else {
      res.hidden = true;
    }
  }
  // ── 起動フェードイン ────────────────────────────────────────────────────────
  if (_fadeInT < 1 && !res.hidden) {
    res.color = _hexLerp(_BG_COLOR, res.color || data.color, _fadeInT);
  }
  return res;
}

const container = document.getElementById('sigma-container');
const renderer  = new SigmaClass(graph, container, {
  zIndex:                true,
  renderEdgeLabels:      false,
  defaultEdgeType:       'arrow',
  labelFont:             'Inter, system-ui, sans-serif',
  labelSize:             12,
  labelWeight:           '500',
  labelThreshold:        7,
  labelColor:            { color: THEME.onSurfaceVariant },
  defaultNodeColor:      THEME.nodeZotero,
  defaultEdgeColor:      THEME.edgeDefault,
  minCameraRatio:        0.01,
  maxCameraRatio:        20,
  // sigma 3.x ではエッジイベント検出は単一の enableEdgeEvents で制御する。
  // v2 の enableEdgeClickEvents / enableEdgeHoverEvents は設定キーとして残るが
  // 当たり判定を有効化しないため、これを true にしないとエッジをクリックできない。
  enableEdgeEvents: true,
  defaultDrawNodeHover:  customDrawHover,
  nodeReducer:           nodeReducer,
  edgeReducer:           edgeReducer,
});
var camera = renderer.getCamera();

// ── Camera-follow for Voronoi overlay ──
var _hullRedrawTimer = null;
camera.on('updated', function() {
  if (_viewMode !== 'semantic' || !_clusterData.length) return;
  if (_hullRedrawTimer) clearTimeout(_hullRedrawTimer);
  _hullRedrawTimer = setTimeout(_drawClusterHulls, 50);
});

// Fix sigma's normalisation to the pre-computed bbox so D3 movements
// produce stable framed-coordinate changes (visible animation).
// bboxR is computed at the top based on graph size.
renderer.setCustomBBox({ x: [-bboxR, bboxR], y: [-bboxR, bboxR] });
// フェードインループに refresh を接続
_fadeInRefresh = function() { renderer.refresh(); };
// debug hooks
window.__g = graph; window.__r = renderer; window.__bboxR = bboxR;
window.__d3nodes = d3nodes; window.__d3nById = d3nodeById;
// window.__sim は simulation 宣言後に設定（下方で代入）


/* ── 4. Tooltip (hover) + click-selection ───────────────── */
// selectedNode / selectedNeighbors are declared before the sigma constructor.
var mouseX = 0, mouseY = 0;
var tooltip = document.getElementById('rag-tooltip');

container.addEventListener('mousemove', function (e) {
  mouseX = e.clientX; mouseY = e.clientY;
  if (tooltip.style.display !== 'none') {
    var tw = tooltip.offsetWidth || 360, th = tooltip.offsetHeight || 80;
    tooltip.style.left = Math.min(mouseX + 15, window.innerWidth  - tw - 8) + 'px';
    tooltip.style.top  = Math.min(mouseY + 10, window.innerHeight - th - 8) + 'px';
  }
});

// Hover: show tooltip only (no dimming)
renderer.on('enterNode', function (ev) {
  var tip = (graph.getNodeAttributes(ev.node).tooltip ||
             graph.getNodeAttribute(ev.node, 'label') || '');
  if (tip) {
    tooltip.textContent = tip;
    tooltip.style.display = 'block';
    tooltip.style.left = (mouseX + 15) + 'px';
    tooltip.style.top  = (mouseY + 10) + 'px';
  }
});
renderer.on('leaveNode', function () {
  tooltip.style.display = 'none';
});

// Edge hover – only highlight when a node is selected (avoids misclicks in dense areas)
renderer.on('enterEdge', function (ev) {
  if (selectedNode === null) return;
  hoveredEdge = ev.edge;
  renderer.refresh();
});
renderer.on('leaveEdge', function () {
  if (hoveredEdge === null) return;
  hoveredEdge = null;
  renderer.refresh();
});

// Edge click – fetch citation contexts and show in context pane
renderer.on('clickEdge', function (ev) {
  if (selectedNode === null) return;
  var edge = ev.edge;
  _activeEdge = edge;
  renderer.refresh();
  var src  = graph.source(edge), tgt = graph.target(edge);
  var srcA = graph.getNodeAttributes(src);
  var tgtA = graph.getNodeAttributes(tgt);
  var edgeA = graph.getEdgeAttributes(edge);
  var myReq = ++_ctxPaneReq;
  var reportButton = edgeA.relationKey
    ? '<button id="relation-report-btn" style="margin-left:auto;background:none;' +
      'border:1px solid var(--outline-variant);border-radius:4px;color:var(--on-surface-variant);' +
      'padding:3px 7px;cursor:pointer;font-size:10.5px">誤りを報告</button>'
    : '';

  _showContextPane(
    '<div class="ctx-pane-header">' +
      '<div class="ctx-pane-title">' +
        esc(srcA.fullTitle || srcA.label) +
        ' <span style="opacity:.6;font-size:0.85em">→</span> ' +
        esc(tgtA.fullTitle || tgtA.label) +
      '</div>' +
      reportButton +
      '<div class="ctx-translate-wrap">' +
        '<span class="ctx-translate-label">翻訳</span>' +
        '<button class="ctx-toggle-btn' + (_translateToggle ? ' on' : '') + '" id="ctx-toggle"></button>' +
      '</div>' +
    '</div>' +
    '<div id="ctx-body"><div class="edge-ctx-loading">読み込み中…</div></div>'
  );

  var relationReportBtn = document.getElementById('relation-report-btn');
  if (relationReportBtn) {
    relationReportBtn.addEventListener('click', function() {
      var details = window.prompt(
        '誤りと思う具体的な根拠を入力してください。\n' +
        '分野が違うという印象だけでなく、原資料に存在しない、別著作の識別子である、などを記載してください。'
      );
      if (!details || !details.trim()) return;
      relationReportBtn.disabled = true;
      relationReportBtn.textContent = '報告中…';
      fetch('/api/relation/report', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          direction: edgeA.direction,
          item_key: edgeA.itemKey,
          external_paper_id: edgeA.externalPaperId,
          external_title: edgeA.externalTitle || '',
          reason: 'other',
          details: details.trim()
        })
      }).then(function(r) { return r.json(); }).then(function(d) {
        if (d.error) throw new Error(d.error);
        relationReportBtn.textContent = '報告済み';
        relationReportBtn.title = 'メンテナンス時の確認待ちです';
      }).catch(function(e) {
        relationReportBtn.disabled = false;
        relationReportBtn.textContent = '誤りを報告';
        window.alert('報告できませんでした: ' + e.message);
      });
    });
  }

  fetch('/api/edge/contexts?src=' + encodeURIComponent(src) + '&tgt=' + encodeURIComponent(tgt))
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (myReq !== _ctxPaneReq) return;  // 別のノード/エッジに切り替え済み
      var ctxs = data.contexts || [];
      var ctxBody = document.getElementById('ctx-body');
      if (!ctxBody) return;

      function _renderContexts() {
        var html = '';
        if (ctxs.length === 0) {
          html = '<div style="padding:8px 0;opacity:.6;font-size:0.85em">引用コンテクスト情報なし</div>';
        } else {
          ctxs.forEach(function(c, i) {
            html += '<div class="edge-ctx">';
            if (c.page) html += '<div class="edge-ctx-page">p.' + esc(String(c.page)) + '</div>';
            html += '<div class="edge-ctx-snippet">' + esc(c.snippet || '') + '</div>';
            html += '<div class="edge-ctx-translation" data-idx="' + i + '" style="display:none">' +
                    (c.translation ? esc(c.translation) : '<span style="opacity:.5">翻訳中…</span>') +
                    '</div>';
            html += '</div>';
          });
        }
        ctxBody.innerHTML = html;

        var toggleBtn = document.getElementById('ctx-toggle');
        if (toggleBtn) {
          toggleBtn.addEventListener('click', function() {
            _translateToggle = !_translateToggle;
            toggleBtn.className = 'ctx-toggle-btn' + (_translateToggle ? ' on' : '');
            _applyToggle();
          });
        }
        if (_translateToggle) { _applyToggle(); }
      }

      function _applyToggle() {
        var ctxBodyEl = document.getElementById('ctx-body');
        if (!ctxBodyEl) return;
        if (!_translateToggle) {
          ctxBodyEl.querySelectorAll('.edge-ctx-translation').forEach(function(d) {
            d.style.display = 'none';
          });
          return;
        }
        ctxs.forEach(function(c, i) {
          var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + i + '"]');
          if (!div) return;
          if (c.translation) { div.textContent = c.translation; div.style.display = 'block'; }
        });
        var pending = [];
        ctxs.forEach(function(c, i) {
          if (!c.translation && (c.snippet || '')) pending.push({ idx: i, snippet: c.snippet });
        });
        if (pending.length === 0) return;
        pending.forEach(function(p) {
          var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + p.idx + '"]');
          if (div) { div.innerHTML = '<span style="opacity:.5">翻訳中…</span>'; div.style.display = 'block'; }
        });
        fetch('/api/translate/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texts: pending.map(function(p) { return p.snippet; }) })
        })
          .then(function(r) { return r.json(); })
          .then(function(d) {
            if (d.error) {
              pending.forEach(function(p) {
                var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + p.idx + '"]');
                if (div) { div.style.color = '#f87171'; div.textContent = 'エラー: ' + d.error; }
              });
            } else {
              pending.forEach(function(p, j) {
                ctxs[p.idx].translation = d.translations[j];
                var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + p.idx + '"]');
                if (div) { div.style.color = ''; div.textContent = d.translations[j]; }
              });
            }
          })
          .catch(function(e) {
            pending.forEach(function(p) {
              var div = ctxBodyEl.querySelector('.edge-ctx-translation[data-idx="' + p.idx + '"]');
              if (div) { div.style.color = '#f87171'; div.textContent = 'エラー: ' + e.message; }
            });
          });
      }

      _renderContexts();
    })
    .catch(function(err) {
      if (myReq !== _ctxPaneReq) return;
      var ctxBody = document.getElementById('ctx-body');
      if (ctxBody) ctxBody.innerHTML =
        '<div style="color:#f87171;font-size:12px">エラー: ' + esc(err.message) + '</div>';
    });
});

// Click: toggle selection focus (same node again → deselect)
renderer.on('clickNode', function (ev) {
  if (hasDragged) return;
  _activeEdge = null;
  if (selectedNode === ev.node) { clearSelection(); }
  else {
    recomputeSelection(ev.node);
    if (window._panToNode) window._panToNode(ev.node);
    // Show abstract in context pane (zotero → 概要+階層要約 / 外部論文 → S2概要)
    var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === ev.node; });
    if (n && n.group === 'zotero' && n.itemKey) {
      _showNodeAbstract(ev.node);
    } else if (n && (n.group === 'external' || n.group === 'reference')) {
      _showExternalAbstract(ev.node);
    }
  }
});

// Click on empty canvas → deselect, or cluster-filter in semantic mode
renderer.on('clickStage', function () {
  _activeEdge = null;

  // 意味マップモード: クラスタクリック検出
  if (_viewMode === 'semantic' && _clusterData.length) {
    var dim = renderer.getDimensions();
    var rx = _lastCanvasClick.x / dim.width;
    var ry = _lastCanvasClick.y / dim.height;
    var gpos = renderer.viewportToGraph({ x: rx, y: ry });
    var hitCluster = false;
    for (var ci = 0; ci < _clusterData.length; ci++) {
      var hull = _clusterData[ci].hull;
      if (!hull || hull.length < 3) continue;
      if (_pointInPolygon(gpos.x, gpos.y, hull)) {
        _selectCluster(_selectedClusterId === _clusterData[ci].id ? -1 : _clusterData[ci].id);
        hitCluster = true;
        break;
      }
    }
    if (hitCluster) return;
    if (_selectedClusterId >= 0) _selectCluster(-1);
    // クラスタ外クリック時はノード選択解除にフォールスルー
  }

  if (selectedNode !== null) { clearSelection(); }
});

/* ── 5. Smooth zoom ─────────────────────────────────────── */
var zoomVel = 0, zoomId = null;
container.addEventListener('wheel', function (e) {
  e.preventDefault(); e.stopPropagation();
  zoomVel = Math.max(-0.06, Math.min(0.06, zoomVel + (e.deltaY < 0 ? 0.008 : -0.008)));
  if (!zoomId) (function az() {
    if (Math.abs(zoomVel) < 0.001) { zoomId = null; return; }
    var newRatio = Math.max(0.01, Math.min(20, camera.getState().ratio * (1 - zoomVel)));
    camera.setState(renderer.getViewportZoomedState({ x: mouseX, y: mouseY }, newRatio));
    zoomVel *= 0.72;
    zoomId = requestAnimationFrame(az);
  })();
}, { capture: true, passive: false });

/* ── 6. Pan inertia + Node drag ─────────────────────────── */
// draggedNode: ドラッグ中のノードID (null = パン中)
// isPanning:   カメラパン中フラグ
var isPanning = false, velCamX = 0, velCamY = 0, inertiaId = null;
var prevCamX = 0, prevCamY = 0;
var draggedNode = null;
var hasDragged  = false;  // ドラッグが実際に起きたかのフラグ（クリック判定用）
var dragDownX   = 0, dragDownY = 0;
var mc = renderer.getMouseCaptor();

// ノードのdown: ドラッグ開始準備（意味マップモードでは無効）
renderer.on('downNode', function (ev) {
  if (_viewMode === 'semantic') return;  // 意味マップではドラッグ不可
  draggedNode = ev.node;
  hasDragged  = false;
  dragDownX   = ev.event.x;
  dragDownY   = ev.event.y;
  isPanning   = false;
  cancelAnimationFrame(inertiaId); inertiaId = null;
  // D3にノードを固定させてシミュレーションに引き戻されないようにする
  var d3n = d3nodeById[draggedNode];
  if (d3n) {
    var attrs = graph.getNodeAttributes(draggedNode);
    d3n.fx = attrs.x; d3n.fy = attrs.y;
  }
});

// キャンバスのdown: パン開始 (ノードドラッグ中は無視)
mc.on('mousedown', function () {
  if (draggedNode) return;
  cancelAnimationFrame(inertiaId); inertiaId = null;
  isPanning = true;
  velCamX = velCamY = 0;
  var s = camera.getState(); prevCamX = s.x; prevCamY = s.y;
});

// mousemovebody: ノードドラッグ or パン速度追跡
// sigma は mousemovebody でカメラを動かすので、ノードドラッグ時は
// ev.preventSigmaDefault() でそれを止める
mc.on('mousemovebody', function (ev) {
  if (draggedNode) {
    // 3px以上動いたら「ドラッグあり」とみなす
    if (!hasDragged) {
      var dx = ev.x - dragDownX, dy = ev.y - dragDownY;
      if (dx*dx + dy*dy > 9) hasDragged = true;
    }
    if (!hasDragged) return;  // 微小な動きは無視（クリック扱いにする）
    // ノードドラッグ: マウス位置をグラフ座標に変換してノードを移動
    var gpos = renderer.viewportToGraph({ x: ev.x, y: ev.y });
    graph.setNodeAttribute(draggedNode, 'x', gpos.x);
    graph.setNodeAttribute(draggedNode, 'y', gpos.y);
    var d3n = d3nodeById[draggedNode];
    if (d3n) { d3n.fx = gpos.x; d3n.fy = gpos.y; d3n.vx = 0; d3n.vy = 0; }
    // sigmaのカメラパンを抑制
    ev.preventSigmaDefault();
    ev.original.preventDefault();
    ev.original.stopPropagation();
    renderer.refresh();
    if (layoutDone && _viewMode !== 'semantic') {
      layoutDone = false;
      simulation.alpha(0.2).restart();
      requestAnimationFrame(layoutStep);
    }
  } else if (isPanning) {
    // パン中: カメラ速度を追跡（イナーシャ用）
    var s = camera.getState(), a = 0.45;
    velCamX = velCamX * (1-a) + (s.x - prevCamX) * a;
    velCamY = velCamY * (1-a) + (s.y - prevCamY) * a;
    prevCamX = s.x; prevCamY = s.y;
  }
});

mc.on('mouseup', function () {
  if (draggedNode) {
    // ノードドラッグ終了
    var d3n = d3nodeById[draggedNode];
    if (d3n) { d3n.fx = null; d3n.fy = null; }
    draggedNode = null;
    return;
  }
  if (!isPanning) return;
  isPanning = false;
  if (Math.abs(velCamX) + Math.abs(velCamY) < 0.00005) return;
  (function glide() {
    velCamX *= 0.93; velCamY *= 0.93;
    if (Math.abs(velCamX) + Math.abs(velCamY) < 0.00003) { inertiaId = null; return; }
    var s = camera.getState();
    camera.setState({ x: s.x + velCamX, y: s.y + velCamY });
    inertiaId = requestAnimationFrame(glide);
  })();
});

// window mouseup: ブラウザ外でボタンを離した場合のフォールバック
window.addEventListener('mouseup', function () {
  if (draggedNode) {
    var d3n = d3nodeById[draggedNode];
    if (d3n) { d3n.fx = null; d3n.fy = null; }
    draggedNode = null;
  }
});

/* ── 7. D3-force layout ──────────────────────────────────────
   If the Python backend pre-computed positions (FA2), we skip the initial
   layout entirely and just build the D3 node/link arrays for drag support
   and the expand/collapse reheat.
   Otherwise we fall back to progressive batch loading. */

var gup = 8;

var nNodes = GRAPH_DATA.nodes.length;
// ノードサイズに比例した斥力（大きいノードほど周囲を広く押しのける）
var chargeBase = nNodes < 100  ? -1200 :
                 nNodes < 500  ? -800  :
                 nNodes < 2000 ? -500  : -280;
function chargeStr(d) {
  var sz = 5;
  try { sz = graph.getNodeAttribute(d.id, 'size') || 5; } catch (_) {}
  // size は 1〜8 の範囲。基準サイズ 2.5 を 1.0 として二乗比例
  return chargeBase * (sz / 2.5) * (sz / 2.5);
}
var linkDist  = nNodes < 100  ?  300  :
                nNodes < 500  ?  260  :
                nNodes < 2000 ?  170  : 110;

// ── D3 node/link arrays (used for drag + expand/collapse reheat) ──
var d3nodes = [], d3nodeById = {};
var activeNodeIds = new Set();

function _addD3Nodes(ids) {
  ids.forEach(function (id) {
    if (d3nodeById[id]) return;
    var attrs = graph.getNodeAttributes(id);
    var n = { id: id, x: attrs.x, y: attrs.y };
    d3nodes.push(n); d3nodeById[id] = n; activeNodeIds.add(id);
  });
}

var d3links = [];
function _rebuildLinks() {
  d3links = [];
  graph.forEachEdge(function (e, attrs, src, tgt) {
    if (activeNodeIds.has(src) && activeNodeIds.has(tgt))
      d3links.push({ source: src, target: tgt });
  });
}

var simulation = d3.forceSimulation([])
  .force('charge', d3.forceManyBody().strength(chargeStr).theta(0.8))
  .force('link',   d3.forceLink([]).id(function (d) { return d.id; }).distance(linkDist))
  .force('collide', d3.forceCollide()
    .radius(function (d) {
      var sz = 5;
      try { sz = graph.getNodeAttribute(d.id, 'size') || 5; } catch (_) {}
      return sz * gup * 1.6;
    })
    .strength(0.85).iterations(3))
  .force('x', d3.forceX(0).strength(0.05))
  .force('y', d3.forceY(0).strength(0.05))
  .velocityDecay(0.45)
  .alphaDecay(0.012)
  .stop();

window.__sim = simulation;  // simulation 宣言後にグローバル公開

var badge    = document.getElementById('layout-badge');
var badgePct = document.getElementById('layout-pct');
var layoutDone = false;

function finishLayout() {
  if (layoutDone) return;
  layoutDone = true;
  badge.style.display = 'none';
  console.log('[RAG] layout ready');
  // 90% 未満で完了した場合（alphaMin到達）や最終フィット未実施ならここでフィット
  if (!_finalFitDone && !_layoutUserInteracted) _fitView(true);
  else renderer.refresh();
}

document.getElementById('layout-skip').addEventListener('click', finishLayout);

// ── レイアウト中のカメラフィット ──────────────────────────────────────────
// ユーザーがスクロール・パンした場合は自動フィットをスキップする。
var _layoutUserInteracted = false;
var _layoutFitting = false;   // 自分のコードによるカメラ変更中フラグ

// camera の updated イベント: 自分のコード以外によるカメラ変更を検出
camera.on('updated', function() {
  if (!_layoutFitting && !layoutDone) _layoutUserInteracted = true;
});

function _fitView(animated) {
  var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  graph.forEachNode(function(id, attrs) {
    if (attrs.x != null) {
      if (attrs.x < minX) minX = attrs.x; if (attrs.x > maxX) maxX = attrs.x;
      if (attrs.y < minY) minY = attrs.y; if (attrs.y > maxY) maxY = attrs.y;
    }
  });
  if (!isFinite(minX)) return;
  var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  var dim = renderer.getDimensions();
  var S = Math.min(dim.width, dim.height), ratio2 = 2 * bboxR;
  var newRatio = Math.max(
    (maxX - minX) * S / (0.88 * ratio2 * dim.width),
    (maxY - minY) * S / (0.88 * ratio2 * dim.height)
  );
  newRatio = Math.max(0.1, Math.min(newRatio, 5));
  var target = { x: cx / ratio2 + 0.5, y: cy / ratio2 + 0.5, ratio: newRatio };
  _layoutFitting = true;
  if (animated) {
    camera.animate(target, { duration: 600, easing: 'quadraticInOut' });
    setTimeout(function() { _layoutFitting = false; }, 650);
  } else {
    camera.setState(target);
    _layoutFitting = false;
  }
}

// 起動から 0.5 秒後に一度だけ即時フィット
setTimeout(function() { if (!layoutDone && !_layoutUserInteracted) _fitView(false); }, 500);

var _finalFitDone = false;

function layoutStep() {
  if (layoutDone) return;
  var tpf = nNodes < 500 ? 1 : nNodes < 2000 ? 2 : nNodes < 5000 ? 3 : 5;
  for (var i = 0; i < tpf; i++) simulation.tick();
  d3nodes.forEach(function (n) {
    if (isFinite(n.x) && isFinite(n.y)) {
      graph.setNodeAttribute(n.id, 'x', n.x);
      graph.setNodeAttribute(n.id, 'y', n.y);
    }
  });
  renderer.refresh();
  var pct = Math.round((1 - simulation.alpha()) * 100);
  var _label = badge.getAttribute('data-label') || 'レイアウト計算中';
  badgePct.textContent = _label + ' ' + pct + '%';
  // skipボタンはレイアウト中のみ表示
  var _skip = document.getElementById('layout-skip');
  if (_skip) _skip.style.display = '';
  // 90% 到達時に最終フィット（ユーザー操作なしの場合のみ）
  if (!_finalFitDone && pct >= 80 && !_layoutUserInteracted) {
    _finalFitDone = true;
    _fitView(true);
  }
  if (simulation.alpha() <= simulation.alphaMin()) { finishLayout(); return; }
  requestAnimationFrame(layoutStep);
}

if (_hasPrecomputedLayout) {
  // Pre-computed positions give a good starting point.
  // Run the simulation from there — nodes are already near equilibrium so
  // it converges in a few seconds, giving a visible but brief settling animation.
  graph.forEachNode(function (id) { _addD3Nodes([id]); });
  _rebuildLinks();
  simulation.nodes(d3nodes);
  simulation.force('link').links(d3links);
  simulation.alpha(0.5);   // start from pre-computed → fast convergence
  badge.style.display = 'block';
  requestAnimationFrame(layoutStep);
  console.log('[RAG] pre-computed layout, running simulation from good start, nodes=' + d3nodes.length);
} else {
  // Fallback: progressive batch simulation.
  var _allNodes = { zotero: [], external: [], reference: [] };
  graph.forEachNode(function (id, attrs) {
    var g = attrs.group || 'external';
    if (_allNodes[g]) _allNodes[g].push(id); else _allNodes.external.push(id);
  });
  var BATCH_SIZE = 400, BATCH_TICKS = 40;
  var _batchQueue = [];
  var _arr1 = _allNodes.external, _arr2 = _allNodes.reference;
  for (var _bi = 0; _bi < _arr1.length; _bi += BATCH_SIZE)
    _batchQueue.push({ ids: _arr1.slice(_bi, _bi + BATCH_SIZE), phase: 'citers' });
  for (var _bi2 = 0; _bi2 < _arr2.length; _bi2 += BATCH_SIZE)
    _batchQueue.push({ ids: _arr2.slice(_bi2, _bi2 + BATCH_SIZE), phase: 'refs' });

  _addD3Nodes(_allNodes.zotero);
  _rebuildLinks();
  simulation.nodes(d3nodes);
  simulation.force('link').links(d3links);

  badge.style.display = 'block';
  var _tickCount = 0, _nextBatchAt = BATCH_TICKS * 2;
  var _origLayoutStep = layoutStep;
  layoutStep = function () {
    if (layoutDone) return;
    var tpf = nNodes < 500 ? 1 : nNodes < 2000 ? 2 : nNodes < 5000 ? 3 : 5;
    for (var i = 0; i < tpf; i++) { simulation.tick(); _tickCount++; }
    if (_batchQueue.length > 0 && _tickCount >= _nextBatchAt) {
      var batch = _batchQueue.shift();
      _addD3Nodes(batch.ids);
      _rebuildLinks();
      simulation.nodes(d3nodes); simulation.force('link').links(d3links);
      simulation.alpha(Math.max(simulation.alpha(), 0.25));
      _nextBatchAt = _tickCount + BATCH_TICKS;
    }
    d3nodes.forEach(function (n) {
      if (isFinite(n.x) && isFinite(n.y)) {
        graph.setNodeAttribute(n.id, 'x', n.x);
        graph.setNodeAttribute(n.id, 'y', n.y);
      }
    });
    renderer.refresh();
    var _lbl2 = badge.getAttribute('data-label') || 'レイアウト計算中';
    if (_batchQueue.length > 0) {
      badgePct.textContent = _lbl2 + ' ' + Math.round(d3nodes.length / nNodes * 100) + '%';
    } else {
      badgePct.textContent = _lbl2 + ' ' + Math.round((1 - simulation.alpha()) * 100) + '%';
    }
    if (_batchQueue.length === 0 && simulation.alpha() <= simulation.alphaMin()) {
      finishLayout(); return;
    }
    requestAnimationFrame(layoutStep);
  };
  requestAnimationFrame(layoutStep);
  console.log('[RAG] fallback progressive layout, batches=' + _batchQueue.length);
}

/* ── 8. Chunk drill-down ─────────────────────────────────── */

// ── Legend minimize ──────────────────────────────────────────────────────────
(function() {
  var legend = document.getElementById('rag-legend');
  var btn    = document.getElementById('legend-minimize');
  btn.addEventListener('click', function() {
    var minimized = legend.classList.toggle('minimized');
    btn.textContent = minimized ? '+' : '−';
    btn.title       = minimized ? '展開' : '最小化';
  });

  // Section collapse/expand
  legend.querySelectorAll('.sect-header').forEach(function(hdr) {
    hdr.addEventListener('click', function() {
      var section = this.dataset.section;
      var body = legend.querySelector('.sect-body[data-section="' + section + '"]');
      var isOpen = this.classList.toggle('open');
      body.classList.toggle('open', isOpen);
    });
  });
})();

// ── Collection filter ────────────────────────────────────────────────────────
(function() {
  var sel     = document.getElementById('col-filter');
  var msg     = document.getElementById('col-filter-msg');
  var reloadBtn = document.getElementById('col-reload');

  function _applyColFilter() {
    if (window.__r) window.__r.refresh();
    if (window._renderList) window._renderList();
    // 表示ノードだけでレイアウトを再計算
    _relayoutVisible();
  }

  function _isNodeVisible(id, attrs) {
    // CCフィルタチェック
    if (attrs.group === 'external'  && filterCiterCC > 0 && (attrs.cc || 0) < filterCiterCC) return false;
    if (attrs.group === 'reference' && filterRefCC   > 0 && (attrs.cc || 0) < filterRefCC)   return false;

    var colKeys = window._colItemKeys;
    if (colKeys === null) return true;
    if (attrs.group === 'zotero' || attrs.group === 'chunk') {
      var nk = attrs.itemKey || (id.indexOf(':') >= 0 ? id.split(':')[1] : id);
      return colKeys.has(nk);
    }
    // 外部論文: フィルタ後の Zotero ノードに隣接していれば表示
    var ok = false;
    try {
      window.__g.forEachNeighbor(id, function(nid, nd) {
        if (nd.group === 'zotero' || nd.group === 'chunk') {
          var nk2 = nd.itemKey || (nid.indexOf(':') >= 0 ? nid.split(':')[1] : nid);
          if (colKeys.has(nk2)) ok = true;
        }
      });
    } catch(_) {}
    return ok;
  }

  var _colSimRaf = null;
  var _colBadge  = document.getElementById('col-layout-badge');
  var _colPct    = document.getElementById('col-layout-pct');

  function _relayoutVisible() {
    console.log('[col-relayout] start, __sim=', !!window.__sim, '__g=', !!window.__g);
    if (!window.__sim || !window.__g) return;
    var g   = window.__g;
    var sim = window.__sim;

    if (_colSimRaf) { cancelAnimationFrame(_colSimRaf); _colSimRaf = null; }
    _colBadge.classList.remove('show');
    sim.stop();

    var visIds = [];
    g.forEachNode(function(id, attrs) {
      if (_isNodeVisible(id, attrs)) visIds.push(id);
    });
    console.log('[col-relayout] visible nodes:', visIds.length, '/', g.order);
    if (visIds.length === 0) return;

    var visSet   = new Set(visIds);
    var visNodes = visIds.map(function(id) {
      var a = g.getNodeAttributes(id);
      return { id: id, x: a.x || 0, y: a.y || 0 };
    });
    var visLinks = [];
    g.forEachEdge(function(e, attrs, src, tgt) {
      if (visSet.has(src) && visSet.has(tgt))
        visLinks.push({ source: src, target: tgt });
    });
    console.log('[col-relayout] vis links:', visLinks.length, 'alpha before:', sim.alpha());

    sim.nodes(visNodes);
    sim.force('link').links(visLinks);
    sim.alpha(0.6);
    console.log('[col-relayout] alpha after set:', sim.alpha(), 'alphaMin:', sim.alphaMin());

    var _startAlpha = sim.alpha();
    _colBadge.classList.add('show');
    _colPct.textContent = '0%';

    var _tickCount = 0;
    var _fitTriggered = false;
    function _colTick() {
      for (var i = 0; i < 4; i++) sim.tick();
      _tickCount++;
      visNodes.forEach(function(n) {
        g.setNodeAttribute(n.id, 'x', n.x);
        g.setNodeAttribute(n.id, 'y', n.y);
      });
      window.__r.refresh();
      var pct = Math.min(99, Math.round((1 - sim.alpha() / _startAlpha) * 100));
      _colPct.textContent = pct + '%';
      if (_tickCount === 1) console.log('[col-relayout] first tick done, alpha=', sim.alpha());
      if (!_fitTriggered && pct >= 80) {
        _fitTriggered = true;
        _fitVisibleNodes(true);
      }
      if (sim.alpha() > sim.alphaMin()) {
        _colSimRaf = requestAnimationFrame(_colTick);
      } else {
        console.log('[col-relayout] converged after', _tickCount, 'frames');
        _colPct.textContent = '100%';
        setTimeout(function() { _colBadge.classList.remove('show'); }, 600);
        _colSimRaf = null;
      }
    }
    _colSimRaf = requestAnimationFrame(_colTick);
    console.log('[col-relayout] rAF scheduled, id=', _colSimRaf);
  }

  function _fitVisibleNodes(animated) {
    if (!window.__r || !window.__g) return;
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    window.__g.forEachNode(function(id, attrs) {
      if (_isNodeVisible(id, attrs) && attrs.x != null) {
        if (attrs.x < minX) minX = attrs.x; if (attrs.x > maxX) maxX = attrs.x;
        if (attrs.y < minY) minY = attrs.y; if (attrs.y > maxY) maxY = attrs.y;
      }
    });
    if (!isFinite(minX)) return;
    var bboxR2 = window.__bboxR || 1;
    var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    var dim = window.__r.getDimensions();
    var ratio2 = 2 * bboxR2;
    var newRatio = Math.max(
      (maxX - minX) / (0.85 * ratio2),
      (maxY - minY) * dim.width / (0.85 * ratio2 * dim.height)
    );
    newRatio = Math.max(0.1, Math.min(newRatio, 8));
    var target = { x: cx / ratio2 + 0.5, y: cy / ratio2 + 0.5, ratio: newRatio };
    var cam = window.__r.getCamera();
    if (animated) cam.animate(target, { duration: 500, easing: 'quadraticInOut' });
    else cam.setState(target);
  }

  function _loadCollections() {
    sel.innerHTML = '<option value="">（読み込み中…）</option>';
    msg.style.display = 'none';
    fetch('/api/collections').then(function(r) { return r.json(); }).then(function(data) {
      if (data.error) {
        sel.innerHTML = '<option value="">（取得失敗）</option>';
        msg.textContent = data.message ||
          'Zotero を起動し、ローカルサーバーが立ち上がっているか確認してください。';
        msg.style.display = 'block';
        return;
      }
      var cols = data.collections || [];
      sel.innerHTML = '<option value="">すべてのコレクション</option>';
      cols.forEach(function(c) {
        var opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = c.path;
        opt.dataset.keys = JSON.stringify(c.item_keys);
        sel.appendChild(opt);
      });
    }).catch(function(e) {
      sel.innerHTML = '<option value="">（取得失敗）</option>';
      msg.textContent = 'Zotero を起動し、データベースにアクセスできるか確認してください。';
      msg.style.display = 'block';
    });
  }

  sel.addEventListener('change', function() {
    var opt = sel.options[sel.selectedIndex];
    if (!opt || opt.value === '') {
      window._colItemKeys = null;
      _applyColFilter();
      // フィルタ解除: 全ノードでレイアウト再計算
      if (window.__sim && window.__g) {
        var allNodes = [];
        var allById  = {};
        window.__g.forEachNode(function(id, attrs) {
          var n = { id: id, x: attrs.x || 0, y: attrs.y || 0 };
          allNodes.push(n); allById[id] = n;
        });
        var allLinks = [];
        window.__g.forEachEdge(function(e, attrs, src, tgt) {
          allLinks.push({ source: src, target: tgt });
        });
        window.__sim.stop();
        window.__sim.nodes(allNodes);
        window.__sim.force('link').links(allLinks);
        window.__sim.alpha(0.4).restart();
      }
    } else {
      var keys = JSON.parse(opt.dataset.keys || '[]');
      window._colItemKeys = new Set(keys);
      _applyColFilter();
    }
  });

  reloadBtn.addEventListener('click', _loadCollections);

  // グラフデータが届いたあとに読み込む（GRAPH_DATA が確定してから）
  // フォールバックとして DOMContentLoaded 後に即時実行
  _loadCollections();
  window._relayoutVisible = _relayoutVisible;
})();

// ── CC filter sliders (対数スケール: slider 0-100 → value 0-10000) ───────────
(function() {
  // slider=0 → 0, slider=25 → 10, slider=50 → 100, slider=75 → 1000, slider=100 → 10000
  function _sliderToVal(s) {
    s = parseInt(s);
    return s <= 0 ? 0 : Math.round(Math.pow(10, s / 25));
  }
  function _updateLabel(sliderId, labelId) {
    var v = _sliderToVal(document.getElementById(sliderId).value);
    var el = document.getElementById(labelId);
    if (el) el.textContent = v.toLocaleString();
    return v;
  }
  function _applyFilter() {
    filterCiterCC = _sliderToVal(document.getElementById('filter-citer-cc').value);
    filterRefCC   = _sliderToVal(document.getElementById('filter-ref-cc').value);
    renderer.refresh();

    // 表示数カウント更新（被引用元・参照先それぞれ）
    var visCiter = 0, visRef = 0, visTotal = 0;
    graph.forEachNode(function(n, a) {
      if (a.group === 'zotero') { visTotal++; return; }
      if (a.group === 'external') {
        if (filterCiterCC > 0 && (a.cc||0) < filterCiterCC) return;
        visCiter++; visTotal++;
      } else if (a.group === 'reference') {
        if (filterRefCC > 0 && (a.cc||0) < filterRefCC) return;
        visRef++; visTotal++;
      }
    });
    var ec = document.getElementById('stat-citer-vis');
    var er = document.getElementById('stat-ref-vis');
    var en = document.getElementById('stat-nodes');
    if (ec) ec.textContent = visCiter;
    if (er) er.textContent = visRef;
    if (en) en.textContent = visTotal;

    if (layoutDone && window._relayoutVisible) window._relayoutVisible();
  }
  // ラベルリアルタイム更新＋変更確定時にフィルタ適用
  document.getElementById('filter-citer-cc').addEventListener('input', function() {
    _updateLabel('filter-citer-cc', 'filter-citer-cc-val');
  });
  document.getElementById('filter-ref-cc').addEventListener('input', function() {
    _updateLabel('filter-ref-cc', 'filter-ref-cc-val');
  });
  document.getElementById('filter-citer-cc').addEventListener('change', _applyFilter);
  document.getElementById('filter-ref-cc').addEventListener('change', _applyFilter);
  // 初期ラベル設定
  _updateLabel('filter-citer-cc', 'filter-citer-cc-val');
  _updateLabel('filter-ref-cc', 'filter-ref-cc-val');

  // 起動時にフィルターを適用（input の初期値を変数に反映して描画を更新）
  _applyFilter();
})();

// ── 詳細パネル・共通ヘルパー（イベントハンドラとサイドバー両方から参照） ──
var detail          = document.getElementById('sb-detail');
var detailHeader    = document.getElementById('sb-detail-header');
var detailBody      = document.getElementById('sb-detail-body');
var _metaBanner     = document.getElementById('meta-edit-banner');
var _metaForm       = document.getElementById('meta-edit-form');
var _meTitle        = document.getElementById('me-title');
var _meAuthors      = document.getElementById('me-authors');
var _meYear         = document.getElementById('me-year');
var _meCitations    = document.getElementById('me-citations');
var _metaNodeId     = null;  // フォームが対象としているノードID
var detailResizeHnd = document.getElementById('sb-resize-y');
function _setDetailActive(active) {
  detail.className          = active ? 'sb-active' : '';
  detailResizeHnd.className = active ? 'sb-active' : '';
  if (!active) { detailHeader.innerHTML = ''; detailBody.innerHTML = ''; }
}
function kv(k, v) {
  return '<div class="sb-kv"><span class="sb-k">' + esc(k) + '</span><span class="sb-v">' + esc(v) + '</span></div>';
}
function kvH(k, vHtml) {
  return '<div class="sb-kv"><span class="sb-k">' + esc(k) + '</span><span class="sb-v">' + vHtml + '</span></div>';
}
// ── Zotero編集ヒントモーダル ───────────────────────────────────────────────
var _zehModal   = document.getElementById('zotero-edit-hint');
var _zehOpen    = document.getElementById('zeh-open');
var _zehClose   = document.getElementById('zeh-close');
var _zehSuppress = document.getElementById('zeh-suppress-chk');
var _zehKey     = '';
var _LS_KEY     = 'hideZoteroEditHint';

function _showZoteroHint(itemKey) {
  if (localStorage.getItem(_LS_KEY) === '1') {
    // 非表示設定済み → そのままZoteroを開く
    window.location.href = 'zotero://select/library/items/' + itemKey;
    return;
  }
  _zehKey = itemKey;
  _zehSuppress.checked = false;
  _zehModal.classList.add('show');
}

_zehOpen.addEventListener('click', function() {
  if (_zehSuppress.checked) localStorage.setItem(_LS_KEY, '1');
  _zehModal.classList.remove('show');
  window.location.href = 'zotero://select/library/items/' + _zehKey;
});
_zehClose.addEventListener('click', function() {
  if (_zehSuppress.checked) localStorage.setItem(_LS_KEY, '1');
  _zehModal.classList.remove('show');
});
// 背景クリックで閉じる
_zehModal.addEventListener('click', function(e) {
  if (e.target === _zehModal) _zehModal.classList.remove('show');
});

// 識別子（DOI/ISBN）行を生成するヘルパー
// isZotero=true → 編集ボタンはZoteroディープリンク
// isZotero=false → 編集ボタンはインライン入力に切り替え
function _kvIdentifier(label, linkHtml, rawVal, nodeId, field, isZotero, itemKey) {
  var editBtn;
  if (isZotero) {
    editBtn = '<button class="id-edit-btn"' +
              ' data-zotero-key="' + esc(itemKey) + '"' +
              ' title="Zoteroで編集">✎</button>';
  } else {
    editBtn = '<button class="id-edit-btn"' +
              ' data-node="' + esc(nodeId) + '"' +
              ' data-field="' + esc(field) + '"' +
              ' data-val="'   + esc(rawVal)  + '"' +
              ' title="編集">✎</button>';
  }
  return '<div class="sb-kv"><span class="sb-k">' + esc(label) + '</span>' +
         '<span class="sb-v" data-id-field="' + esc(field) + '">' +
           linkHtml + editBtn +
         '</span></div>';
}

// インライン編集を起動（外部論文専用）
function _startIdEdit(btn) {
  var nodeId = btn.dataset.node;
  var field  = btn.dataset.field;
  var curVal = btn.dataset.val || '';
  var span   = btn.parentElement;
  span.innerHTML =
    '<input class="id-edit-input" value="' + esc(curVal) + '">' +
    '<button class="id-cancel-btn" data-node="' + esc(nodeId) + '">✕</button>';
  var input = span.querySelector('.id-edit-input');
  input.focus(); input.select();
  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      _saveIdentifier(nodeId, field, input.value.trim());
    } else if (e.key === 'Escape') {
      showDetail(nodeId);
    }
  });
  span.querySelector('.id-cancel-btn').addEventListener('click', function() {
    showDetail(nodeId);
  });
}

// 識別子を保存してパネルを更新
// 重複警告バナー
var _dupWarn      = document.getElementById('dup-warn');
var _dupWarnTitle = document.getElementById('dup-warn-title');
var _dupWarnMsg   = document.getElementById('dup-warn-msg');
document.getElementById('dup-warn-close').addEventListener('click', function() {
  _dupWarn.classList.remove('show');
});
function _normIsbn(s) {
  return s.replace(/[-\s]/g, '').toLowerCase();
}
function _isbnSet(raw) {
  // スペース区切りで複数ISBNが入る場合があるため、各要素をnormして Set にする
  return new Set((raw || '').split(/\s+/).map(_normIsbn).filter(Boolean));
}
function _checkDuplicate(nodeId, field, newVal) {
  if (!newVal || field === 'title') return;
  var dups;
  if (field === 'doi') {
    var norm = newVal.trim().toLowerCase();
    dups = GRAPH_DATA.nodes.filter(function(nd) {
      if (nd.id === nodeId) return false;
      return (nd.doi || '').trim().toLowerCase() === norm;
    });
  } else {
    // ISBN: 入力値・保存値ともにスペース区切り複数対応 + ハイフン正規化
    var inputSet = _isbnSet(newVal);
    dups = GRAPH_DATA.nodes.filter(function(nd) {
      if (nd.id === nodeId) return false;
      var ndSet = _isbnSet(nd.isbn || '');
      for (var v of inputSet) { if (ndSet.has(v)) return true; }
      return false;
    });
  }
  if (dups.length === 0) { _dupWarn.classList.remove('show'); return; }
  var fieldLabel = field === 'doi' ? 'DOI' : 'ISBN';
  var names = dups.map(function(nd) { return '「' + (nd.fullTitle || nd.label || nd.id) + '」'; }).join('、');
  _dupWarnTitle.textContent = fieldLabel + ' が他のノードと重複しています';
  _dupWarnMsg.textContent = names + ' と同じ ' + fieldLabel + ' です。ブラウザをリロードすると同一ノードに統合されます。';
  _dupWarn.classList.add('show');
}

function _saveIdentifier(nodeId, field, newVal) {
  fetch('/api/node/identifier', {
    method:  'POST',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify({node_id: nodeId, field: field, value: newVal}),
  }).then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.ok) {
        // GRAPH_DATA（詳細パネル用）を更新
        var node = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
        if (node) {
          node[field] = newVal;
          if (field === 'title') { node.fullTitle = newVal; }
        }
        // sigma の graph オブジェクト（ラベル・ツールチップ）も更新
        if (graph.hasNode(nodeId)) {
          graph.setNodeAttribute(nodeId, field, newVal);
          if (field === 'title') {
            graph.setNodeAttribute(nodeId, 'fullTitle', newVal);
            graph.setNodeAttribute(nodeId, 'label', newVal.slice(0, 24));
          }
          renderer.refresh();
        }
        _checkDuplicate(nodeId, field, newVal);
        showDetail(nodeId);
        // DOI / ISBN 修正時はメタデータ確認バナーとフォームを表示
        if (field === 'doi' || field === 'isbn') {
          _showMetaEditForm(nodeId, true);
        }
      }
    });
}

// ── メタデータ手動編集フォーム ────────────────────────────────────────────────
function _showMetaEditForm(nodeId, showBanner) {
  var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
  if (!n) return;
  _metaNodeId          = nodeId;
  _meTitle.value       = n.fullTitle || n.label || '';
  _meAuthors.value     = n.authors || '';
  _meYear.value        = n.year || '';
  var _ccVal = n.citations != null ? n.citations : (n.cc != null ? n.cc : '');
  _meCitations.value   = _ccVal !== '' ? String(_ccVal) : '';
  _metaForm.classList.add('show');
  if (showBanner) _metaBanner.classList.add('show');
}

function _hideMetaForm() {
  _metaForm.classList.remove('show');
  _metaBanner.classList.remove('show');
  _metaNodeId = null;
}

function _saveMetaField(nodeId, field, val) {
  return fetch('/api/node/identifier', {
    method:  'POST',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify({node_id: nodeId, field: field, value: val}),
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) {
      var node = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
      if (node) {
        node[field] = val;
        if (field === 'title')     { node.fullTitle = val; }
        if (field === 'citations') { node.citations = val ? parseInt(val) : null;
                                     node.cc        = node.citations; }
      }
      if (graph.hasNode(nodeId)) {
        graph.setNodeAttribute(nodeId, field, val);
        if (field === 'title') {
          graph.setNodeAttribute(nodeId, 'fullTitle', val);
          graph.setNodeAttribute(nodeId, 'label', val.slice(0, 24));
        }
        if (field === 'citations') {
          var ccNum = val ? parseInt(val) : 0;
          graph.setNodeAttribute(nodeId, 'cc', ccNum);
          // ノードサイズを再計算（被引用数に比例）
          var newSize = Math.max(4, Math.min(28, 4 + Math.sqrt(ccNum) * 1.2));
          graph.setNodeAttribute(nodeId, 'size', newSize);
        }
      }
    }
    return d;
  });
}

document.getElementById('me-save').addEventListener('click', function() {
  if (!_metaNodeId) return;
  var nid = _metaNodeId;
  Promise.all([
    _saveMetaField(nid, 'title',     _meTitle.value.trim()),
    _saveMetaField(nid, 'authors',   _meAuthors.value.trim()),
    _saveMetaField(nid, 'year',      _meYear.value.trim()),
    _saveMetaField(nid, 'citations', _meCitations.value.trim()),
  ]).then(function() {
    renderer.refresh();
    showDetail(nid);
    _hideMetaForm();
  });
});

document.getElementById('me-cancel').addEventListener('click', _hideMetaForm);

function showDetail(nodeId) {
  var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
  if (!n) { _setDetailActive(false); return; }
  // 別ノードへ切り替えたらメタフォームを閉じる
  if (_metaNodeId && _metaNodeId !== nodeId) { _hideMetaForm(); }
  var doi  = n.doi  || '';
  var isbn = n.isbn || '';
  var isZotero   = (n.group === 'zotero');
  var isExternal = (n.group === 'external' || n.group === 'reference');
  var itemKey    = n.itemKey || '';

  var doiLinkHtml = doi
    ? '<a href="https://doi.org/' + encodeURIComponent(doi) + '" target="_blank" rel="noopener noreferrer">' + esc(doi) + '</a>'
    : '<span style="opacity:.5">—</span>';
  var isbnLinkHtml = isbn
    ? isbn.trim().split(/\s+/).map(function(i) {
        return '<a href="https://worldcat.org/isbn/' + encodeURIComponent(i) + '" target="_blank" rel="noopener noreferrer">' + esc(i) + '</a>';
      }).join('<span style="color:var(--text-dis)"> / </span>')
    : '<span style="opacity:.5">—</span>';

  _setDetailActive(true);

  // ヘッダー: タイトル（外部論文はタイトルも編集可）
  if (isExternal) {
    var titleEditBtn = '<button class="id-edit-btn"' +
      ' data-node="' + esc(n.id) + '"' +
      ' data-field="title"' +
      ' data-val="' + esc(n.fullTitle || n.label) + '"' +
      ' title="タイトルを編集">✎</button>';
    var metaEditBtn = '<button id="meta-edit-open" title="著者・年・タイトルを手動編集" style="' +
      'background:none;border:1px solid var(--outline-variant);border-radius:4px;' +
      'cursor:pointer;font-size:11px;color:var(--on-surface-variant);padding:2px 7px;' +
      'margin-left:6px;flex-shrink:0">メタデータ編集</button>';
    detailHeader.innerHTML =
      '<div class="sb-detail-title-row">' +
        '<div id="sb-detail-title" style="flex:1">' + esc(n.fullTitle || n.label) + titleEditBtn + '</div>' +
        metaEditBtn +
      '</div>';
    document.getElementById('meta-edit-open').addEventListener('click', function() {
      _showMetaEditForm(n.id, false);
    });
  } else {
    detailHeader.innerHTML =
      '<div class="sb-detail-title-row">' +
        '<div id="sb-detail-title">' + esc(n.fullTitle || n.label) + '</div>' +
      '</div>';
  }

  var body = '';
  body += kv('著者', n.authors || '—');
  body += kv('年', n.year || '—');
  body += _kvIdentifier('DOI',  doiLinkHtml,  doi,  n.id, 'doi',  isZotero, itemKey);
  body += _kvIdentifier('ISBN', isbnLinkHtml, isbn, n.id, 'isbn', isZotero, itemKey);
  body += kv('被引用数', n.citations != null ? Number(n.citations).toLocaleString()
                       : n.cc        != null ? Number(n.cc).toLocaleString() : '—');
  if (!isExternal) {
    body += kv('参照数', n.refCount ? Number(n.refCount).toLocaleString() : '—');
    body += kv('Key', n.itemKey || '');
  }
  detailBody.innerHTML = body;

  // 編集ボタンにイベントを登録（ヘッダー・ボディ両方）
  [detailHeader, detailBody].forEach(function(root) {
    // 外部論文：インライン編集
    root.querySelectorAll('.id-edit-btn[data-node]').forEach(function(btn) {
      btn.addEventListener('click', function(e) {
        e.preventDefault();
        _startIdEdit(btn);
      });
    });
    // Zoteroアイテム：ヒントモーダル → Zoteroを開く
    root.querySelectorAll('.id-edit-btn[data-zotero-key]').forEach(function(btn) {
      btn.addEventListener('click', function(e) {
        e.preventDefault();
        _showZoteroHint(btn.dataset.zoteroKey);
      });
    });
  });
}

/* ── 8a. Tab switching ───────────────────────────────────── */
var _ctxPane   = document.getElementById('sb-context-pane');
var _tabList   = document.getElementById('sb-tab-list');

function _switchTab(name) {
  document.querySelectorAll('.sb-tab').forEach(function(t) {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  if (name === 'list') {
    _tabList.style.display = '';
    _ctxPane.classList.remove('active');
  } else {
    _tabList.style.display = 'none';
    _ctxPane.classList.add('active');
  }
}

document.querySelectorAll('.sb-tab').forEach(function(tab) {
  tab.addEventListener('click', function() { _switchTab(tab.dataset.tab); });
});

/* ── 8b. Context pane: abstract + summary (node) / contexts (edge) ── */
var _abstrTranslateOn = false;
var _ctxPaneReq       = 0;  // 連続クリック時に古いfetch応答が上書きするのを防ぐトークン

function _showContextPane(html) {
  _ctxPane.innerHTML = html;
  _switchTab('context');
}

function _showNodeAbstract(nodeId) {
  var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
  if (!n || !n.itemKey) return;
  var myReq = ++_ctxPaneReq;
  InsightsPane.open(n, myReq);
}

function _applyAbstractToggle(abstract) {
  var transDiv = document.getElementById('abs-translation');
  if (!transDiv) return;
  if (!_abstrTranslateOn) {
    transDiv.classList.remove('show');
    return;
  }
  if (transDiv.dataset.cached) {
    transDiv.textContent = transDiv.dataset.cached;
    transDiv.classList.add('show');
    return;
  }
  transDiv.innerHTML = '<span style="opacity:.5">翻訳中…</span>';
  transDiv.classList.add('show');
  fetch('/api/translate/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts: [abstract] })
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.translations && d.translations[0]) {
        transDiv.dataset.cached = d.translations[0];
        transDiv.textContent = d.translations[0];
      } else {
        transDiv.textContent = d.error || '翻訳失敗';
      }
    })
    .catch(function(e) {
      transDiv.style.color = '#f87171';
      transDiv.textContent = 'エラー: ' + e.message;
    });
}

// 外部論文ノード（group=external/reference）の概要を取得して表示。
// Crossref（Abstract）優先 → S2（abstract + tldr）フォールバック。翻訳トグル付き。
// 本文チャンクが無いため階層要約は出さない。両ソースに情報が無ければ「概要情報なし」を明示。
function _showExternalAbstract(nodeId) {
  var n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
  if (!n) return;
  var myReq = ++_ctxPaneReq;
  var paperId = (n.id || '').replace(/^(paper|ref):/, '');
  var doi = n.doi || '';

  _showContextPane(
    '<div class="ctx-pane-header">' +
      '<div class="ctx-pane-title">' + esc(n.fullTitle || n.label) + '</div>' +
      '<div class="ctx-translate-wrap" id="ext-trans-wrap" style="display:none">' +
        '<span class="ctx-translate-label">翻訳</span>' +
        '<button class="ctx-toggle-btn' + (_abstrTranslateOn ? ' on' : '') + '" id="ext-toggle"></button>' +
      '</div>' +
    '</div>' +
    '<div id="ext-loading" style="font-size:12px;color:var(--text-dis)">概要を取得中…（Crossref / Semantic Scholar）</div>'
  );

  // Crossref は DOI、S2 フォールバックは paper_id を使うため両方渡す。
  var parts = [];
  if (doi)     parts.push('doi='      + encodeURIComponent(doi));
  if (paperId) parts.push('paper_id=' + encodeURIComponent(paperId));
  fetch('/api/node/external-abstract?' + parts.join('&'))
    .then(function(r) { return r.json().then(function(d){ return { ok: r.ok, d: d }; }); })
    .then(function(res) {
      if (myReq !== _ctxPaneReq) return;
      var loadEl = document.getElementById('ext-loading');
      if (!loadEl) return;
      var d = res.d || {};
      if (!res.ok || d.error) {
        loadEl.innerHTML = '<span style="color:#f87171">' + esc(d.error || '取得に失敗しました') + '</span>';
        return;
      }
      var abstract = d.abstract || '';
      var tldr     = d.tldr || '';
      if (!abstract && !tldr) {
        loadEl.outerHTML =
          '<div style="font-size:12px;color:var(--text-dis);line-height:1.7">' +
            'Crossref・Semantic Scholar のいずれにも概要情報がありません。<br>' +
            '参照できる情報がないため、要約は生成できません。' +
          '</div>';
        return;
      }
      // 概要本文を組み立て（tldr → abstract の順。両方あれば両方表示）
      var html = '';
      if (tldr) {
        html +=
          '<div style="margin-bottom:10px">' +
            '<div class="summary-section-label">S2 TL;DR</div>' +
            '<div class="abstract-text" id="ext-tldr-text">' + esc(tldr) + '</div>' +
            '<div class="abstract-translation" id="ext-tldr-trans"></div>' +
          '</div>';
      }
      if (abstract) {
        html +=
          '<div>' +
            '<div class="summary-section-label">Abstract</div>' +
            '<div class="abstract-text" id="ext-abs-text">' + esc(abstract) + '</div>' +
            '<div class="abstract-translation" id="ext-abs-trans"></div>' +
          '</div>';
      }
      loadEl.outerHTML = html;

      // 翻訳対象（英語と思われる場合だけトグルを出す）
      var transTargets = [];
      if (tldr)     transTargets.push({ src: tldr,     transId: 'ext-tldr-trans' });
      if (abstract) transTargets.push({ src: abstract, transId: 'ext-abs-trans' });
      var firstText = (tldr || abstract || '');
      var isEnglish = firstText.length > 20 && firstText.charCodeAt(0) < 0x100;
      var transWrap = document.getElementById('ext-trans-wrap');
      if (transWrap && isEnglish) transWrap.style.display = '';

      var toggleBtn = document.getElementById('ext-toggle');
      if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
          _abstrTranslateOn = !_abstrTranslateOn;
          toggleBtn.className = 'ctx-toggle-btn' + (_abstrTranslateOn ? ' on' : '');
          _applyExternalToggle(transTargets);
        });
      }
      if (_abstrTranslateOn) _applyExternalToggle(transTargets);
    })
    .catch(function(e) {
      if (myReq !== _ctxPaneReq) return;
      var loadEl = document.getElementById('ext-loading');
      if (loadEl) loadEl.innerHTML = '<span style="color:#f87171">エラー: ' + esc(e.message) + '</span>';
    });
}

// 外部論文の abstract / tldr をまとめて翻訳・表示する。
function _applyExternalToggle(targets) {
  if (!_abstrTranslateOn) {
    targets.forEach(function(t) {
      var div = document.getElementById(t.transId);
      if (div) div.classList.remove('show');
    });
    return;
  }
  // キャッシュ済みは即表示し、未翻訳のものだけ1リクエストで翻訳
  var pending = [];
  targets.forEach(function(t) {
    var div = document.getElementById(t.transId);
    if (!div) return;
    if (div.dataset.cached) {
      div.textContent = div.dataset.cached;
      div.classList.add('show');
    } else {
      div.innerHTML = '<span style="opacity:.5">翻訳中…</span>';
      div.classList.add('show');
      pending.push(t);
    }
  });
  if (pending.length === 0) return;
  fetch('/api/translate/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts: pending.map(function(t) { return t.src; }) })
  })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      pending.forEach(function(t, j) {
        var div = document.getElementById(t.transId);
        if (!div) return;
        if (d.error) { div.style.color = '#f87171'; div.textContent = 'エラー: ' + d.error; }
        else {
          var tr = d.translations[j];
          div.dataset.cached = tr;
          div.style.color = '';
          div.textContent = tr;
        }
      });
    })
    .catch(function(e) {
      pending.forEach(function(t) {
        var div = document.getElementById(t.transId);
        if (div) { div.style.color = '#f87171'; div.textContent = 'エラー: ' + e.message; }
      });
    });
}

function _fmtSummaryDate(iso) {
  if (!iso) return '';
  // SQLite CURRENT_TIMESTAMP は UTC ("YYYY-MM-DD HH:MM:SS")。Zが無ければ付与してパース。
  var s = iso.replace(' ', 'T');
  if (!/Z|[+-]\d\d:?\d\d$/.test(s)) s += 'Z';
  var d = new Date(s);
  return isNaN(d) ? '' : d.toLocaleDateString('ja-JP');
}

function _renderSummarySection(itemKey, summaryData) {
  var sec = document.getElementById('abs-summary-section');
  if (!sec) return;
  sec.style.display = '';
  var bodyEl = document.getElementById('abs-summary-body');
  if (!bodyEl) return;

  function _renderSummaryBody(sd) {
    if (!sd) {
      bodyEl.innerHTML =
        '<div class="ctx-pane-empty">階層要約は未生成です。Maintenance Widgetで要約更新を実行してください。</div>';
    } else {
      var modelLabel = sd.model ? ' (' + esc(sd.model) + ')' : '';
      var dateStr = _fmtSummaryDate(sd.updated_at);
      var reportBadge = sd.report_status === 'pending'
        ? '<span class="reported-badge">報告済み・判定待ち</span>'
        : (sd.report_status === 'disabled' ? '<span class="reported-badge">品質判定により検索対象外</span>' : '');
      bodyEl.innerHTML =
        '<div class="summary-text" id="sum-text">' + esc(sd.summary) + '</div>' +
        '<div style="font-size:10.5px;color:var(--text-dis);margin-top:4px">' +
          esc(dateStr) + modelLabel +
        '</div>' +
        reportBadge +
        '<div class="summary-actions" id="sum-actions">' +
          '<button class="summary-btn" id="sum-report-btn"' +
            (sd.report_status === 'pending' || sd.report_status === 'disabled' ? ' disabled' : '') + '>問題を報告</button>' +
        '</div>';
    }
    var reportBtn = document.getElementById('sum-report-btn');
    if (reportBtn) {
      reportBtn.addEventListener('click', function() {
        _openQualityReport({
          targetType: 'item_summary', itemKey: itemKey, evidence: [],
          returnFocusSelector: '#ins-tab-overview',
          onSaved: function() {
            sd.report_status = 'pending';
            _renderSummaryBody(sd);
          }
        });
      });
    }
  }

  _renderSummaryBody(summaryData);
}

function _insightsApi(url, options) {
  return fetch(url, options).then(function(response) {
    return response.json().then(function(data) {
      if (!response.ok || data.error) throw new Error(data.error || ('HTTP ' + response.status));
      return data;
    });
  });
}

var _qualityContext = null;
var _qualityReturnFocus = null;

function _showInsightsToast(message) {
  var old = document.querySelector('.insights-toast');
  if (old) old.remove();
  var toast = document.createElement('div');
  toast.className = 'insights-toast';
  toast.setAttribute('role', 'status');
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(function() { if (toast.parentNode) toast.remove(); }, 3500);
}

function _qualityReasons(targetType) {
  return [
    ['unsupported_claim', '原文にない内容を含む'],
    ['missing_context', '重要な限定条件が失われている'],
    ['wrong_work', '対象または因果関係の取り違え'],
    ['misleading_summary', '意味が通らない・誤解を招く'],
    ['other', 'その他']
  ];
}

function _ensureQualityDialog() {
  var overlay = document.getElementById('quality-report-overlay');
  if (overlay) return overlay;
  overlay = document.createElement('div');
  overlay.id = 'quality-report-overlay';
  overlay.className = 'quality-overlay';
  overlay.innerHTML =
    '<div class="quality-dialog" role="dialog" aria-modal="true" aria-labelledby="quality-title">' +
      '<h2 class="quality-title" id="quality-title">品質上の問題を報告</h2>' +
      '<div class="insights-help">報告しても内容はすぐには削除されません。メンテナンス時の確認対象になります。</div>' +
      '<label class="quality-field"><span>問題の種類</span><select class="insights-select" id="quality-reason"></select></label>' +
      '<label class="quality-field"><span>具体的な根拠（10文字以上）</span><textarea class="quality-details" id="quality-details"></textarea></label>' +
      '<div class="quality-evidence" id="quality-evidence"></div>' +
      '<div class="quality-error" id="quality-error" role="alert"></div>' +
      '<div class="quality-actions">' +
        '<button class="insight-btn" id="quality-cancel">キャンセル</button>' +
        '<button class="insight-btn primary" id="quality-submit">報告する</button>' +
      '</div>' +
    '</div>';
  document.body.appendChild(overlay);

  function closeDialog() {
    var fallbackSelector = _qualityContext && _qualityContext.returnFocusSelector;
    overlay.classList.remove('show');
    _qualityContext = null;
    var focusTarget = _qualityReturnFocus;
    if (!focusTarget || !document.contains(focusTarget) || focusTarget.disabled) {
      focusTarget = fallbackSelector ? document.querySelector(fallbackSelector) : null;
    }
    if (focusTarget) focusTarget.focus();
    _qualityReturnFocus = null;
  }
  document.getElementById('quality-cancel').addEventListener('click', closeDialog);
  overlay.addEventListener('mousedown', function(event) {
    if (event.target === overlay) closeDialog();
  });
  overlay.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') { event.preventDefault(); closeDialog(); return; }
    if (event.key !== 'Tab') return;
    var focusable = Array.prototype.slice.call(overlay.querySelectorAll(
      'button:not([disabled]), select:not([disabled]), textarea:not([disabled]), input:not([disabled])'
    ));
    if (!focusable.length) return;
    var first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault(); last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault(); first.focus();
    }
  });
  document.getElementById('quality-submit').addEventListener('click', function() {
    if (!_qualityContext) return;
    var details = document.getElementById('quality-details').value.trim();
    var error = document.getElementById('quality-error');
    if (details.length < 10) {
      error.textContent = '具体的な根拠を10文字以上で入力してください。';
      return;
    }
    var submit = document.getElementById('quality-submit');
    submit.disabled = true;
    error.textContent = '';
    var selectedChunks = Array.prototype.slice.call(
      document.querySelectorAll('#quality-evidence input:checked')
    ).map(function(input) { return input.value; });
    var body = {
      target_type: _qualityContext.targetType,
      item_key: _qualityContext.itemKey || '',
      section_id: _qualityContext.sectionId || '',
      reason: document.getElementById('quality-reason').value,
      details: details,
      evidence_chunk_ids: selectedChunks
    };
    _insightsApi('/api/quality-report', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    }).then(function() {
      var onSaved = _qualityContext && _qualityContext.onSaved;
      if (onSaved) onSaved();
      closeDialog();
      _showInsightsToast('報告を保存しました。メンテナンス時の確認待ちです。');
    }).catch(function(err) {
      error.textContent = '報告できませんでした: ' + err.message;
    }).finally(function() { submit.disabled = false; });
  });
  return overlay;
}

function _openQualityReport(context, trigger) {
  var overlay = _ensureQualityDialog();
  _qualityContext = context;
  _qualityReturnFocus = trigger || document.activeElement;
  var reason = document.getElementById('quality-reason');
  reason.innerHTML = '';
  _qualityReasons(context.targetType).forEach(function(row) {
    var option = document.createElement('option');
    option.value = row[0]; option.textContent = row[1]; reason.appendChild(option);
  });
  document.getElementById('quality-details').value = '';
  document.getElementById('quality-error').textContent = '';
  var evidenceWrap = document.getElementById('quality-evidence');
  evidenceWrap.innerHTML = '';
  var evidence = context.evidence || [];
  if (evidence.length) {
    var heading = document.createElement('div');
    heading.className = 'insights-help'; heading.textContent = '関連する原文チャンク';
    evidenceWrap.appendChild(heading);
    var seen = {};
    evidence.forEach(function(row) {
      var chunkId = String(row.chunk_id || '');
      if (!chunkId || seen[chunkId]) return;
      seen[chunkId] = true;
      var label = document.createElement('label');
      var input = document.createElement('input');
      input.type = 'checkbox'; input.value = chunkId; input.checked = true;
      var text = document.createElement('span'); text.textContent = chunkId;
      label.appendChild(input); label.appendChild(text); evidenceWrap.appendChild(label);
    });
  }
  overlay.classList.add('show');
  setTimeout(function() { reason.focus(); }, 0);
}

var InsightsPane = (function() {
  var cache = {};
  var current = null;

  function skeleton() {
    return '<div aria-label="読み込み中"><div class="insights-skeleton"></div>' +
      '<div class="insights-skeleton"></div><div class="insights-skeleton"></div></div>';
  }

  function statusText(status, kind) {
    if (status === 'processed_empty') {
      return 'この資料には表示できる節要約がありません。';
    }
    return '節要約はまだ生成されていません。Maintenance Widgetで要約更新を実行してください。';
  }

  function shell(node, requestId) {
    var byline = [node.authors || '', node.year || ''].filter(Boolean).join(' · ');
    _showContextPane(
      '<div class="insights-shell">' +
        '<div class="insights-header">' +
          '<div class="insights-title-row">' +
            '<div class="insights-title-wrap"><div class="insights-title" id="insights-title">' +
              esc(node.fullTitle || node.label || '') + '</div>' +
              (byline ? '<div class="insights-byline">' + esc(byline) + '</div>' : '') +
            '</div>' +
            '<button class="insights-width-btn" id="insights-width" aria-label="サイドバー幅を切り替える" title="表示幅を切り替える">↔</button>' +
          '</div>' +
          '<div class="insights-tabs" role="tablist" aria-label="資料の詳細">' +
            '<button class="insights-tab" role="tab" id="ins-tab-overview" data-insight-tab="overview" aria-controls="ins-panel-overview" aria-selected="true">概要</button>' +
            '<button class="insights-tab" role="tab" id="ins-tab-outline" data-insight-tab="outline" aria-controls="ins-panel-outline" aria-selected="false">構造</button>' +
            '<button class="insights-tab" role="tab" id="ins-tab-sections" data-insight-tab="sections" aria-controls="ins-panel-sections" aria-selected="false">節要約 …</button>' +
            '<button class="insights-tab" role="tab" id="ins-tab-processing" data-insight-tab="processing" aria-controls="ins-panel-processing" aria-selected="false">処理状態</button>' +
          '</div>' +
        '</div>' +
        '<div class="insights-content" aria-live="polite">' +
          '<section class="insights-panel" role="tabpanel" id="ins-panel-overview" aria-labelledby="ins-tab-overview">' + skeleton() + '</section>' +
          '<section class="insights-panel" role="tabpanel" id="ins-panel-outline" aria-labelledby="ins-tab-outline" hidden></section>' +
          '<section class="insights-panel" role="tabpanel" id="ins-panel-sections" aria-labelledby="ins-tab-sections" hidden></section>' +
          '<section class="insights-panel" role="tabpanel" id="ins-panel-processing" aria-labelledby="ins-tab-processing" hidden></section>' +
        '</div>' +
      '</div>'
    );
    var titleEl = document.getElementById('insights-title');
    if (titleEl) titleEl.title = node.fullTitle || node.label || '';
    var widthBtn = document.getElementById('insights-width');
    if (widthBtn) widthBtn.addEventListener('click', function() {
      var sidebar = document.getElementById('sidebar');
      var width = sidebar.offsetWidth;
      if (width < 460) {
        window._insightsPreviousWidth = width;
        document.documentElement.style.setProperty('--sb-width', '480px');
      } else {
        document.documentElement.style.setProperty(
          '--sb-width', Math.max(280, window._insightsPreviousWidth || 340) + 'px'
        );
      }
      if (window.__r) { window.__r.resize(); window.__r.refresh(); }
    });

    var tabs = Array.prototype.slice.call(document.querySelectorAll('.insights-tab'));
    tabs.forEach(function(tab, index) {
      tab.tabIndex = index === 0 ? 0 : -1;
      tab.addEventListener('click', function() { activate(tab.dataset.insightTab); });
      tab.addEventListener('keydown', function(event) {
        var next = null;
        if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = tabs.length - 1;
        if (next !== null) {
          event.preventDefault(); tabs[next].focus(); activate(tabs[next].dataset.insightTab);
        }
      });
    });
  }

  function activate(name) {
    if (!current) return;
    current.activeTab = name;
    document.querySelectorAll('.insights-tab').forEach(function(tab) {
      var selected = tab.dataset.insightTab === name;
      tab.setAttribute('aria-selected', selected ? 'true' : 'false');
      tab.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('.insights-panel').forEach(function(panel) {
      panel.hidden = panel.id !== 'ins-panel-' + name;
    });
    if (name === 'sections' && !current.sectionsReady) setupSections(current);
    if (name === 'outline' && !current.outlineReady) setupOutline(current);
    if (name === 'processing') renderProcessing(current);
  }

  function updateCounts(state) {
    var sectionTab = document.getElementById('ins-tab-sections');
    if (sectionTab) sectionTab.textContent = '節要約 ' + Number(state.data.sections.count || 0);
  }

  function renderAbstract(state, abstract) {
    var body = document.getElementById('insight-abstract-body');
    if (!body) return;
    var isEnglish = abstract.length > 20 && abstract.charCodeAt(0) < 0x100;
    body.innerHTML =
      (isEnglish ? '<div class="ctx-translate-wrap" style="justify-content:flex-end;margin-bottom:6px">' +
        '<span class="ctx-translate-label">翻訳</span><button class="ctx-toggle-btn' +
        (_abstrTranslateOn ? ' on' : '') + '" id="abs-toggle" aria-label="概要の翻訳を切り替える"></button></div>' : '') +
      '<div class="abstract-text" id="abs-text">' + esc(abstract) + '</div>' +
      '<div class="abstract-translation" id="abs-translation"></div>';
    var toggle = document.getElementById('abs-toggle');
    if (toggle) toggle.addEventListener('click', function() {
      _abstrTranslateOn = !_abstrTranslateOn;
      toggle.className = 'ctx-toggle-btn' + (_abstrTranslateOn ? ' on' : '');
      _applyAbstractToggle(abstract);
    });
    if (_abstrTranslateOn && isEnglish) _applyAbstractToggle(abstract);
  }

  function fetchAbstract(state) {
    var body = document.getElementById('insight-abstract-body');
    if (!body) return;
    body.innerHTML = '<div class="insights-help">Zoteroから概要を取得中…</div>';
    _insightsApi('/api/node/fetch-abstract', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({item_key: state.itemKey})
    }).then(function(data) {
      if (current !== state || state.requestId !== _ctxPaneReq) return;
      if (data.abstract) {
        state.data.abstract = data.abstract; renderAbstract(state, data.abstract);
      } else {
        throw new Error(data.error || 'この資料には概要情報がありませんでした');
      }
    }).catch(function(error) {
      if (current !== state) return;
      body.innerHTML = '<div class="insights-error">' + esc(error.message) + '</div>' +
        '<button class="insight-btn" id="insight-abstract-retry">再取得</button>';
      var retry = document.getElementById('insight-abstract-retry');
      if (retry) retry.addEventListener('click', function() { fetchAbstract(state); });
    });
  }

  function renderOverview(state) {
    var panel = document.getElementById('ins-panel-overview');
    if (!panel || current !== state) return;
    panel.innerHTML =
      '<div class="insights-section"><div class="insights-label">ZOTERO 概要</div>' +
        '<div id="insight-abstract-body"></div></div>' +
      '<div class="summary-section" id="abs-summary-section"><div class="summary-section-label">階層要約</div>' +
        '<div id="abs-summary-body"></div></div>' +
      '<div class="processing-summary" id="processing-summary"></div>' +
      '<div class="insights-help" style="margin-top:16px">検索と読解を補助する自動生成情報です。重要な判断では原文を確認してください。</div>';
    if (state.data.abstract) renderAbstract(state, state.data.abstract);
    else fetchAbstract(state);
    _renderSummarySection(state.itemKey, state.data.summary);
    if (state.data.summary && state.data.summary.kind === 'extractive') {
      var label = document.querySelector('#abs-summary-section .summary-section-label');
      if (label) label.textContent = '抽出的要約';
    }
    renderProcessingSummary(state);
  }

  function processingLabel(status) {
    return ({
      complete: '完了', not_processed: '未処理', pending: '処理待ち',
      degraded: '代替処理中', needs_attention: '要対応',
      success: '完了', empty: '空結果', blocked: '処理不能', failed: '失敗',
      stale: '再処理待ち', running: '実行中', excluded: '除外'
    })[status] || status || '未確認';
  }

  function renderProcessingSummary(state) {
    var target = document.getElementById('processing-summary');
    if (!target || !state.data || !state.data.processing) return;
    var processing = state.data.processing;
    target.innerHTML = '<div class="insights-label">処理状態</div><div class="processing-overall">' +
      esc(processingLabel(processing.overall)) + '</div>' +
      '<button class="insight-btn" id="open-processing-tab">工程の詳細を表示</button>';
    var button = document.getElementById('open-processing-tab');
    if (button) button.addEventListener('click', function() {
      var tab = document.getElementById('ins-tab-processing'); if (tab) tab.click();
    });
  }

  function renderProcessing(state) {
    var panel = document.getElementById('ins-panel-processing');
    if (!panel || current !== state) return;
    var processing = (state.data && state.data.processing) || {};
    var artifacts = processing.artifacts || [];
    if (!artifacts.length) {
      panel.innerHTML = '<div class="insights-empty">まだ工程別の処理記録はありません。次回のメンテナンスで記録されます。</div>';
      return;
    }
    panel.innerHTML = '<div class="processing-overall">全体: ' + esc(processingLabel(processing.overall)) + '</div>' +
      artifacts.map(function(row) {
        var status = String(row.status || '');
        var detail = [row.reason_code || '', row.message || '', row.fallback_kind ? ('代替: ' + row.fallback_kind) : '']
          .filter(Boolean).join(' · ');
        return '<div class="processing-row"><div class="processing-row-head"><span>' +
          esc(row.artifact_type || '') + '</span><span class="processing-status ' +
          esc(status === 'blocked' || status === 'failed' ? 'needs-attention' : status) + '">' +
          esc(processingLabel(status)) + '</span></div>' +
          (detail ? '<div class="processing-detail">' + esc(detail) + '</div>' : '') +
          (row.updated_at ? '<div class="processing-detail">最終更新: ' + esc(row.updated_at) + '</div>' : '') +
          '</div>';
      }).join('');
  }

  function setupOutline(state) {
    state.outlineReady = true;
    var panel = document.getElementById('ins-panel-outline');
    if (!panel) return;
    panel.innerHTML = skeleton();
    _insightsApi('/api/node/outline?key=' + encodeURIComponent(state.itemKey)).then(function(data) {
      if (current !== state) return;
      var nodes = data.nodes || [];
      if (!nodes.length) {
        panel.innerHTML = '<div class="insights-empty">文書構造はまだ作成されていません。Maintenance Widgetでライブラリまたは要約を更新してください。</div>';
        return;
      }
      panel.innerHTML = '<div class="insights-help">原資料の見出し順を保持した構造です。要約は原文チャンクを探すための索引であり、引用根拠ではありません。</div>' +
        nodes.map(function(row) {
          var indent = Math.max(0, Number(row.depth || 0) - 1) * 12;
          var title = row.title || (row.node_type === 'semantic_segment' ? '本文範囲' : row.node_type);
          var meta = [row.node_type, row.content_chars ? (Number(row.content_chars).toLocaleString() + '字') : '',
            row.summary_kind === 'extractive' ? '抽出的要約' : (row.summary_kind ? 'AI要約' : '')].filter(Boolean).join(' · ');
          return '<div class="outline-node" style="margin-left:' + indent + 'px"><div class="outline-node-title">' +
            esc(title) + '</div><div class="outline-node-meta">' + esc(meta) + '</div>' +
            (row.summary ? '<div class="outline-node-summary">' + esc(row.summary) + '</div>' : '') + '</div>';
        }).join('');
    }).catch(function(error) {
      if (current === state) panel.innerHTML = '<div class="insights-error">構造を読み込めませんでした: ' + esc(error.message) + '</div>';
    });
  }

  function setupSections(state) {
    state.sectionsReady = true;
    state.sectionRows = [];
    state.sectionCursor = null;
    state.sectionQuery = '';
    var panel = document.getElementById('ins-panel-sections');
    if (!panel) return;
    if (state.data.sections.status !== 'available') {
      panel.innerHTML = '<div class="insights-empty">' +
        esc(statusText(state.data.sections.status, 'sections')) + '</div>';
      return;
    }
    panel.innerHTML =
      '<div class="insights-toolbar"><input class="insights-input" id="section-search" type="search" placeholder="節を検索…" aria-label="節要約を検索">' +
        '<div class="insights-countline" id="section-countline"></div></div>' +
      '<div id="section-list">' + skeleton() + '</div>' +
      '<button class="insight-btn insights-more" id="section-more" hidden>さらに表示</button>';
    var search = document.getElementById('section-search');
    var timer = null;
    search.addEventListener('input', function() {
      clearTimeout(timer);
      timer = setTimeout(function() {
        state.sectionQuery = search.value.trim(); loadSections(state, true);
      }, 200);
    });
    document.getElementById('section-more').addEventListener('click', function() {
      loadSections(state, false);
    });
    loadSections(state, true);
  }

  function loadSections(state, reset) {
    if (reset) { state.sectionRows = []; state.sectionCursor = null; }
    var list = document.getElementById('section-list');
    if (!list || current !== state) return;
    if (reset) list.innerHTML = skeleton();
    var fetchSeq = ++state.sectionFetchSeq;
    var url = '/api/node/sections?key=' + encodeURIComponent(state.itemKey) +
      '&q=' + encodeURIComponent(state.sectionQuery) + '&limit=50' +
      (state.sectionCursor ? '&cursor=' + encodeURIComponent(state.sectionCursor) : '');
    _insightsApi(url).then(function(data) {
      if (current !== state || state.requestId !== _ctxPaneReq || fetchSeq !== state.sectionFetchSeq) return;
      state.sectionRows = state.sectionRows.concat(data.items || []);
      state.sectionCursor = data.next_cursor;
      renderSections(state, data.total || 0);
    }).catch(function(error) {
      if (current !== state) return;
      list.innerHTML = '<div class="insights-error">読み込めませんでした: ' + esc(error.message) + '</div>' +
        '<button class="insight-btn insights-more" id="section-retry">再試行</button>';
      var retry = document.getElementById('section-retry');
      if (retry) retry.addEventListener('click', function() { loadSections(state, reset); });
    });
  }

  function renderSections(state, total) {
    var list = document.getElementById('section-list');
    var countline = document.getElementById('section-countline');
    if (!list) return;
    if (countline) countline.textContent = total + '節 · 文書順';
    if (!state.sectionRows.length) {
      list.innerHTML = '<div class="insights-empty">該当する節要約はありません。' +
        (state.sectionQuery ? '<br><button class="insight-btn" id="section-clear">条件を解除</button>' : '') + '</div>';
      var clear = document.getElementById('section-clear');
      if (clear) clear.addEventListener('click', function() {
        document.getElementById('section-search').value = ''; state.sectionQuery = ''; loadSections(state, true);
      });
    } else {
      list.innerHTML = state.sectionRows.map(function(row, index) {
        var title = row.chapter || ('節 ' + row.section_id);
        var reported = row.report_status === 'pending'
          ? '<span class="reported-badge">報告済み・判定待ち</span>'
          : (row.report_status === 'disabled' ? '<span class="reported-badge">品質判定により検索対象外</span>' : '');
        return '<article class="insight-card">' +
          '<button class="insight-card-toggle" data-section-index="' + index + '" aria-expanded="false" aria-controls="section-body-' + index + '">' +
            '<div class="insight-card-title">▸ ' + esc(title) + '</div>' +
            '<div class="insight-card-preview">' + esc(row.summary) + '</div>' +
          '</button>' +
          '<div class="insight-card-body" id="section-body-' + index + '" hidden>' +
            '<div class="insight-card-text">' + esc(row.summary) + '</div>' +
            '<div class="insight-meta">' + esc([row.model, _fmtSummaryDate(row.updated_at)].filter(Boolean).join(' · ')) + '</div>' +
            reported +
            '<div class="insight-actions">' +
              '<button class="insight-btn section-source-btn" data-section-index="' + index + '">原文を表示</button>' +
              '<button class="insight-btn section-report-btn" data-section-index="' + index + '"' +
                (row.report_status === 'pending' || row.report_status === 'disabled' ? ' disabled' : '') + '>問題を報告</button>' +
            '</div><div class="insight-source" id="section-source-' + index + '" hidden></div>' +
          '</div></article>';
      }).join('');
    }
    list.querySelectorAll('.insight-card-toggle').forEach(function(button) {
      button.addEventListener('click', function() {
        var body = document.getElementById(button.getAttribute('aria-controls'));
        var open = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', open ? 'false' : 'true');
        body.hidden = open;
        var title = button.querySelector('.insight-card-title');
        if (title) title.textContent = (open ? '▸ ' : '▾ ') +
          (state.sectionRows[Number(button.dataset.sectionIndex)].chapter ||
           ('節 ' + state.sectionRows[Number(button.dataset.sectionIndex)].section_id));
      });
    });
    list.querySelectorAll('.section-source-btn').forEach(function(button) {
      button.addEventListener('click', function() {
        showSectionSource(state, Number(button.dataset.sectionIndex), button);
      });
    });
    list.querySelectorAll('.section-report-btn').forEach(function(button) {
      button.addEventListener('click', function() {
        var row = state.sectionRows[Number(button.dataset.sectionIndex)];
        var source = state.sectionSourceCache[row.section_id];
        _openQualityReport({
          targetType: 'section_summary', itemKey: state.itemKey, sectionId: row.section_id,
          evidence: source ? source.chunks : [],
          returnFocusSelector: '.insight-card-toggle[data-section-index="' + button.dataset.sectionIndex + '"]',
          onSaved: function() { row.report_status = 'pending'; renderSections(state, total); }
        }, button);
      });
    });
    var more = document.getElementById('section-more');
    if (more) { more.hidden = !state.sectionCursor; more.textContent = 'さらに表示'; }
  }

  function showSectionSource(state, index, button) {
    var row = state.sectionRows[index];
    var wrap = document.getElementById('section-source-' + index);
    if (!row || !wrap) return;
    if (!wrap.hidden) { wrap.hidden = true; button.textContent = '原文を表示'; return; }
    wrap.hidden = false; button.textContent = '原文を隠す';
    function draw(data) {
      wrap.innerHTML = (data.chunks || []).map(function(chunk, i) {
        var locator = chunk.chunk_id + (chunk.page ? ' · p.' + chunk.page : '');
        return '<div class="source-chunk"><div class="source-chunk-head">原文 ' + (i + 1) + ' / ' +
          data.chunks.length + ' · ' + esc(locator) + '</div><blockquote class="source-quote">' +
          esc(chunk.text) + '</blockquote></div>';
      }).join('') || '<div class="insights-error">原文データを読み込めません。</div>';
    }
    if (state.sectionSourceCache[row.section_id]) {
      draw(state.sectionSourceCache[row.section_id]); return;
    }
    wrap.innerHTML = skeleton();
    _insightsApi('/api/node/section-source?key=' + encodeURIComponent(state.itemKey) +
      '&section_id=' + encodeURIComponent(row.section_id)).then(function(data) {
      if (current !== state) return;
      state.sectionSourceCache[row.section_id] = data; draw(data);
    }).catch(function(error) {
      wrap.innerHTML = '<div class="insights-error">原文を読み込めませんでした: ' + esc(error.message) + '</div>';
    });
  }

  function open(node, requestId) {
    var state = {
      node: node, itemKey: node.itemKey, requestId: requestId, activeTab: 'overview',
      data: null, sectionsReady: false, outlineReady: false,
      sectionSourceCache: {}, sectionFetchSeq: 0
    };
    current = state;
    shell(node, requestId);
    var cached = cache[node.itemKey];
    var promise = cached ? Promise.resolve(cached) : _insightsApi(
      '/api/node/insights?key=' + encodeURIComponent(node.itemKey)
    );
    promise.then(function(data) {
      if (current !== state || requestId !== _ctxPaneReq) return;
      cache[node.itemKey] = data;
      state.data = data; updateCounts(state); renderOverview(state);
    }).catch(function(error) {
      if (current !== state) return;
      var panel = document.getElementById('ins-panel-overview');
      if (panel) panel.innerHTML = '<div class="insights-error">読み込めませんでした: ' + esc(error.message) + '</div>' +
        '<button class="insight-btn insights-more" id="insights-retry">再試行</button>';
      var retry = document.getElementById('insights-retry');
      if (retry) retry.addEventListener('click', function() { cache[node.itemKey] = null; open(node, ++_ctxPaneReq); });
    });
  }

  return {open: open};
})();

/* ── 8. Sidebar: 資料一覧 ────────────────────────────────── */
(function() {
  var sbNodes = GRAPH_DATA.nodes.filter(function(n) {
    return n.group === 'zotero' || n.group === 'external' || n.group === 'reference';
  });
  var sortCol = 'citations';
  var sortDir = -1;   // -1=降順, 1=昇順
  var filterQ = '';
  var filterZoteroOnly = false;
  var sbActiveId = null;

  var toggle       = document.getElementById('sb-toggle');
  var sbCountEl    = document.getElementById('sb-count');
  var searchEl     = document.getElementById('sb-search');
  var zoteroOnlyEl = document.getElementById('sb-zotero-only');
  var tbody        = document.getElementById('sb-list-body');

  // ── トグル ──────────────────────────────────────────────
  toggle.addEventListener('click', function() {
    document.body.classList.toggle('sb-collapsed');
    toggle.textContent = document.body.classList.contains('sb-collapsed') ? '‹' : '›';
    setTimeout(function() { renderer.refresh(); }, 260);
  });

  // ── 列ヘッダーでソート ────────────────────────────────
  document.querySelectorAll('#sb-list-head-table th').forEach(function(th) {
    th.addEventListener('click', function() {
      var col = th.dataset.col;
      if (sortCol === col) { sortDir *= -1; }
      else { sortCol = col; sortDir = (col === 'title') ? 1 : -1; }
      document.querySelectorAll('#sb-list-head-table th').forEach(function(t) { t.className = ''; });
      th.className = sortDir === 1 ? 'sort-asc' : 'sort-desc';
      renderList();
    });
  });

  // ── 絞り込み ─────────────────────────────────────────
  searchEl.addEventListener('input', function() {
    filterQ = searchEl.value.toLowerCase();
    renderList();
  });
  zoteroOnlyEl.addEventListener('change', function() {
    filterZoteroOnly = zoteroOnlyEl.checked;
    renderList();
  });

  // ── ソートキー取得 ────────────────────────────────────
  function getVal(n, col) {
    if (col === 'title')     return (n.fullTitle || n.label || '').toLowerCase();
    if (col === 'year')      return parseInt(n.year) || 0;
    if (col === 'citations') return n.citations != null ? n.citations : (n.cc || 0);
    if (col === 'refCount')  return n.refCount  || 0;
    if (col === 'inZotero')  return n.group === 'zotero' ? 1 : 0;
    return '';
  }

  // ── リスト描画 ────────────────────────────────────────
  function _sbNodeVisible(n) {
    // CCフィルタ
    if (n.group === 'external'  && filterCiterCC > 0 && (n.cc || 0) < filterCiterCC) return false;
    if (n.group === 'reference' && filterRefCC   > 0 && (n.cc || 0) < filterRefCC)   return false;
    // コレクションフィルタ
    var colKeys = window._colItemKeys;
    if (colKeys === null) return true;
    if (n.group === 'zotero') {
      var nk = n.itemKey || (n.id.indexOf(':') >= 0 ? n.id.split(':')[1] : n.id);
      return colKeys.has(nk);
    }
    var edges = graph.edges(n.id);
    for (var ei = 0; ei < edges.length; ei++) {
      var nbr = graph.source(edges[ei]) === n.id
                ? graph.target(edges[ei]) : graph.source(edges[ei]);
      var nd = graph.getNodeAttributes(nbr);
      if (nd.group === 'zotero') {
        var nk2 = nd.itemKey || (nbr.indexOf(':') >= 0 ? nbr.split(':')[1] : nbr);
        if (colKeys.has(nk2)) return true;
      }
    }
    return false;
  }

  function renderList() {
    var sortFn = function(a, b) {
      var av = getVal(a, sortCol), bv = getVal(b, sortCol);
      if (typeof av === 'string') return av < bv ? -sortDir : av > bv ? sortDir : 0;
      return (av - bv) * sortDir;
    };

    var baseNodes = sbNodes.filter(function(n) {
      if (zoteroOnlyEl.checked && n.group !== 'zotero') return false;
      if (!filterQ) return true;
      return (n.fullTitle || n.label || '').toLowerCase().indexOf(filterQ) !== -1 ||
             (n.authors   || '').toLowerCase().indexOf(filterQ) !== -1;
    });

    var visNodes   = baseNodes.filter(function(n) { return  _sbNodeVisible(n); });
    var grayNodes  = baseNodes.filter(function(n) { return !_sbNodeVisible(n); });
    visNodes.sort(sortFn);
    grayNodes.sort(sortFn);

    sbCountEl.textContent = ' (' + visNodes.length + ')';
    tbody.innerHTML = '';

    function makeRow(n, grayed) {
      var tr = document.createElement('tr');
      if (n.id === sbActiveId) tr.className = 'sb-active';
      if (grayed) tr.style.opacity = '0.38';
      var titleStr = n.fullTitle || n.label || '';
      var shortTitle = titleStr.length > 34 ? titleStr.slice(0, 34) + '…' : titleStr;
      var citVal = n.citations != null ? n.citations : (n.cc != null ? n.cc : null);
      var inZ = n.group === 'zotero';
      tr.innerHTML =
        '<td style="max-width:145px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(shortTitle) + '</td>' +
        '<td class="sb-num">' + esc(n.year || '—') + '</td>' +
        '<td class="sb-num">' + (citVal != null ? Number(citVal).toLocaleString() : '—') + '</td>' +
        '<td class="sb-num">' + (n.refCount ? Number(n.refCount).toLocaleString() : '—') + '</td>' +
        '<td class="sb-num" style="color:' + (inZ ? 'var(--node-zotero)' : 'var(--on-surface-variant)') + '">' + (inZ ? '✓' : '—') + '</td>';
      if (grayed) {
        tr.addEventListener('click', function() {
          sbActiveId = n.id;
          renderList();
          showDetail(n.id);
        });
      } else {
        tr.addEventListener('click', function() { selectFromSidebar(n.id); });
      }
      tbody.appendChild(tr);
    }

    visNodes.forEach(function(n)  { makeRow(n, false); });
    grayNodes.forEach(function(n) { makeRow(n, true);  });
  }

  // ── サイドバーからノード選択 ──────────────────────────
  function selectFromSidebar(nodeId) {
    sbActiveId = nodeId;
    recomputeSelection(nodeId);
    renderer.refresh();
    renderList();
    showDetail(nodeId);
    window._panToNode(nodeId);
    var _n = GRAPH_DATA.nodes.find(function(nd) { return nd.id === nodeId; });
    if (_n && _n.group === 'zotero' && _n.itemKey) { _showNodeAbstract(nodeId); }
    else if (_n && (_n.group === 'external' || _n.group === 'reference')) { _showExternalAbstract(nodeId); }
    // アクティブ行をスクロールで表示
    setTimeout(function() {
      var activeRow = tbody.querySelector('tr.sb-active');
      if (activeRow) activeRow.scrollIntoView({ block: 'nearest' });
    }, 0);
  }

  // ── カメラをノードに向ける ────────────────────────────
  // viewportToGraph を2点で使い、「フレーム座標/グラフ座標」の正規化係数を逆算。
  // これでグラフ座標→フレーム座標の変換ができ、カメラ中心に設定する。
  // ── ノーマライゼーション係数キャッシュ ─────────────────
  // sigma のカメラ座標（framed unit）と graph 座標の変換係数を一度だけキャッシュする。
  // camera.setState 後に viewportToGraph が信頼できなくなるため、
  // カメラが初期位置(0.5,0.5)にある時点で計算する。
  //
  // sigma の framed 座標は「ビューポート幅を1とした割合」単位。
  //   viewportToGraph(center+100px) の差分 = scalePixel = graph/px
  //   normRatio = scalePixel × dim.width = graph / framed_unit = 2 × bboxR
  // sigma の normalizationFunction: apply(gx) = gx/(2×bboxR) + 0.5
  // bboxR はメインスコープで定義済み（setCustomBBox に渡した値）。
  // viewportToGraph に依存しないため、カメラ移動後でも正確。

  // sigma の normalizationFunction: apply(gx) = gx/(2×bboxR) + 0.5
  // bboxR はメインスコープで定義済み（setCustomBBox に渡した値）。
  // viewportToGraph に依存しないため、カメラ移動後でも正確。
  function _graphToFramed(gx, gy) {
    var ratio2 = 2 * bboxR;
    return {
      x: gx / ratio2 + 0.5,
      y: gy / ratio2 + 0.5,
    };
  }

  window._panToNode = function panToNode(nodeId) {
    try {
      var nx = graph.getNodeAttribute(nodeId, 'x');
      var ny = graph.getNodeAttribute(nodeId, 'y');
      if (nx == null || ny == null) { console.warn('[panToNode] no position'); return; }

      // 選択ノードを中心に固定し、表示中の隣接ノードの最大距離でズームを決める。
      // CCフィルターで非表示になっているノードは除外する。
      function _isVisible(n) {
        var attrs = graph.getNodeAttributes(n);
        if (attrs.group === 'external'  && filterCiterCC > 0 && (attrs.cc || 0) < filterCiterCC) return false;
        if (attrs.group === 'reference' && filterRefCC   > 0 && (attrs.cc || 0) < filterRefCC)   return false;
        return true;
      }

      // 選択ノード中心からの最大距離（X/Y 別）を収集
      var halfW = 0, halfH = 0;
      function _updateHalf(n) {
        if (!_isVisible(n)) return;
        var x = graph.getNodeAttribute(n, 'x'), y = graph.getNodeAttribute(n, 'y');
        if (x != null) halfW = Math.max(halfW, Math.abs(x - nx));
        if (y != null) halfH = Math.max(halfH, Math.abs(y - ny));
      }
      selectedNeighbors.forEach(_updateHalf);
      selectedChunkNeighbors.forEach(_updateHalf);

      // 選択ノードを画面中心へ
      var framed = _graphToFramed(nx, ny);

      // ratio 計算: 選択ノード中心で、隣接ノードが画面 80% に収まるよう
      // visible_graph_half_X = bboxR * dim.width * ratio / min(dim.width, dim.height)
      // → ratio = halfW * min / (0.8 * bboxR * dim.width) × 0.5  [half なので ×0.5 不要、full span = 2*half]
      var dim = renderer.getDimensions();
      var S   = Math.min(dim.width, dim.height);
      var minSpan = bboxR * 0.03;  // 隣接なし時の最小表示範囲
      var spanW = Math.max(halfW * 2, minSpan);
      var spanH = Math.max(halfH * 2, minSpan);
      var newRatio = Math.max(
        spanW * S / (0.8 * 2 * bboxR * dim.width),
        spanH * S / (0.8 * 2 * bboxR * dim.height)
      );
      newRatio = Math.max(0.03, Math.min(newRatio, 2));

      console.log('[panToNode] center=(%f,%f) half=(%f,%f) framed=(%f,%f) ratio=%f',
        nx, ny, halfW, halfH, framed.x, framed.y, newRatio);

      camera.animate({ x: framed.x, y: framed.y, ratio: newRatio },
                     { duration: 400, easing: 'quadraticInOut' });
    } catch(e) { console.error('[panToNode] error:', e); }
  }

  // ── 詳細パネル表示 ────────────────────────────────────
  // ── グラフクリックとサイドバーを同期 ─────────────────
  renderer.on('clickNode', function(ev) {
    if (hasDragged) return;
    if (ev.node) {
      // 主ハンドラー実行後、selectedNode が更新されてから動く
      setTimeout(function() {
        if (selectedNode === ev.node) {
          sbActiveId = ev.node;
          showDetail(ev.node);
        } else {
          sbActiveId = null;
          _setDetailActive(false);
        }
        renderList();
      }, 0);
    }
  });
  renderer.on('clickStage', function() {
    sbActiveId = null;
    _setDetailActive(false);
    renderList();
  });

  // 初期ソート表示（引用数降順）
  document.querySelector('#sb-list-head-table th[data-col="citations"]').className = 'sort-desc';
  renderList();

  // コレクションフィルタから呼び出せるように公開
  window._renderList = renderList;

  // ── サイドバー幅リサイズ ──────────────────────────────────────────────────
  (function() {
    var hnd = document.getElementById('sb-resize-x');
    var sb  = document.getElementById('sidebar');
    var MIN_W = 280, MAX_W = 700;
    hnd.addEventListener('mousedown', function(e) {
      e.preventDefault();
      hnd.classList.add('dragging');
      var startX = e.clientX;
      var startW = sb.offsetWidth;
      function onMove(e) {
        var w = Math.max(MIN_W, Math.min(MAX_W, startW - (e.clientX - startX)));
        document.documentElement.style.setProperty('--sb-width', w + 'px');
      }
      function onUp() {
        hnd.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup',   onUp);
        // ドラッグ完了後にキャンバスサイズを更新して正しい領域に再描画する
        if (window.__r) { window.__r.resize(); window.__r.refresh(); }
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
    });
  })();

  // ── 詳細パネル高さリサイズ ───────────────────────────────────────────────
  (function() {
    var hnd   = detailResizeHnd;
    var MIN_H = 80, MAX_H = 600;
    hnd.addEventListener('mousedown', function(e) {
      e.preventDefault();
      hnd.classList.add('dragging');
      var startY = e.clientY;
      var startH = detail.offsetHeight;
      function onMove(e) {
        var h = Math.max(MIN_H, Math.min(MAX_H, startH - (e.clientY - startY)));
        detail.style.height = h + 'px';
      }
      function onUp() {
        hnd.classList.remove('dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup',   onUp);
      }
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup',   onUp);
    });
  })();
})();

console.log('[RAG] sigma ready – nodes:', graph.order, 'edges:', graph.size,
            '· D3-force layout starting…');

  }) // end fetch .then(GRAPH_DATA)
  .catch(function(e) {
    var el = document.getElementById('loading');
    if (el) el.innerHTML =
      '<div style="color:#f87171;font-size:13px">グラフ読み込みエラー: ' + esc(e.message) + '</div>';
    console.error('[RAG] fetch /api/graph failed:', e);
  });

})();