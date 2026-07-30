// Gene/Isoform 카트 — localStorage 기반, DB/세션 없는 이 앱의 유일한 영속화 패턴(theme.js와 동일)을 따른다.
// base.html 에서 모든 페이지에 로드 → window.Cart 로 어디서든 담기/빼기/메모/Hub-push 가능.
// 플로팅 버튼 + 드래그 가능한 미니 콘솔을 이 스크립트가 직접 DOM에 주입한다(템플릿 변경 불필요).
'use strict';
(function(){
  const KEY = 'prism-cart';
  const POS_KEY = 'prism-cart-pos';
  const listeners = [];

  function load(){
    try {
      const raw = localStorage.getItem(KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      return (parsed && Array.isArray(parsed.items)) ? parsed : { items: [] };
    } catch(e) { return { items: [] }; }
  }
  function save(state){
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch(e) {}
    listeners.forEach(fn => { try { fn(state); } catch(e) {} });
  }
  function onChange(fn){ listeners.push(fn); }

  function find(state, id){ return state.items.find(it => it.id === id); }
  function has(id){ return !!find(load(), id); }
  function isInHub(id){ const it = find(load(), id); return !!(it && it.inHub); }

  function add(id, kind, gene){
    id = (id || '').trim();
    if(!id) return;
    const state = load();
    if(find(state, id)) return;
    state.items.push({ id, kind: kind || 'gene', gene: gene || id, reason: '', memo: '', inHub: false, addedAt: Date.now() });
    save(state);
  }
  // Hub에 올라간 항목은 여기서 지우지 않는다 — Hub에서 빼려면 반드시 removeFromHub()를 거치게 해
  // "관심 리스트에서 삭제 = Hub에서도 사라짐" 사고를 원천 차단한다.
  function remove(id){
    const state = load();
    const it = find(state, id);
    if(it && it.inHub) return;
    state.items = state.items.filter(it => it.id !== id);
    save(state);
  }
  function setReason(id, reason){
    const state = load();
    const it = find(state, id);
    if(it){ it.reason = reason; save(state); }
  }
  function setMemo(id, memo){
    const state = load();
    const it = find(state, id);
    if(it){ it.memo = memo; save(state); }
  }
  function pushToHub(ids){
    const state = load();
    state.items.forEach(it => { if(ids.includes(it.id)) it.inHub = true; });
    save(state);
  }
  function removeFromHub(id){
    const state = load();
    const it = find(state, id);
    if(it){ it.inHub = false; save(state); }
  }

  window.Cart = { load, has, isInHub, add, remove, setReason, setMemo, pushToHub, removeFromHub, onChange };

  // ── ID kind 추측 — isoform_id 는 "SYMBOL-<번호>" 꼴(예: A1BG-204) ──
  function guessKind(id){ return /-\d+$/.test(id.trim()) ? 'isoform' : 'gene'; }
  function guessGene(id){ return id.replace(/-\d+$/, ''); }

  // ── UI 주입 ──
  let fabEl, consoleEl, listEl, hubListEl, badgeEl, kindSel;

  function esc(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  // fab 배지는 "아직 push 안 한, 체크 대기중" 항목 수만 센다 — Hub로 넘어간 건 더 이상 액션 대기가 아니다.
  function renderBadge(){
    const n = load().items.filter(it => !it.inHub).length;
    badgeEl.textContent = n;
    badgeEl.style.display = n > 0 ? '' : 'none';
  }

  function renderConsole(){
    // reason 입력 중(포커스 상태)엔 리렌더로 커서가 튀지 않도록 건너뛴다 — badge만 갱신.
    if(document.activeElement && document.activeElement.classList.contains('cart-item-reason')){
      renderBadge();
      return;
    }
    const state = load();
    // 대기 리스트 = 아직 Hub로 push 되지 않은 항목만. push된 항목은 여기서 사라지고 "IN HUB" 목록에만 남는다.
    const pending = state.items.filter(it => !it.inHub);
    listEl.innerHTML = pending.length ? pending.map(it => `
      <div class="cart-item-row" data-id="${esc(it.id)}">
        <div class="cart-item-top">
          <label class="cart-item-main">
            <input type="checkbox" class="cart-push-chk" data-id="${esc(it.id)}">
            <span class="cart-kind-badge ${it.kind}">${it.kind === 'isoform' ? 'I' : 'G'}</span>
            <span class="cart-item-id">${esc(it.id)}</span>
          </label>
          ${geneCardsLink(it.gene)}
          <button class="cart-item-rm" data-id="${esc(it.id)}" title="Remove from cart" type="button">×</button>
        </div>
        <input type="text" class="cart-item-reason" data-id="${esc(it.id)}"
          placeholder="why did you add this? (optional)" value="${esc(it.reason || '')}">
      </div>`).join('') : '<p class="muted mono sm cart-empty">No items yet.</p>';

    const hubItems = state.items.filter(it => it.inHub);
    hubListEl.innerHTML = hubItems.length ? hubItems.map(it => {
      const preview = it.reason || it.memo || '';
      return `
      <div class="cart-hub-preview">
        <span class="cart-kind-badge ${it.kind}">${it.kind === 'isoform' ? 'I' : 'G'}</span>
        <span class="cart-item-id">${esc(it.id)}</span>${geneCardsLink(it.gene)}
        <span class="muted sm">${esc(preview.slice(0, 40))}${preview.length > 40 ? '…' : ''}</span>
      </div>`;
    }).join('') : '<p class="muted mono sm cart-empty">No items pushed to Hub yet.</p>';

    renderBadge();
  }

  function buildDom(){
    fabEl = document.createElement('button');
    fabEl.type = 'button';
    fabEl.className = 'cart-fab';
    fabEl.title = 'Gene/Isoform Cart';
    fabEl.innerHTML = '🧺<span class="cart-fab-badge" id="cartFabBadge">0</span>';
    document.body.appendChild(fabEl);
    badgeEl = fabEl.querySelector('#cartFabBadge');

    consoleEl = document.createElement('div');
    consoleEl.className = 'cart-console hidden';
    consoleEl.innerHTML = `
      <div class="cart-console-header">
        <span class="tag-label sig">▸ CART</span>
        <button class="cart-console-close" type="button" title="Close">×</button>
      </div>
      <div class="cart-console-body">
        <form class="cart-add-form" id="cartAddForm">
          <input id="cartAddInput" placeholder="Gene symbol or isoform ID" autocomplete="off">
          <div class="seg cart-kind-seg" id="cartKindSeg">
            <button type="button" data-kind="auto" class="on">Auto</button>
            <button type="button" data-kind="gene">Gene</button>
            <button type="button" data-kind="isoform">Isoform</button>
          </div>
          <button class="btn primary" type="submit">+ Add</button>
        </form>
        <div class="cart-list" id="cartList"></div>
        <div class="cart-push-row">
          <label class="cart-select-all"><input type="checkbox" id="cartSelectAll"> select all</label>
          <button class="btn cart-push-btn" id="cartPushBtn" type="button">Push checked → Hub</button>
        </div>
        <div class="cart-divider"></div>
        <div class="tag-label" style="font-size:.6rem">▸ IN HUB</div>
        <div class="cart-hub-list" id="cartHubList"></div>
        <a class="cart-hub-link" href="/analysis">Open full Hub →</a>
      </div>`;
    document.body.appendChild(consoleEl);
    listEl = consoleEl.querySelector('#cartList');
    hubListEl = consoleEl.querySelector('#cartHubList');
    kindSel = consoleEl.querySelector('#cartKindSeg');

    // 저장된 위치 복원
    try {
      const pos = JSON.parse(localStorage.getItem(POS_KEY) || 'null');
      if(pos && typeof pos.left === 'number' && typeof pos.top === 'number'){
        consoleEl.style.left = pos.left + 'px';
        consoleEl.style.top = pos.top + 'px';
        consoleEl.style.right = 'auto';
        consoleEl.style.bottom = 'auto';
      }
    } catch(e) {}

    wireEvents();
    renderConsole();
  }

  function wireEvents(){
    fabEl.addEventListener('click', () => consoleEl.classList.toggle('hidden'));
    consoleEl.querySelector('.cart-console-close').addEventListener('click', () => consoleEl.classList.add('hidden'));

    kindSel.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-kind]');
      if(!btn) return;
      kindSel.querySelectorAll('button').forEach(b => b.classList.toggle('on', b === btn));
    });

    consoleEl.querySelector('#cartAddForm').addEventListener('submit', (e) => {
      e.preventDefault();
      const input = consoleEl.querySelector('#cartAddInput');
      const id = input.value.trim();
      if(!id) return;
      const kindBtn = kindSel.querySelector('button.on');
      const kind = kindBtn.dataset.kind === 'auto' ? guessKind(id) : kindBtn.dataset.kind;
      const gene = kind === 'isoform' ? guessGene(id) : id;
      add(id, kind, gene);
      input.value = '';
    });

    listEl.addEventListener('click', (e) => {
      const rm = e.target.closest('.cart-item-rm');
      if(rm) remove(rm.dataset.id);
    });

    let reasonTimers = {};
    listEl.addEventListener('input', (e) => {
      const inp = e.target.closest('.cart-item-reason');
      if(!inp) return;
      const id = inp.dataset.id;
      clearTimeout(reasonTimers[id]);
      reasonTimers[id] = setTimeout(() => setReason(id, inp.value), 400);
    });

    consoleEl.querySelector('#cartPushBtn').addEventListener('click', () => {
      const ids = Array.from(listEl.querySelectorAll('.cart-push-chk:checked')).map(c => c.dataset.id);
      if(ids.length) pushToHub(ids);
    });

    // one-shot toggle (not persisted/synced) — checking it checks every currently-listed pending
    // item so "push checked → Hub" can act on all of them without clicking each one first.
    consoleEl.querySelector('#cartSelectAll').addEventListener('change', (e) => {
      listEl.querySelectorAll('.cart-push-chk').forEach(c => { c.checked = e.target.checked; });
    });

    // ── 드래그 (헤더만) ──
    const header = consoleEl.querySelector('.cart-console-header');
    let dragging = false, dx = 0, dy = 0;
    header.addEventListener('mousedown', (e) => {
      if(e.target.closest('.cart-console-close')) return;
      dragging = true;
      const rect = consoleEl.getBoundingClientRect();
      dx = e.clientX - rect.left; dy = e.clientY - rect.top;
      consoleEl.style.right = 'auto'; consoleEl.style.bottom = 'auto';
      e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
      if(!dragging) return;
      const left = Math.max(0, Math.min(window.innerWidth - 60, e.clientX - dx));
      const top = Math.max(0, Math.min(window.innerHeight - 40, e.clientY - dy));
      consoleEl.style.left = left + 'px';
      consoleEl.style.top = top + 'px';
    });
    document.addEventListener('mouseup', () => {
      if(!dragging) return;
      dragging = false;
      try {
        const rect = consoleEl.getBoundingClientRect();
        localStorage.setItem(POS_KEY, JSON.stringify({ left: rect.left, top: rect.top }));
      } catch(e) {}
    });

    onChange(renderConsole);
  }

  function boot(){
    buildDom();
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
