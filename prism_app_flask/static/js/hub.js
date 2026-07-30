// Analysis Hub — 카트에서 Hub로 push된 gene/isoform 케이스를 메모와 함께 정리하는 개인 대시보드.
'use strict';
(function(){
  function esc(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
  let debounceTimers = {};

  function render(){
    const host = document.getElementById('hub-cart');
    if(!host) return;
    const items = Cart.load().items.filter(it => it.inHub);
    if(!items.length){
      host.innerHTML = `<p class="hint">No cases pushed to the Hub yet. Use the
        <span class="sig">🧺 Cart</span> button in the bottom-right to add genes/isoforms of interest and push them here.</p>`;
      return;
    }
    host.innerHTML = items.map(it => `
      <div class="hub-card" data-id="${esc(it.id)}">
        <div class="hub-card-head">
          <span class="cart-kind-badge ${it.kind}">${it.kind === 'isoform' ? 'I' : 'G'}</span>
          <a class="hub-card-id" href="/gene/${encodeURIComponent(it.id)}">${esc(it.id)}</a>${geneCardsLink(it.gene)}
          <button class="hub-card-rm" data-id="${esc(it.id)}" type="button">Remove from Hub</button>
        </div>
        <div class="hub-card-reason-row">
          <span class="hub-card-reason-k">WHY</span>
          <input class="hub-card-reason" data-id="${esc(it.id)}" type="text"
            placeholder="why did you add this? (optional)" value="${esc(it.reason || '')}">
        </div>
        <textarea class="hub-card-memo" data-id="${esc(it.id)}"
          placeholder="Notes / observations — write freely…">${esc(it.memo || '')}</textarea>
      </div>`).join('');
  }

  function wire(){
    const host = document.getElementById('hub-cart');
    if(!host) return;
    host.addEventListener('click', (e) => {
      const rm = e.target.closest('.hub-card-rm');
      if(rm) Cart.removeFromHub(rm.dataset.id);
    });
    host.addEventListener('input', (e) => {
      const memo = e.target.closest('.hub-card-memo');
      const reason = e.target.closest('.hub-card-reason');
      const field = memo || reason;
      if(!field) return;
      const id = field.dataset.id;
      const key = id + (memo ? ':memo' : ':reason');
      clearTimeout(debounceTimers[key]);
      debounceTimers[key] = setTimeout(() => {
        if(memo) Cart.setMemo(id, field.value); else Cart.setReason(id, field.value);
      }, 400);
    });
    Cart.onChange(() => {
      // 메모/reason 입력 중 리렌더로 커서 위치가 튀지 않도록, 포커스가 해당 필드에 있지 않을 때만 갱신
      const active = document.activeElement;
      if(active && (active.classList.contains('hub-card-memo') || active.classList.contains('hub-card-reason'))) return;
      render();
    });
  }

  function boot(){
    // cart.js 가 base.html 에서 먼저 로드되므로 window.Cart 는 이미 존재
    wire();
    render();
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
