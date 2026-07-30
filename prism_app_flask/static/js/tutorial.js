// Tutorial — each sidebar menu item is a fully separate panel (not a scroll-to-anchor within one long
// page). Only the active .tut-post is shown; the URL hash tracks it so a section is directly linkable
// and the browser back/forward buttons work. Math (KaTeX) is rendered once across the whole hidden+visible
// DOM on load, since toggling is just a CSS display swap and all panels already exist.
'use strict';
(function(){
  function boot(){
    const toc = document.getElementById('tutToc');
    const posts = Array.from(document.querySelectorAll('.tut-post'));
    if(!toc || !posts.length) return;
    const items = Array.from(toc.querySelectorAll('.tut-toc-item'));
    const ids = posts.map(p => p.id);

    function activate(id, opts){
      opts = opts || {};
      if(!ids.includes(id)) id = ids[0];
      posts.forEach(p => p.classList.toggle('active', p.id === id));
      items.forEach(a => a.classList.toggle('on', a.dataset.id === id));
      if(opts.pushHash !== false && location.hash !== '#' + id){
        history.replaceState(null, '', '#' + id);
      }
      if(opts.scrollTop !== false){
        window.scrollTo({ top: 0, behavior: 'auto' });
      }
    }

    // in-body anchor (e.g. a §-quicknav link like #principles-problem inside a post's own
    // "On this page" box) → which .tut-post panel actually contains it, if any. Needed because
    // those sub-anchor ids are NOT in `ids` (which only lists top-level section ids).
    function postContaining(anchorId){
      const el = document.getElementById(anchorId);
      const post = el && el.closest('.tut-post');
      return post ? post.id : null;
    }

    items.forEach(a => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        activate(a.dataset.id);
      });
    });

    window.addEventListener('hashchange', () => {
      const id = location.hash.replace('#', '');
      if(ids.includes(id)){
        activate(id, { pushHash: false });
        return;
      }
      // sub-anchor inside a post body (quicknav link) — its target lives inside whichever
      // panel contains it. Previously this always fell back to ids[0] (the first section),
      // which made in-page "On this page" links look like they reset to the Tutorial home
      // screen instead of jumping to the clicked heading. Only touch panel visibility if the
      // containing panel isn't already the active one (e.g. a shared deep link); otherwise the
      // panel is already visible and the browser's native anchor-scroll just works.
      const containingId = postContaining(id);
      if(containingId){
        const post = posts.find(p => p.id === containingId);
        if(post && !post.classList.contains('active')){
          activate(containingId, { pushHash: false, scrollTop: false });
        }
        const el = document.getElementById(id);
        if(el) requestAnimationFrame(() => el.scrollIntoView({ block: 'start' }));
      }
    });

    const initialHash = location.hash.replace('#', '');
    const initial = ids.includes(initialHash) ? initialHash : (postContaining(initialHash) || ids[0]);
    activate(initial, { scrollTop: false, pushHash: false });
    if(initialHash && !ids.includes(initialHash) && document.getElementById(initialHash)){
      requestAnimationFrame(() => document.getElementById(initialHash).scrollIntoView({ block: 'start' }));
    }

    if(window.renderMathInElement){
      renderMathInElement(document.querySelector('.tut-posts'), {
        delimiters: [
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
        ],
        throwOnError: false
      });
    }
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
