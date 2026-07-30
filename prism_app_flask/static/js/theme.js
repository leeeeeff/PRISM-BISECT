// 흑/백(dark/light) 테마 토글. base.html 이 모든 페이지에서 로드.
// 크롬(CSS)은 토큰 재정의로 즉시 전환되고, Plotly 차트는 refreshTheme() 로 T 를 갱신한 뒤
// 각 페이지가 등록한 window.__rerenderCharts() 로 현재 화면만 다시 그린다(리로드 없음).
'use strict';
(function(){
  function current(){ return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'; }
  function apply(theme){
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('prism-theme', theme); } catch(e) {}
    // Plotly 페이지면 T 를 새 테마로 갱신하고 현재 차트만 다시 그린다.
    if (typeof window.refreshTheme === 'function') window.refreshTheme();
    if (typeof window.__rerenderCharts === 'function') window.__rerenderCharts();
  }
  function boot(){
    const btn = document.getElementById('theme-toggle');
    if(!btn) return;
    btn.addEventListener('click', () => apply(current() === 'light' ? 'dark' : 'light'));
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
