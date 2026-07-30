// GeneCards search link — auto-generated wherever a gene symbol (or an isoform, via its parent
// gene — GeneCards has no isoform-level pages) is displayed. Loaded globally (base.html) since
// gene/isoform text shows up on every analysis page plus the floating cart widget.
'use strict';
function geneCardsUrl(gene){
  return 'https://www.genecards.org/cgi-bin/carddisp.pl?gene=' + encodeURIComponent(gene);
}
// Small inline badge link, opens in a new tab. `event.stopPropagation()` matters wherever this
// sits inside a row/card that itself has an onclick navigation handler (common in this app's
// tables) — otherwise clicking the badge would also fire the row's internal navigation.
function geneCardsLink(gene, cls){
  if(!gene) return '';
  return `<a class="gc-link${cls?(' '+cls):''}" href="${geneCardsUrl(gene)}" target="_blank" rel="noopener" `+
    `title="Look up ${gene} on GeneCards" onclick="event.stopPropagation()">GC↗</a>`;
}
