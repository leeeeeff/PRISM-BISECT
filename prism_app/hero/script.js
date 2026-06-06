/* ── Config ─────────────────────────────────────────────────────────────────── */
const STREAMLIT = `http://${window.location.hostname}:8501`;

/* ── GSAP setup ─────────────────────────────────────────────────────────────── */
gsap.registerPlugin(ScrollTrigger);

/* ── Utility: reveal one or more elements as scroll progresses ───────────────── *
 *
 *  Creates a pinned section where elements appear sequentially as the user
 *  scrolls. `scrub: 1.5` ties the animation tightly to scroll position —
 *  scroll back up and the animation reverses. This is the Apple scroll-scrub feel.
 *
 *  travel: how many viewport-heights of scroll travel to spend on this scene.
 *          e.g. 180 = 1.8× the viewport height.
 */
function pinReveal(trigger, items, { travel = 180, scrub = 1.5, stagger = 0.35 } = {}) {
  const tl = gsap.timeline({
    scrollTrigger: {
      trigger,
      start: 'top top',
      end: `+=${travel}%`,
      pin: true,
      pinSpacing: true,
      scrub,
    },
  });

  items.forEach((el, i) => {
    tl.to(el,
      { opacity: 1, y: 0, ease: 'power2.out', duration: 0.6 },
      i * stagger           // offset each element's start time in the timeline
    );
  });

  return tl;
}

/* ── Scene 2: One gene. Many proteins. ──────────────────────────────────────── */
pinReveal('#s2', [
  '#s2-l1',
  '#s2-l2',
  '#s2-sub',
].map(s => document.querySelector(s)), { travel: 200 });

/* ── Scene 3: Each isoform carries a different function. ─────────────────────── */
pinReveal('#s3', [
  '#s3-l1',
  '#s3-l2',
].map(s => document.querySelector(s)), { travel: 160 });

/* ── Scene 4: Under disease. ─────────────────────────────────────────────────── */
pinReveal('#s4', [
  '#s4-l1',
  '#s4-l2',
  '#s4-sub',
].map(s => document.querySelector(s)), { travel: 200 });

/* ── Scene 5: No existing tool could see the difference. ────────────────────── */
pinReveal('#s5', [
  '#s5-l1',
  '#s5-l2',
].map(s => document.querySelector(s)), { travel: 160 });

/* ── Background: dark → light as scene 6 approaches ─────────────────────────── *
 *
 *  Animates body background from black to off-white. The visual moment
 *  right before "PRISM can." — contrast makes the reveal land harder.
 */
gsap.to('body', {
  backgroundColor: '#f5f5f7',
  ease: 'none',
  scrollTrigger: {
    trigger: '#s6',
    start: 'top 85%',
    end: 'top 15%',
    scrub: true,
  },
});

/* ── Scene 6: PRISM can. ─────────────────────────────────────────────────────── */
pinReveal('#s6', [
  '#s6-main',
  '#s6-sub',
].map(s => document.querySelector(s)), {
  travel: 200,
  scrub: 1.2,
  stagger: 0.5,
});

/* ── Scene 7: CTA fade-in ───────────────────────────────────────────────────── *
 *
 *  Not pinned — just a simple entrance as the section scrolls into view.
 *  toggleActions: plays forward on enter, reverses on scroll-up.
 */
gsap.to('.cta-headline', {
  opacity: 1, y: 0, ease: 'power2.out', duration: 0.8,
  scrollTrigger: {
    trigger: '#s7',
    start: 'top 72%',
    toggleActions: 'play none none reverse',
  },
});

gsap.to('.cta-buttons', {
  opacity: 1, y: 0, ease: 'power2.out', duration: 0.7,
  scrollTrigger: {
    trigger: '#s7',
    start: 'top 60%',
    toggleActions: 'play none none reverse',
  },
});

gsap.to('.cta-caption', {
  opacity: 1, ease: 'power1.out', duration: 0.6,
  scrollTrigger: {
    trigger: '#s7',
    start: 'top 45%',
    toggleActions: 'play none none reverse',
  },
});

/* ── Scroll hint: fade out on first scroll ───────────────────────────────────── */
gsap.to('.scroll-hint', {
  opacity: 0,
  ease: 'none',
  scrollTrigger: {
    trigger: 'body',
    start: '40px top',
    end: '120px top',
    scrub: true,
  },
});

/* ── CTA: navigate to Streamlit ─────────────────────────────────────────────── */
document.getElementById('btn-demo').addEventListener('click', () => {
  window.location.href = `${STREAMLIT}/?mode=demo&autostart=1`;
});

document.getElementById('btn-upload').addEventListener('click', () => {
  window.location.href = `${STREAMLIT}/?mode=upload`;
});
