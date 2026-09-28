// Top navigation bar behaviour. Loaded after the theme's main.js, which already
// wires #theme-toggle, #search-open and the #sidebar overflow menu.

// Scroll behaviour. The header is `position: sticky`; toggling the
// `topnav-hidden` class on it slides it out of view (CSS handles the animation).
//
// TODO(you): decide how the bar should react to scrolling. Options:
//   a) Do nothing: always visible (simplest, costs ~3.5rem of height permanently).
//   b) "Headroom": hide when scrolling down, reveal when scrolling up.
//   c) Hide only once the reader is past the article header (e.g. scrollY > 320,
//      the same threshold toc.js uses to pin the ToC).
function initTopnavScroll() {
  const nav = document.getElementById("topnav");
  if (!nav) return;
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initTopnavScroll);
else initTopnavScroll();
