(() => {
  // <stdin>
  function initTopnavScroll() {
    const nav = document.getElementById("topnav");
    if (!nav) return;
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initTopnavScroll);
  else initTopnavScroll();
})();
