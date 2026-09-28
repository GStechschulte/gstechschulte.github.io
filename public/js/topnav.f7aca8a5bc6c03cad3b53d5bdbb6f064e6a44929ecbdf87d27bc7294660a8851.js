(() => {
  // <stdin>
  function initRailProxies() {
    document.querySelectorAll(".rail [data-proxy]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        document.getElementById(btn.dataset.proxy)?.click();
      });
    });
  }
  function initTopnavScroll() {
    const nav = document.getElementById("topnav");
    if (!nav) return;
  }
  function init() {
    initRailProxies();
    initTopnavScroll();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
