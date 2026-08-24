(() => {
  // ns-hugo-imp:/Users/hslu-n0006897/projects/gstechschulte.github.io/themes/hugo-omegion/assets/js/theme.js
  function applyFaviconTheme(theme) {
    document.querySelectorAll("link[data-icon-theme]").forEach((link) => {
      link.media = link.dataset.iconTheme === theme ? "" : "not all";
    });
  }
  function initTheme() {
    applyFaviconTheme(document.documentElement.getAttribute("data-theme"));
    const toggle = document.getElementById("theme-toggle");
    if (!toggle) return;
    toggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      applyFaviconTheme(next);
      try {
        localStorage.setItem("theme", next);
      } catch (e) {
      }
    });
  }

  // ns-hugo-imp:/Users/hslu-n0006897/projects/gstechschulte.github.io/themes/hugo-omegion/assets/js/search.js
  function initSearch() {
    const openBtn = document.getElementById("search-open");
    const closeBtn = document.getElementById("search-close");
    const modal = document.getElementById("search-modal");
    const backdrop = document.getElementById("search-backdrop");
    const input = document.getElementById("search-input");
    const results = document.getElementById("search-results");
    if (!openBtn || !modal || !input || !results) return;
    const recentPostsCount = parseInt(input.dataset.recentPosts, 10) || 3;
    const recentProjectsCount = parseInt(input.dataset.recentProjects, 10) || 3;
    let index = null;
    function loadIndex() {
      if (index) return Promise.resolve(index);
      return fetch(input.dataset.indexUrl || "/index.json").then((res) => res.json()).then((data) => {
        index = data;
        return index;
      }).catch(() => {
        index = [];
        return index;
      });
    }
    function escapeHtml(str) {
      return String(str).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      })[c]);
    }
    function renderRow(item) {
      return `<a class="link-row" href="${item.url}"><span class="link-row-title">${escapeHtml(item.title)}</span><span class="link-row-meta">${escapeHtml(item.date)}</span></a>`;
    }
    function render(items, label) {
      if (!items.length) {
        results.innerHTML = '<p class="search-empty">No results.</p>';
        return;
      }
      const rows = items.slice(0, 20).map(renderRow).join("");
      results.innerHTML = (label ? `<div class="search-label">${escapeHtml(label)}</div>` : "") + rows;
    }
    function renderGroups(groups) {
      const nonEmpty = groups.filter((g) => g.items.length);
      if (!nonEmpty.length) {
        results.innerHTML = '<p class="search-empty">No results.</p>';
        return;
      }
      results.innerHTML = nonEmpty.map((g) => `<div class="search-label">${escapeHtml(g.label)}</div>` + g.items.map(renderRow).join("")).join("");
    }
    function renderSuggestions() {
      const posts = (index || []).filter((item) => item.section === "posts").slice(0, recentPostsCount);
      const projects = (index || []).filter((item) => item.section === "projects").slice(0, recentProjectsCount);
      renderGroups([
        { label: "Recent posts", items: posts },
        { label: "Recent projects", items: projects }
      ]);
    }
    function search(query) {
      const q = query.trim().toLowerCase();
      if (!q) {
        renderSuggestions();
        return;
      }
      const matches = (index || []).filter((item) => {
        const haystack = [item.title, item.summary, (item.tags || []).join(" ")].join(" ").toLowerCase();
        return haystack.includes(q);
      });
      render(matches);
    }
    const panel = modal.querySelector(".search-panel");
    function open() {
      modal.hidden = false;
      document.body.classList.add("no-scroll");
      input.value = "";
      results.innerHTML = "";
      loadIndex().then(renderSuggestions);
      requestAnimationFrame(() => {
        modal.classList.add("is-open");
        input.focus();
      });
    }
    function close() {
      if (!modal.classList.contains("is-open")) return;
      modal.classList.remove("is-open");
      document.body.classList.remove("no-scroll");
      panel.addEventListener(
        "transitionend",
        () => {
          modal.hidden = true;
        },
        { once: true }
      );
    }
    openBtn.addEventListener("click", open);
    closeBtn.addEventListener("click", close);
    backdrop.addEventListener("click", close);
    input.addEventListener("input", (e) => search(e.target.value));
    document.addEventListener("keydown", (e) => {
      if (e.key === "/" && document.activeElement !== input && !modal.contains(document.activeElement)) {
        e.preventDefault();
        open();
      } else if (e.key === "Escape" && !modal.hidden) {
        close();
      }
    });
  }

  // ns-hugo-imp:/Users/hslu-n0006897/projects/gstechschulte.github.io/themes/hugo-omegion/assets/js/toc.js
  function initToc() {
    const toc = document.getElementById("toc");
    if (!toc) return;
    function syncTocTop() {
      const prose = document.querySelector(".prose");
      if (!prose) return;
      toc.style.setProperty("--toc-top", prose.offsetTop + "px");
    }
    syncTocTop();
    window.addEventListener("resize", syncTocTop);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(syncTocTop);
    }
    const toggle = document.getElementById("toc-toggle");
    const links = Array.from(toc.querySelectorAll(".toc-link"));
    const targets = links.map((link) => {
      const id = decodeURIComponent(link.getAttribute("href").slice(1));
      return { link, el: document.getElementById(id) };
    }).filter((t) => t.el);
    if (toggle) {
      let closeDrawer = function() {
        toc.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      }, openDrawer = function() {
        toc.classList.add("is-open");
        toggle.setAttribute("aria-expanded", "true");
      };
      toggle.addEventListener("click", (e) => {
        e.stopPropagation();
        if (toc.classList.contains("is-open")) closeDrawer();
        else openDrawer();
      });
      links.forEach((link) => link.addEventListener("click", closeDrawer));
      document.addEventListener("click", (e) => {
        if (!toc.classList.contains("is-open")) return;
        if (toc.contains(e.target) || toggle.contains(e.target)) return;
        closeDrawer();
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeDrawer();
      });
    }
    const PIN_THRESHOLD = 320;
    function updatePinned() {
      toc.classList.toggle("toc-pinned", window.scrollY > PIN_THRESHOLD);
    }
    if (!targets.length) {
      updatePinned();
      window.addEventListener("scroll", updatePinned, { passive: true });
      return;
    }
    const OFFSET = 96;
    let ticking = false;
    function setActive(link) {
      links.forEach((l) => l.classList.toggle("toc-active", l === link));
    }
    function update() {
      ticking = false;
      updatePinned();
      let current = targets[0];
      for (const target of targets) {
        if (target.el.getBoundingClientRect().top - OFFSET <= 0) {
          current = target;
        } else {
          break;
        }
      }
      setActive(current.link);
    }
    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
  }

  // ns-hugo-imp:/Users/hslu-n0006897/projects/gstechschulte.github.io/themes/hugo-omegion/assets/js/sidebar.js
  function initLogoIntro() {
    const logos = document.querySelectorAll(".logo-letters");
    logos.forEach((logo) => {
      const spans = logo.querySelectorAll("span");
      if (!spans.length) return;
      logo.classList.add("intro");
      const totalDuration = (spans.length - 1) * 70 + 1100;
      setTimeout(() => logo.classList.remove("intro"), totalDuration + 50);
    });
  }
  function initSidebar() {
    const sidebar = document.getElementById("sidebar");
    const toggle = document.getElementById("sidebar-toggle");
    const backdrop = document.getElementById("sidebar-backdrop");
    if (!sidebar || !toggle || !backdrop) return;
    function close() {
      sidebar.classList.remove("is-open");
      backdrop.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
      document.body.classList.remove("no-scroll");
    }
    function open() {
      sidebar.classList.add("is-open");
      backdrop.hidden = false;
      toggle.setAttribute("aria-expanded", "true");
      document.body.classList.add("no-scroll");
    }
    toggle.addEventListener("click", () => {
      if (sidebar.classList.contains("is-open")) close();
      else open();
    });
    backdrop.addEventListener("click", close);
    sidebar.querySelectorAll("a").forEach((link) => link.addEventListener("click", close));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") close();
    });
    window.addEventListener("resize", () => {
      if (window.innerWidth >= 900) close();
    });
  }

  // ns-hugo-imp:/Users/hslu-n0006897/projects/gstechschulte.github.io/themes/hugo-omegion/assets/js/codeblock.js
  function initCodeCopy() {
    const buttons = document.querySelectorAll(".code-copy");
    if (!buttons.length) return;
    buttons.forEach((button) => {
      const label = button.querySelector(".code-copy-label");
      const defaultLabel = label ? label.textContent : "";
      let resetTimer;
      button.addEventListener("click", async () => {
        const pre = button.closest(".code-block")?.querySelector("pre");
        if (!pre) return;
        try {
          await navigator.clipboard.writeText(pre.textContent.replace(/\n$/, ""));
        } catch {
          return;
        }
        button.classList.add("is-copied");
        if (label) label.textContent = "Copied";
        clearTimeout(resetTimer);
        resetTimer = setTimeout(() => {
          button.classList.remove("is-copied");
          if (label) label.textContent = defaultLabel;
        }, 1500);
      });
    });
  }

  // <stdin>
  function initResizeGuard() {
    const root = document.documentElement;
    let resizeTimer;
    window.addEventListener("resize", () => {
      root.classList.add("is-resizing");
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => root.classList.remove("is-resizing"), 150);
    });
  }
  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initSearch();
    initToc();
    initSidebar();
    initLogoIntro();
    initResizeGuard();
    initCodeCopy();
  });
})();
