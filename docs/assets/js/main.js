(function () {
  "use strict";

  /* Mobile nav */
  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.querySelector(".nav-links");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      const expanded = navToggle.getAttribute("aria-expanded") === "true";
      navToggle.setAttribute("aria-expanded", String(!expanded));
      navLinks.classList.toggle("open");
    });

    navLinks.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navToggle.setAttribute("aria-expanded", "false");
        navLinks.classList.remove("open");
      });
    });
  }

  /* Nav background on scroll + scroll spy */
  const nav = document.querySelector(".site-nav");
  const sections = document.querySelectorAll("section[id]");
  const navAnchors = document.querySelectorAll(".nav-links a");

  function onScroll() {
    if (nav) {
      nav.classList.toggle("scrolled", window.scrollY > 20);
    }

    let current = "";
    sections.forEach((section) => {
      const sectionTop = section.offsetTop;
      if (window.scrollY >= sectionTop - 120) {
        current = section.getAttribute("id");
      }
    });

    navAnchors.forEach((anchor) => {
      anchor.classList.toggle("active", anchor.getAttribute("href") === `#${current}`);
    });
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* Before/after sliders */
  function initSliders() {
    const sliders = document.querySelectorAll(".ba-slider");

    sliders.forEach((slider) => {
      const handle = slider.querySelector(".ba-handle");
      if (!handle) return;

      // Add labels
      const labels = document.createElement("div");
      labels.className = "ba-labels";
      labels.innerHTML = '<span class="ba-label before">Input</span><span class="ba-label after">DeLux</span>';
      slider.appendChild(labels);

      let isDragging = false;

      function getPct() {
        return parseFloat(slider.style.getPropertyValue("--pct")) || 50;
      }

      function setPct(pct) {
        pct = Math.max(2, Math.min(98, pct));
        slider.style.setProperty("--pct", `${pct}%`);
      }

      function updateFromClientX(clientX) {
        const rect = slider.getBoundingClientRect();
        let x = clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));
        setPct((x / rect.width) * 100);
      }

      function start(e) {
        isDragging = true;
        e.preventDefault();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        updateFromClientX(clientX);
      }

      function move(e) {
        if (!isDragging) return;
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        updateFromClientX(clientX);
      }

      function end() {
        isDragging = false;
      }

      slider.addEventListener("mousedown", start);
      slider.addEventListener("touchstart", start, { passive: false });

      window.addEventListener("mousemove", move);
      window.addEventListener("touchmove", move, { passive: false });

      window.addEventListener("mouseup", end);
      window.addEventListener("touchend", end);

      // Keyboard support
      handle.addEventListener("keydown", (e) => {
        const step = 3;
        const current = getPct();
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          setPct(current - step);
        } else if (e.key === "ArrowRight") {
          e.preventDefault();
          setPct(current + step);
        } else if (e.key === "Home") {
          e.preventDefault();
          setPct(2);
        } else if (e.key === "End") {
          e.preventDefault();
          setPct(98);
        }
      });
    });
  }

  initSliders();

  /* Citation copy */
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const target = document.querySelector(btn.getAttribute("data-copy"));
      if (!target) return;
      const text = target.textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
      } catch (err) {
        // Fallback
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      const original = btn.innerHTML;
      btn.classList.add("copied");
      btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
      setTimeout(() => {
        btn.classList.remove("copied");
        btn.innerHTML = original;
      }, 2000);
    });
  });
})();
