// Client-side theme switcher engine for Bus Headways Dashboard
(function () {
  function getStoredTheme() {
    return localStorage.getItem("headways-theme") || "system";
  }

  function applyTheme(theme) {
    if (theme === "system") {
      const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");
    } else {
      document.documentElement.setAttribute("data-theme", theme);
    }
    updateButtons(theme);
  }

  function updateButtons(activeTheme) {
    document.querySelectorAll(".theme-btn").forEach((btn) => {
      const targetTheme = btn.getAttribute("data-set-theme");
      if (targetTheme === activeTheme) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  window.setDashboardTheme = function (theme) {
    localStorage.setItem("headways-theme", theme);
    applyTheme(theme);
  };

  // Listen for OS color scheme changes
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", function () {
      if (getStoredTheme() === "system") {
        applyTheme("system");
      }
    });

  // Apply immediately before DOM render to prevent flash
  const initialTheme = getStoredTheme();
  applyTheme(initialTheme);

  // Initialize button state when DOM is ready
  document.addEventListener("DOMContentLoaded", function () {
    updateButtons(initialTheme);
  });
})();

// Guide Modal Controls
window.openGuideModal = function () {
  const modal = document.getElementById("guide-modal-backdrop");
  if (modal) {
    modal.style.display = "flex";
    document.body.style.overflow = "hidden";
  }
};

window.closeGuideModal = function () {
  const modal = document.getElementById("guide-modal-backdrop");
  if (modal) {
    modal.style.display = "none";
    document.body.style.overflow = "";
  }
};

// Close on Escape key
document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") {
    window.closeGuideModal();
  }
});
