(() => {
  const PRESETS = {
    plain:
      "linear-gradient(180deg, #3e424a 0%, #363a42 38%, #2f3238 72%, #282b30 100%)",
    graphite: "#363a42",
    aurora:
      "radial-gradient(900px 520px at 18% -5%, rgba(255,255,255,0.07) 0%, transparent 58%), linear-gradient(180deg, #3c4048 0%, #363a42 40%, #2e3138 100%)",
    midnight: "linear-gradient(180deg, #3a3e46 0%, #363a42 35%, #2f3238 70%, #282b30 100%)",
    ocean:
      "radial-gradient(ellipse 85% 55% at 50% -18%, rgba(255, 255, 255, 0.06) 0%, transparent 52%), linear-gradient(180deg, #3e424a 0%, #32363d 55%, #2a2d32 100%)",
    ember:
      "radial-gradient(800px 480px at 88% 8%, rgba(255,255,255,0.05) 0%, transparent 50%), linear-gradient(180deg, #383c44 0%, #30343a 50%, #282b30 100%)",
    wallpaper: null,
    custom: null,
  };

  function wallpaperBackgroundCss(rel) {
    const u = window.AppPrefs.catalogDownloadHref(rel);
    return `linear-gradient(rgba(14,15,18,0.52), rgba(14,15,18,0.52)), url("${u}") center / cover no-repeat fixed`;
  }

  function resolveBackground() {
    const { preset, custom, wallpaperRel } = window.AppPrefs.getUiBackground();
    const rel = (wallpaperRel || "").trim();
    if (rel.startsWith("wallpaper/") && !rel.includes("..")) {
      return wallpaperBackgroundCss(rel);
    }

    const key = (preset || "plain").toLowerCase();

    if (key === "custom") {
      const c = (custom || "").trim();
      if (c) return c;
    }

    const built = PRESETS[key];
    if (built) return built;
    return PRESETS.plain;
  }

  function apply() {
    document.documentElement.style.setProperty("--app-body-bg", resolveBackground());
  }

  window.AppTheme = { apply, PRESETS };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  } else {
    apply();
  }

  window.addEventListener("storage", (ev) => {
    if (!window.AppPrefs) return;
    const k = ev.key;
    if (
      k === AppPrefs.keys.UI_BG_PRESET ||
      k === AppPrefs.keys.UI_BG_CUSTOM ||
      k === AppPrefs.keys.UI_WALLPAPER_REL
    ) {
      apply();
    }
  });
})();
