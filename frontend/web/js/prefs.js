(() => {
  /** Strip trailing `/api` so fetch(`${origin}/api/...`) does not become `/api/api/...`. */
  function normalizeApiBase(s) {
    let x = String(s || "").trim().replace(/\/+$/, "");
    x = x.replace(/\/api\/?$/i, "");
    return x.replace(/\/+$/, "");
  }

  function defaultBackendOrigin() {
    try {
      if (location.protocol !== "http:" && location.protocol !== "https:") {
        return "http://127.0.0.1:8000";
      }
      const port = location.port || (location.protocol === "https:" ? "443" : "80");
      const staticDevPorts = new Set(["5500", "5501", "5173", "4173", "3000", "8080", "1234"]);
      if (staticDevPorts.has(port)) {
        return `${location.protocol}//${location.hostname}:8000`;
      }
      return location.origin;
    } catch {
      return "http://127.0.0.1:8000";
    }
  }

  const defaults = Object.freeze({
    deepseekApiKey: "",
    deepseekBaseUrl: "https://api.deepseek.com",
    deepseekModel: "deepseek-v4-pro",
  });

  window.AppPrefs = {
    keys: {
      BACKEND_ORIGIN: "agent.backend_origin",
      DS_KEY: "agent.deepseek_api_key",
      DS_BASE: "agent.deepseek_base_url",
      DS_MODEL: "agent.deepseek_model",
      UI_BG_PRESET: "agent.ui_bg_preset",
      UI_BG_CUSTOM: "agent.ui_bg_custom",
      UI_WALLPAPER_REL: "agent.ui_wallpaper_rel",
    },

    defaults,

    backendOrigin() {
      const saved = localStorage.getItem(this.keys.BACKEND_ORIGIN);
      const raw =
        saved && saved.trim() ? saved.trim() : defaultBackendOrigin();
      return normalizeApiBase(raw);
    },

    setBackendOrigin(v) {
      const t = normalizeApiBase((v || "").trim());
      if (t) localStorage.setItem(this.keys.BACKEND_ORIGIN, t);
      else localStorage.removeItem(this.keys.BACKEND_ORIGIN);
    },

    catalogDownloadHref(relativePath) {
      const q = encodeURIComponent(String(relativePath || "").trim());
      const p = `/api/catalog/download?path=${q}`;
      const base = this.backendOrigin();
      try {
        const u = new URL(base);
        if (u.origin === location.origin) return p;
        return `${u.origin}${p}`;
      } catch {
        return p;
      }
    },

    deepseek() {
      return {
        apiKey: localStorage.getItem(this.keys.DS_KEY) || "",
        baseUrl: localStorage.getItem(this.keys.DS_BASE) || defaults.deepseekBaseUrl,
        model: localStorage.getItem(this.keys.DS_MODEL) || defaults.deepseekModel,
      };
    },

    setDeepseek({ apiKey, baseUrl, model }) {
      if (typeof apiKey === "string") {
        localStorage.setItem(this.keys.DS_KEY, apiKey.trim());
      }
      if (typeof baseUrl === "string" && baseUrl.trim()) {
        localStorage.setItem(this.keys.DS_BASE, baseUrl.trim().replace(/\/+$/, ""));
      }
      if (typeof model === "string" && model.trim()) {
        localStorage.setItem(this.keys.DS_MODEL, model.trim());
      }
    },

    wsAgentURL() {
      try {
        const u = new URL(this.backendOrigin());
        u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
        return `${u.origin}/ws/agent`;
      } catch {
        const proto = location.protocol === "https:" ? "wss" : "ws";
        return `${proto}://${location.host}/ws/agent`;
      }
    },

    getUiBackground() {
      return {
        preset: localStorage.getItem(this.keys.UI_BG_PRESET) || "plain",
        custom: localStorage.getItem(this.keys.UI_BG_CUSTOM) || "",
        wallpaperRel: localStorage.getItem(this.keys.UI_WALLPAPER_REL) || "",
      };
    },

    setUiBackground({ preset, custom, wallpaperRel }) {
      if (typeof preset === "string" && preset.trim()) {
        localStorage.setItem(this.keys.UI_BG_PRESET, preset.trim());
      }
      if (custom !== undefined) {
        localStorage.setItem(this.keys.UI_BG_CUSTOM, String(custom || ""));
      }
      if (wallpaperRel !== undefined) {
        localStorage.setItem(this.keys.UI_WALLPAPER_REL, String(wallpaperRel || "").trim());
      }
    },
  };
})();
