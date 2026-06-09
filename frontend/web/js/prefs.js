(() => {
  const PREFIX = "souti.";

  const PROVIDERS = Object.freeze({
    deepseek: {
      label: "DeepSeek",
      baseUrl: "https://api.deepseek.com",
      model: "deepseek-v4-pro",
    },
    openai: {
      label: "OpenAI",
      baseUrl: "https://api.openai.com/v1",
      model: "gpt-4o",
    },
    anthropic: {
      label: "Anthropic 网关",
      baseUrl: "https://api.anthropic.com/v1",
      model: "claude-sonnet-4-20250514",
    },
    custom: {
      label: "自定义 / 本地",
      baseUrl: "http://127.0.0.1:11434/v1",
      model: "llama3",
    },
  });

  function key(name) {
    return PREFIX + name;
  }

  function defaultBackendOrigin() {
    try {
      if (location.protocol === "http:" || location.protocol === "https:") {
        return location.origin;
      }
    } catch (_) {}
    return "http://127.0.0.1:8010";
  }

  function loadProviderConfig(providerId) {
    const def = PROVIDERS[providerId] || PROVIDERS.deepseek;
    return {
      apiKey: localStorage.getItem(key(`key.${providerId}`)) || "",
      baseUrl: localStorage.getItem(key(`base.${providerId}`)) || def.baseUrl,
      model: localStorage.getItem(key(`model.${providerId}`)) || def.model,
    };
  }

  const prefs = {
    PROVIDERS,

    // 与 ciallo agent 共用 localStorage 键，壁纸/预设可跨应用同步
    keys: {
      BACKEND_ORIGIN: key("backend"),
      UI_BG_PRESET: "agent.ui_bg_preset",
      UI_BG_CUSTOM: "agent.ui_bg_custom",
      UI_WALLPAPER_REL: "agent.ui_wallpaper_rel",
    },

    backendOrigin() {
      return (
        localStorage.getItem(this.keys.BACKEND_ORIGIN) || defaultBackendOrigin()
      ).replace(/\/+$/, "");
    },

    setBackendOrigin(v) {
      const t = String(v || "").trim().replace(/\/+$/, "");
      if (t) localStorage.setItem(this.keys.BACKEND_ORIGIN, t);
      else localStorage.removeItem(this.keys.BACKEND_ORIGIN);
    },

    activeProvider() {
      return localStorage.getItem(key("provider")) || "deepseek";
    },

    setActiveProvider(id) {
      localStorage.setItem(key("provider"), id);
    },

    loadProviderConfig,
    loadActiveConfig() {
      return loadProviderConfig(this.activeProvider());
    },

    saveProviderConfig(providerId, { apiKey, baseUrl, model }) {
      if (typeof apiKey === "string") {
        localStorage.setItem(key(`key.${providerId}`), apiKey.trim());
      }
      if (typeof baseUrl === "string" && baseUrl.trim()) {
        localStorage.setItem(key(`base.${providerId}`), baseUrl.trim().replace(/\/+$/, ""));
      }
      if (typeof model === "string" && model.trim()) {
        localStorage.setItem(key(`model.${providerId}`), model.trim());
      }
    },

    saveChatHistory() {
      return localStorage.getItem(key("save_chat")) !== "0";
    },

    setSaveChatHistory(on) {
      localStorage.setItem(key("save_chat"), on ? "1" : "0");
    },

    apiUrl(path) {
      const base = this.backendOrigin();
      const p = path.startsWith("/") ? path : `/${path}`;
      return `${base}${p}`;
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

  window.SoutiPrefs = prefs;
  window.AppPrefs = prefs;
})();
