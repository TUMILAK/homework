(() => {
  function waitOpen(ws) {
    return new Promise((resolve, reject) => {
      if (ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      ws.addEventListener("open", () => resolve(), { once: true });
      ws.addEventListener(
        "error",
        () => reject(new Error("WebSocket open failed")),
        { once: true },
      );
    });
  }

  function waitConfigured(ws) {
    return new Promise((resolve, reject) => {
      let timer = window.setTimeout(() => {
        cleanup();
        reject(new Error("Configure timeout"));
      }, 9000);

      function cleanup() {
        window.clearTimeout(timer);
        ws.removeEventListener("message", onMsg);
      }

      function onMsg(ev) {
        let data = null;
        try {
          data = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (!data || !data.type) return;

        if (data.type === "configured") {
          cleanup();
          resolve(data);
          return;
        }

        if (data.type === "error") {
          cleanup();
          reject(new Error(data.detail || "服务端错误"));
        }
      }

      ws.addEventListener("message", onMsg);
    });
  }

  window.AgentSocket = {
    async openAndConfigure() {
      const ws = new WebSocket(window.AppPrefs.wsAgentURL());

      const cfgPromise = waitConfigured(ws);
      await waitOpen(ws);

      const prefs = window.AppPrefs.deepseek();
      ws.send(
        JSON.stringify({
          type: "configure",
          deepseek_api_key: prefs.apiKey,
          deepseek_base_url: prefs.baseUrl,
          model: prefs.model,
        }),
      );

      const cfg = await cfgPromise;
      if (cfg && cfg.warning) {
        ws.close();
        throw new Error(cfg.warning);
      }
      return ws;
    },
  };
})();
