(() => {
  const qs = (id) => document.getElementById(id);

  let mode = "solve";
  /** @type {{role:string,content:string}[]} */
  let history = [];

  function creds() {
    const cfg = SoutiPrefs.loadActiveConfig();
    if (!cfg.apiKey.trim()) {
      throw new Error("请先在「设置」填写 API Key");
    }
    return cfg;
  }

  function updateModelLabel() {
    const pid = SoutiPrefs.activeProvider();
    const cfg = SoutiPrefs.loadActiveConfig();
    const label = SoutiPrefs.PROVIDERS[pid]?.label || pid;
    qs("modelLabel").textContent = `${label} · ${cfg.model}`;
  }

  function logLine(kind, text) {
    const el = document.createElement("pre");
    el.className = "line" + (kind ? " " + kind : "");
    el.textContent = text;
    qs("log").appendChild(el);
    qs("log").scrollTop = qs("log").scrollHeight;
  }

  function addBubble(role, content) {
    const panel = qs("chatPanel");
    const div = document.createElement("div");
    div.className = `chat-bubble ${role}`;
    div.textContent = content;
    panel.appendChild(div);
    panel.scrollTop = panel.scrollHeight;
  }

  function setMode(next) {
    mode = next;
    qs("modeSolve").classList.toggle("active", mode === "solve");
    qs("modeChat").classList.toggle("active", mode === "chat");
  }

  qs("modeSolve").onclick = () => setMode("solve");
  qs("modeChat").onclick = () => setMode("chat");

  qs("saveChatToggle").checked = SoutiPrefs.saveChatHistory();
  qs("saveChatToggle").onchange = () => {
    SoutiPrefs.setSaveChatHistory(qs("saveChatToggle").checked);
  };

  qs("clearChat").onclick = () => {
    history = [];
    qs("chatPanel").innerHTML = "";
    logLine("warn", "[local] 对话已清空");
  };

  async function chatSend(text, opts = {}) {
    const cfg = creds();
    const userText = text.trim();
    if (!userText) return;

    history.push({ role: "user", content: userText });
    addBubble("user", userText);

    const saveAnswer =
      mode === "solve" ? true : SoutiPrefs.saveChatHistory() || opts.forceSave;

    logLine("", `[${mode}] 请求中…`);
    const r = await fetch(SoutiPrefs.apiUrl("/api/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: cfg.apiKey,
        base_url: cfg.baseUrl,
        model: cfg.model,
        messages: history,
        mode,
        save_answer: saveAnswer,
        source: opts.source || "",
      }),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);

    history.push({ role: "assistant", content: j.answer });
    addBubble("assistant", j.answer);
    if (j.saved_file) logLine("", `已存档：answers/${j.saved_file}`);
  }

  qs("sendText").onclick = async () => {
    const text = qs("userInput").value;
    qs("sendText").disabled = true;
    try {
      await chatSend(text);
      qs("userInput").value = "";
    } catch (e) {
      logLine("err", String(e.message || e));
    } finally {
      qs("sendText").disabled = false;
    }
  };

  function firstSelectedFile() {
    const files = qs("imgFiles").files;
    return files && files.length ? files[0] : null;
  }

  qs("ocrOnly").onclick = async () => {
    const file = firstSelectedFile();
    if (!file) {
      logLine("warn", "请先选择图片");
      return;
    }
    const cfg = creds();
    const fd = new FormData();
    fd.append("file", file);
    fd.append("api_key", cfg.apiKey);
    fd.append("base_url", cfg.baseUrl);
    fd.append("model", cfg.model);
    try {
      const r = await fetch(SoutiPrefs.apiUrl("/api/ocr"), { method: "POST", body: fd });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText);
      qs("ocrPreview").value = j.text || "";
      logLine("", `OCR 完成（${j.method}）`);
    } catch (e) {
      logLine("err", String(e.message || e));
    }
  };

  qs("solveImage").onclick = async () => {
    const file = firstSelectedFile();
    const preview = qs("ocrPreview").value.trim();
    if (!file && !preview) {
      logLine("warn", "请选择图片或填写 OCR 预览");
      return;
    }
    const cfg = creds();
    const fd = new FormData();
    if (file) fd.append("file", file);
    fd.append("api_key", cfg.apiKey);
    fd.append("base_url", cfg.baseUrl);
    fd.append("model", cfg.model);
    fd.append("save_answer", "true");
    if (preview) fd.append("confirm_text", preview);
    qs("solveImage").disabled = true;
    try {
      const r = await fetch(SoutiPrefs.apiUrl("/api/solve-image"), { method: "POST", body: fd });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText);
      if (j.ocr_text) qs("ocrPreview").value = j.ocr_text;
      addBubble("user", `[图片] ${file?.name || "预览文本"}\n\n${j.ocr_text || qs("ocrPreview").value}`);
      addBubble("assistant", j.answer);
      if (j.saved_file) logLine("", `已存档：answers/${j.saved_file}`);
    } catch (e) {
      logLine("err", String(e.message || e));
    } finally {
      qs("solveImage").disabled = false;
    }
  };

  qs("solveBatch").onclick = async () => {
    const files = qs("imgFiles").files;
    if (!files || !files.length) {
      logLine("warn", "请选择多张图片");
      return;
    }
    const cfg = creds();
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    fd.append("api_key", cfg.apiKey);
    fd.append("base_url", cfg.baseUrl);
    fd.append("model", cfg.model);
    fd.append("save_answer", "true");
    qs("solveBatch").disabled = true;
    try {
      const r = await fetch(SoutiPrefs.apiUrl("/api/solve-images-batch"), {
        method: "POST",
        body: fd,
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText);
      for (const item of j.results || []) {
        if (item.ok) {
          addBubble("user", `[批量] ${item.filename}\n\n${item.ocr_text}`);
          addBubble("assistant", item.answer);
        } else {
          logLine("err", `${item.filename}: ${item.error}`);
        }
      }
      logLine("", `批量完成：${j.count} 张`);
    } catch (e) {
      logLine("err", String(e.message || e));
    } finally {
      qs("solveBatch").disabled = false;
    }
  };

  qs("uploadFile").onchange = async () => {
    const input = qs("uploadFile");
    const file = input.files?.[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch(SoutiPrefs.apiUrl("/api/upload-temp"), { method: "POST", body: fd });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText);
      qs("dataPath").value = j.path;
      logLine("", `已上传到 data/${j.path}`);
    } catch (e) {
      logLine("err", String(e.message || e));
    }
    input.value = "";
  };

  qs("listData").onclick = async () => {
    const folder = qs("dataPath").value.trim();
    try {
      const r = await fetch(
        SoutiPrefs.apiUrl(`/api/data/list?folder=${encodeURIComponent(folder)}`)
      );
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText);
      qs("filePreview").value = (j.files || []).join("\n") || "（空）";
      logLine("", `共 ${(j.files || []).length} 个可读文件`);
    } catch (e) {
      logLine("err", String(e.message || e));
    }
  };

  qs("readFile").onclick = async () => {
    const path = qs("dataPath").value.trim();
    if (!path) {
      logLine("warn", "请填写相对路径");
      return;
    }
    try {
      const r = await fetch(SoutiPrefs.apiUrl("/api/data/read"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText);
      qs("filePreview").value = j.text || "";
    } catch (e) {
      logLine("err", String(e.message || e));
    }
  };

  qs("readFolder").onclick = async () => {
    const folder = qs("dataPath").value.trim();
    try {
      const r = await fetch(SoutiPrefs.apiUrl("/api/data/read-folder"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder, max_files: 20 }),
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText);
      qs("filePreview").value = j.combined_text || "";
      logLine("", `已合并 ${j.file_count} 个文件`);
    } catch (e) {
      logLine("err", String(e.message || e));
    }
  };

  qs("solveFile").onclick = async () => {
    const path = qs("dataPath").value.trim();
    if (!path) {
      logLine("warn", "请填写文件路径");
      return;
    }
    const cfg = creds();
    const fd = new FormData();
    fd.append("path", path);
    fd.append("api_key", cfg.apiKey);
    fd.append("base_url", cfg.baseUrl);
    fd.append("model", cfg.model);
    fd.append("save_answer", "true");
    qs("solveFile").disabled = true;
    try {
      const r = await fetch(SoutiPrefs.apiUrl("/api/data/solve-file"), {
        method: "POST",
        body: fd,
      });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail || r.statusText);
      addBubble("user", `[文件] ${path}`);
      addBubble("assistant", j.answer);
      if (j.saved_file) logLine("", `已存档：answers/${j.saved_file}`);
    } catch (e) {
      logLine("err", String(e.message || e));
    } finally {
      qs("solveFile").disabled = false;
    }
  };

  updateModelLabel();
  window.addEventListener("storage", updateModelLabel);
  logLine("", "搜题 Agent 就绪。请先在「设置」配置 API Key。");
})();
