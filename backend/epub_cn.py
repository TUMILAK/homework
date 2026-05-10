"""EPUB 汉化：解压 → 按片段批量调用 DeepSeek（JSON 对齐，参考 AiNiee 书籍工作流）→ 重打包。"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup, NavigableString
from openai import AsyncOpenAI

# 与 AiNiee 思路接近：块级正文合并为一批再译，减轻碎片化与 API 次数
_BATCH_MAX_CHARS = 7200
_BATCH_MAX_SEGMENTS = 40
_SINGLE_SEGMENT_MAX = 12000

_SKIP_PARENT_TAGS = frozenset({"script", "style", "head", "title", "svg", "math"})
_SKIP_TEXT_PARENT = frozenset(
    {"script", "style", "head", "title", "svg", "math", "code", "pre", "aside"}
)
_MIN_CHARS = 10

_EPUB_SYSTEM = """你是专业电子书/小说译者（工作流对齐常见 Ai 书籍翻译工具如 AiNiee 的做法）。
用户会给你一个 JSON：键为字符串 "0","1","2"... 值为从 EPUB/XHTML 抽出的待译正文片段。
你必须只回复一个 JSON 对象（不要 markdown 代码围栏），键集合与输入完全一致，不得增删键，值为对应片段的简体中文译文。
保持人称、语气与文学性；全书专有名词、人名译法应前后一致。
若某条几乎无自然语言（纯数字、符号），可原样返回。"""

_POLISH_SYSTEM = """你是中文润色编辑。用户给你一个 JSON：键为 "0","1"... 值为机译后的简体中文小说正文片段。
只回复一个 JSON 对象（不要 markdown 代码围栏），键集合与输入完全一致，不得增删键。
将每条润色为更自然流畅的书面中文：统一标点与空格、消除翻译腔与重复用词，不改变叙事、人称与情节；专名与术语保持一致。
若某条已通顺可原样返回。"""


def _local(tag: str) -> str:
    return tag.split("}")[-1] if tag and "}" in tag else (tag or "")


def _read_container_opf_relpath(extract_root: Path) -> str:
    p = extract_root / "META-INF" / "container.xml"
    if not p.is_file():
        raise ValueError("缺少 META-INF/container.xml")
    tree = ET.parse(p)
    for el in tree.iter():
        if _local(el.tag).lower() == "rootfile":
            fp = el.attrib.get("full-path") or el.attrib.get("Full-path")
            if fp:
                return fp.replace("\\", "/").lstrip("/")
    raise ValueError("container.xml 中未找到 rootfile")


def _parse_opf_spine_hrefs(opf_path: Path) -> List[str]:
    tree = ET.parse(opf_path)
    root = tree.getroot()
    manifest_el = None
    spine_el = None
    for ch in list(root):
        ln = _local(ch.tag).lower()
        if ln == "manifest":
            manifest_el = ch
        elif ln == "spine":
            spine_el = ch
    if manifest_el is None or spine_el is None:
        raise ValueError("content.opf 缺少 manifest 或 spine")

    id_to_href: dict[str, Tuple[str, Optional[str]]] = {}
    for item in list(manifest_el):
        if _local(item.tag).lower() != "item":
            continue
        iid = item.attrib.get("id")
        href = item.attrib.get("href")
        mt = item.attrib.get("media-type")
        if iid and href:
            id_to_href[iid] = (href.replace("\\", "/"), mt)

    hrefs: List[str] = []
    for itemref in list(spine_el):
        if _local(itemref.tag).lower() != "itemref":
            continue
        rid = itemref.attrib.get("idref")
        if not rid or rid not in id_to_href:
            continue
        href, mt = id_to_href[rid]
        low = (href or "").lower()
        mtl = (mt or "").lower()
        if mtl in ("application/xhtml+xml", "application/html", "text/html"):
            hrefs.append(href)
            continue
        if low.endswith((".xhtml", ".html", ".htm")) and "image" not in mtl:
            hrefs.append(href)
    return hrefs


def _looks_like_prose(s: str) -> bool:
    for c in s.strip():
        if c.isalpha():
            return True
        o = ord(c)
        if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF:
            return True
        if 0xAC00 <= o <= 0xD7AF:
            return True
    return False


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp932", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _collect_translatable_nodes(soup: BeautifulSoup) -> List[NavigableString]:
    root = soup.body or soup.find("html")
    if not root:
        return []
    out: List[NavigableString] = []
    for node in root.descendants:
        if not isinstance(node, NavigableString):
            continue
        parent = getattr(node, "parent", None)
        if parent is None:
            continue
        pname = (parent.name or "").lower()
        if pname in _SKIP_TEXT_PARENT:
            continue
        anc = parent
        skip = False
        while anc is not None and getattr(anc, "name", None):
            if anc.name.lower() in _SKIP_PARENT_TAGS:
                skip = True
                break
            anc = anc.parent
        if skip:
            continue
        s = str(node)
        if len(s.strip()) < _MIN_CHARS:
            continue
        if not _looks_like_prose(s):
            continue
        out.append(node)
    return out


def _parse_epub_regex_rules(raw: str) -> List[Tuple[re.Pattern, str]]:
    """每行：`正则|||替换` 或 `正则|||替换|||imsx`（可选 flags 字母 i/m/s/x）。# 开头为注释。"""
    rules: List[Tuple[re.Pattern, str]] = []
    for raw_line in (raw or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|||", 2)
        if len(parts) < 2:
            continue
        pat_s = parts[0].strip()
        rep = parts[1]
        flags_s = parts[2].strip() if len(parts) >= 3 else ""
        flags = 0
        for c in flags_s.lower():
            if c == "i":
                flags |= re.IGNORECASE
            elif c == "m":
                flags |= re.MULTILINE
            elif c == "s":
                flags |= re.DOTALL
            elif c == "x":
                flags |= re.VERBOSE
        rep = rep.replace("\\n", "\n").replace("\\t", "\t")
        try:
            rules.append((re.compile(pat_s, flags), rep))
        except re.error:
            continue
    return rules


def _apply_regex_rules(s: str, rules: List[Tuple[re.Pattern, str]]) -> str:
    if not rules or not s:
        return s
    out = s
    for pat, repl in rules:
        out = pat.sub(repl, out)
    return out


def _strip_json_fence(raw: str) -> str:
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


async def _call_json_segment_batch(
    client: AsyncOpenAI,
    model: str,
    segments: List[str],
    system_instruction: str,
    glossary: str,
) -> List[str]:
    if not segments:
        return []
    user_obj = {str(i): s for i, s in enumerate(segments)}
    user = json.dumps(user_obj, ensure_ascii=False)
    gloss_block = ""
    if glossary.strip():
        gloss_block = "\n【术语表（请严格遵守，原文=译文 每行一条）】\n" + glossary.strip() + "\n"
    messages = [
        {"role": "system", "content": system_instruction + gloss_block},
        {"role": "user", "content": user},
    ]
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            kw = dict(
                model=model,
                messages=messages,
                temperature=max(0.1, 0.22 - attempt * 0.04),
            )
            if attempt == 0:
                kw["response_format"] = {"type": "json_object"}
            resp = await client.chat.completions.create(**kw)
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(_strip_json_fence(raw))
            if not isinstance(data, dict):
                raise ValueError("模型未返回 JSON 对象")
            out: List[str] = []
            for i in range(len(segments)):
                v = data.get(str(i))
                if not isinstance(v, str):
                    out.append(segments[i])
                else:
                    out.append(v if v.strip() else segments[i])
            return out
        except Exception as e:
            last_err = e
    raise RuntimeError(f"EPUB JSON 批处理失败: {last_err}") from last_err


async def _translate_nodes_batched_json(
    client: AsyncOpenAI,
    model: str,
    nodes: List[NavigableString],
    *,
    glossary: str = "",
    regex_rules: List[Tuple[re.Pattern, str]],
    polish_second: bool,
) -> None:
    if not nodes:
        return
    i = 0
    n = len(nodes)
    while i < n:
        batch_nodes: List[NavigableString] = []
        batch_texts: List[str] = []
        char_budget = 0
        while i < n and len(batch_texts) < _BATCH_MAX_SEGMENTS:
            raw = str(nodes[i])
            piece = raw if len(raw) <= _SINGLE_SEGMENT_MAX else raw[:_SINGLE_SEGMENT_MAX] + "\n(…已截断)"
            piece = _apply_regex_rules(piece, regex_rules)
            if char_budget + len(piece) > _BATCH_MAX_CHARS and batch_texts:
                break
            batch_nodes.append(nodes[i])
            batch_texts.append(piece)
            char_budget += len(piece)
            i += 1
        outs = await _call_json_segment_batch(
            client, model, batch_texts, _EPUB_SYSTEM, glossary
        )
        if polish_second and outs:
            outs = await _call_json_segment_batch(
                client, model, outs, _POLISH_SYSTEM, glossary
            )
        if len(outs) != len(batch_nodes):
            outs = batch_texts
        for node, src, out in zip(batch_nodes, batch_texts, outs):
            node.replace_with(out if out.strip() else src)


async def _translate_ncx_file(
    path: Path,
    client: AsyncOpenAI,
    model: str,
    glossary: str,
    regex_rules: List[Tuple[re.Pattern, str]],
    polish_second: bool,
) -> None:
    raw = _read_text_file(path)
    pattern = re.compile(r"(<text\b[^>]*>)(.*?)(</text\s*>)", re.IGNORECASE | re.DOTALL)
    spans: List[Tuple[int, int, str]] = []
    for m in pattern.finditer(raw):
        inner = m.group(2)
        if len(inner.strip()) < 2 or not _looks_like_prose(inner):
            continue
        spans.append((m.start(2), m.end(2), inner))
    if not spans:
        return
    segments = [_apply_regex_rules(s[2], regex_rules) for s in spans]
    outs = await _call_json_segment_batch(client, model, segments, _EPUB_SYSTEM, glossary)
    if polish_second and outs:
        outs = await _call_json_segment_batch(client, model, outs, _POLISH_SYSTEM, glossary)
    if len(outs) != len(segments):
        outs = segments
    new_raw = raw
    for (start, end, _src), out in zip(spans[::-1], outs[::-1]):
        new_raw = new_raw[:start] + out + new_raw[end:]
    path.write_text(new_raw, encoding="utf-8")


async def localize_epub_to_workspace(
    *,
    epub_bytes: bytes,
    stem: str,
    client: AsyncOpenAI,
    model: str,
    out_dir: Path,
    glossary: str = "",
    regex_rules: str = "",
    polish_second: bool = True,
) -> Tuple[str, int]:
    """Write ``out_dir / {stem}.zh.epub``; return relative path ``epub_out/...`` and size."""
    compiled_rules = _parse_epub_regex_rules(regex_rules)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^\w\u4e00-\u9fff\-_.]+", "_", stem)[:80] or "book"
    out_path = out_dir / f"{safe_stem}.zh.epub"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src_epub = root / "in.epub"
        src_epub.write_bytes(epub_bytes)
        extract = root / "unzipped"
        extract.mkdir()
        with zipfile.ZipFile(src_epub, "r") as zf:
            zf.extractall(extract)

        opf_rel = _read_container_opf_relpath(extract)
        opf_path = (extract / opf_rel).resolve()
        try:
            opf_path.relative_to(extract.resolve())
        except ValueError:
            raise ValueError("非法 OPF 路径") from None
        if not opf_path.is_file():
            raise ValueError(f"找不到 OPF: {opf_rel}")

        opf_dir = opf_path.parent
        hrefs = _parse_opf_spine_hrefs(opf_path)
        seen = set()
        for href in hrefs:
            if href in seen:
                continue
            seen.add(href)
            xhtml_path = (opf_dir / href).resolve()
            try:
                xhtml_path.relative_to(extract.resolve())
            except ValueError:
                continue
            if not xhtml_path.is_file():
                continue
            low = xhtml_path.suffix.lower()
            if low not in (".xhtml", ".html", ".htm"):
                continue
            html = _read_text_file(xhtml_path)
            soup = BeautifulSoup(html, "html.parser")
            nodes = _collect_translatable_nodes(soup)
            if nodes:
                await _translate_nodes_batched_json(
                    client,
                    model,
                    nodes,
                    glossary=glossary,
                    regex_rules=compiled_rules,
                    polish_second=polish_second,
                )
            xhtml_path.write_text(str(soup), encoding="utf-8")

        for ncx in sorted(extract.rglob("*.ncx")):
            if not ncx.is_file():
                continue
            try:
                ncx.relative_to(extract.resolve())
            except ValueError:
                continue
            await _translate_ncx_file(
                ncx, client, model, glossary, compiled_rules, polish_second
            )

        _write_epub_zip(extract, out_path)

    rel = f"epub_out/{out_path.name}"
    return rel, out_path.stat().st_size


def _write_epub_zip(source_dir: Path, dest_epub: Path) -> None:
    mimetype = source_dir / "mimetype"
    with zipfile.ZipFile(dest_epub, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if mimetype.is_file():
            zf.write(mimetype, "mimetype", compress_type=zipfile.ZIP_STORED)
        for p in sorted(source_dir.rglob("*")):
            if not p.is_file():
                continue
            arc = p.relative_to(source_dir).as_posix()
            if arc == "mimetype":
                continue
            zf.write(p, arc, compress_type=zipfile.ZIP_DEFLATED)
