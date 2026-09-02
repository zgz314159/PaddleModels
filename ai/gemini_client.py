"""Gemini 模型调用：模型名标准化、模型列表、表格修复"""

import base64
import json
import re
from pathlib import Path
from typing import Optional

GEMINI_FIXED_MODEL = "gemini-1.5-flash"

_LAST_GEMINI_HTTP_STATUS: Optional[int] = None


def normalize_gemini_model_name(model: str) -> str:
    """Normalize model name for REST endpoints.

    The listModels endpoint returns names like "models/gemini-2.5-flash".
    Our generateContent endpoint expects the short name after /models/.

    Accepts either form and also tolerates callers passing a full suffix like
    ":generateContent".
    """

    m = (model or "").strip()
    if m.startswith("models/"):
        m = m[len("models/"):]
    if m.endswith(":generateContent"):
        m = m[:-len(":generateContent")]
    return m.strip()


def list_gemini_models(api_key: str) -> int:
    """List available models for the provided API key.

    This helps diagnose 404 model-not-found issues.
    """

    import urllib.error
    import urllib.request

    if not (api_key or "").strip():
        print("[Gemini] GEMINI_API_KEY 未配置，无法列出模型。")
        return 2

    for api_version in ("v1", "v1beta"):
        url = f"https://generativelanguage.googleapis.com/{api_version}/models?key={api_key.strip()}"
        print(f"[Gemini] Listing models via {api_version}...")
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            print(f"[Gemini] HTTPError {getattr(e, 'code', 'unknown')} {getattr(e, 'reason', '')} body={body[:500]}")
            continue
        except Exception as ex:
            print(f"[Gemini] Request failed: {ex}")
            continue

        try:
            parsed = json.loads(body)
            models = parsed.get("models") or []
            if not models:
                print("[Gemini] (no models returned)")
                continue
            for m in models:
                name = (m or {}).get("name")
                methods = (m or {}).get("supportedGenerationMethods") or []
                if name:
                    short = name
                    if isinstance(short, str) and short.startswith("models/"):
                        short = short[len("models/"):]
                    print(f"  - {short} (raw={name}) methods={methods}")
        except Exception as ex:
            print(f"[Gemini] Failed to parse response: {ex}")
            continue

    return 0


def fix_table_with_gemini(image_path: Path, api_key: str, model: str) -> Optional[str]:
    import urllib.error
    import urllib.request

    global _LAST_GEMINI_HTTP_STATUS
    _LAST_GEMINI_HTTP_STATUS = None

    model = normalize_gemini_model_name(model)

    print(f">>> 尝试调用 Gemini 处理图片: {image_path}")

    if not api_key:
        print(">>> 错误：未发现 API Key，跳过 Gemini")
        return None
    if not image_path.exists():
        return None

    mime = "image/png"
    try:
        raw = image_path.read_bytes()
    except Exception:
        return None

    data_b64 = base64.b64encode(raw).decode("ascii")

    prompt = (
        "你是一个文档专家。请将图中的表格还原为标准的 Markdown 表格。\n"
        "如果单元格内有图片占位符（例如 [IMAGE_REF:xxx.png]），请原样保留。\n"
        "确保跨行合并/空单元格逻辑合理，输出只包含 Markdown 表格本体，不要解释。"
    )

    # Prefer v1 for gemini-1.5-* models; keep v1beta as fallback for older keys.
    api_versions = ("v1", "v1beta")
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime, "data": data_b64}},
                ],
            }
        ],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }

    last_http_error_body = ""
    for api_version in api_versions:
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent?key={api_key}"
        print(f">>> Gemini request: apiVersion={api_version} model={model}")
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                _LAST_GEMINI_HTTP_STATUS = getattr(resp, "status", None)
                break
        except urllib.error.HTTPError as e:
            _LAST_GEMINI_HTTP_STATUS = getattr(e, "code", None)
            try:
                last_http_error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                last_http_error_body = ""
            print(
                f">>> Gemini HTTPError: {getattr(e, 'code', 'unknown')} {getattr(e, 'reason', '')} body={last_http_error_body[:500]}"
            )

            # If model isn't found on this API version, try the next version.
            if getattr(e, "code", None) == 404 and api_version != api_versions[-1]:
                continue

            if getattr(e, "code", None) == 404:
                print(
                    "[Gemini] Model not found (404). Run with --list-gemini-models to see available models, "
                    "then pass --gemini-model <name> (either 'gemini-2.5-flash' or 'models/gemini-2.5-flash' works)."
                )
            return None
        except Exception as ex:
            print(f">>> Gemini 调用异常: {ex}")
            return None
    else:
        return None

    try:
        parsed = json.loads(body)
        candidates = parsed.get("candidates") or []
        if not candidates:
            return None
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
        text = (text or "").strip()
        if not text:
            return None
        # Gemini sometimes wraps in code fences.
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        return text.strip()
    except Exception:
        return None


def gemini_table_fix(markdown: str, api_key: str, model: str) -> Optional[str]:
    """Fix a (possibly garbled) markdown table via Gemini text-only REST API.

    IMPORTANT: must preserve any image placeholders like [IMAGE_REF:xxx.png].
    """

    md = (markdown or "").strip()
    if not md or not api_key:
        return None

    model = normalize_gemini_model_name(model)

    try:
        import requests  # type: ignore
    except Exception:
        print("WARNING: requests not installed; cannot use gemini_table_fix().")
        return None

    prompt = (
        "你是一个文档专家。下面是一段来自 DOCX 转换的 Markdown 表格，但可能存在乱码/错位。\n"
        "请在不丢失信息的前提下，修复为标准 Markdown 表格。\n"
        "如果表格中包含图片占位符（例如 [IMAGE_REF:xxx.png]），必须原样保留并保持在正确单元格中。\n"
        "只输出 Markdown 表格本体，不要解释。\n\n"
        "表格如下：\n"
        f"{md}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }

    try:
        resp = requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=120,
        )
    except Exception:
        return None

    if resp.status_code < 200 or resp.status_code >= 300:
        return None

    try:
        parsed = resp.json()
        candidates = parsed.get("candidates") or []
        if not candidates:
            return None
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
        text = (text or "").strip()
        if not text:
            return None
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        return text.strip()
    except Exception:
        return None
