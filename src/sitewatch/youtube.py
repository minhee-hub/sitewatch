from __future__ import annotations


def prepare_source(value):
    from sitewatch.media import parse_source

    parsed = parse_source(value)
    if isinstance(parsed, str) and is_youtube(parsed):
        info = resolve_youtube(parsed)
        return info["stream_url"], info
    return parsed, None


def is_youtube(value: str) -> bool:
    text = str(value).strip().lower()
    return "youtube.com/" in text or "youtu.be/" in text


def resolve_youtube(url: str) -> dict:
    """공개 유튜브 주소에서 OpenCV가 읽을 재생 주소를 고른다."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp가 없습니다. pip install yt-dlp 후 다시 실행해 주세요.") from exc

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "best[height<=720]/best",
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(str(url).strip(), download=False)
    except Exception as exc:
        raise RuntimeError(f"유튜브 주소를 열지 못했습니다: {exc}") from exc
    if info is None:
        raise RuntimeError("유튜브 영상 정보를 받지 못했습니다.")
    if "entries" in info:
        entries = [item for item in info["entries"] or [] if item]
        if not entries:
            raise RuntimeError("재생할 유튜브 영상을 찾지 못했습니다.")
        info = entries[0]
    stream = _stream_url(info)
    return {
        "stream_url": stream,
        "title": str(info.get("title") or "유튜브 영상"),
        "is_live": bool(info.get("is_live")),
        "webpage_url": str(info.get("webpage_url") or url),
    }


def _stream_url(info: dict) -> str:
    formats = []
    for fmt in info.get("formats") or []:
        url = fmt.get("url")
        if not url or fmt.get("vcodec") in (None, "none"):
            continue
        height = int(fmt.get("height") or 10000)
        ext = str(fmt.get("ext") or "")
        protocol = str(fmt.get("protocol") or "")
        score = 0
        if height <= 720:
            score += 40
        if ext == "mp4":
            score += 25
        if protocol.startswith("http") and "m3u8" not in protocol:
            score += 15
        if "m3u8" in protocol:
            score += 8
        formats.append((score, -min(height, 720), url))
    if formats:
        formats.sort(reverse=True)
        return formats[0][2]
    direct = info.get("url")
    if direct and "youtube.com/watch" not in str(direct):
        return str(direct)
    raise RuntimeError("재생 주소를 찾지 못했습니다. 공개된 유튜브 주소인지 확인해 주세요.")
