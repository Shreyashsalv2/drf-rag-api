"""Split page text into overlapping chunks that never cross a page boundary.

The splitter tries the largest natural break first (blank line, line, sentence, word) and only cuts
mid-word when a single word is longer than the chunk size. Offsets are kept, so every chunk can be
traced back to its exact position on its page.
"""

from dataclasses import dataclass

SEPARATORS = ["\n\n", "\n", ". ", "; ", " "]


@dataclass(frozen=True)
class TextChunk:
    index: int
    page: int  # 1-based
    start_char: int  # offset within the page text
    text: str


def _spans(text: str, start: int, end: int, size: int, separators: list[str]) -> list[tuple[int, int]]:
    if end - start <= size:
        return [(start, end)]
    for i, sep in enumerate(separators):
        cuts = []
        pos = text.find(sep, start, end)
        while pos != -1:
            cuts.append(pos + len(sep))
            pos = text.find(sep, pos + len(sep), end)
        if not cuts:
            continue
        pieces, left = [], start
        for cut in cuts:
            if cut > left:
                pieces.append((left, cut))
                left = cut
        if left < end:
            pieces.append((left, end))
        out = []
        for a, b in pieces:
            out.extend(_spans(text, a, b, size, separators[i + 1 :]) if b - a > size else [(a, b)])
        return out
    return [(a, min(a + size, end)) for a in range(start, end, size)]


def _pack(spans: list[tuple[int, int]], size: int, overlap: int) -> list[tuple[int, int]]:
    """Greedily join neighbouring spans up to `size`; start each new chunk with up to `overlap` chars of the last."""
    chunks, current = [], []
    for span in spans:
        if current and span[1] - current[0][0] > size:
            chunks.append((current[0][0], current[-1][1]))
            keep = []
            for s in reversed(current):
                if current[-1][1] - s[0] <= overlap:
                    keep.insert(0, s)
                else:
                    break
            current = keep if keep and span[1] - keep[0][0] <= size else []
        current.append(span)
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def chunk_pages(pages: list[str], size: int = 800, overlap: int = 120) -> list[TextChunk]:
    if size < 100:
        raise ValueError("chunk size must be at least 100 characters")
    if not 0 <= overlap < size // 2:
        raise ValueError("overlap must be between 0 and half the chunk size")
    out: list[TextChunk] = []
    for page_number, text in enumerate(pages, start=1):
        for a, b in _pack(_spans(text, 0, len(text), size, SEPARATORS), size, overlap):
            piece = text[a:b]
            stripped = piece.strip()
            if not stripped:
                continue
            start = a + (len(piece) - len(piece.lstrip()))
            out.append(TextChunk(index=len(out), page=page_number, start_char=start, text=stripped))
    return out
