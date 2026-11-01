from dataclasses import dataclass

from cv_analyser.sections import Section


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    index: int


def chunk_sections(sections: list[Section], max_chars: int = 800) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[Chunk] = []
    for section in sections:
        for text in _split_text(section.content, max_chars):
            chunks.append(Chunk(text=text, section=section.name, index=len(chunks)))
    return chunks


def _split_text(content: str, max_chars: int) -> list[str]:
    pieces = _pack(content.split("\n\n"), max_chars, "\n\n")
    result: list[str] = []
    for piece in pieces:
        if len(piece) > max_chars:
            result.extend(_pack(piece.split("\n"), max_chars, "\n"))
        else:
            result.append(piece)
    return result


def _pack(units: list[str], max_chars: int, separator: str) -> list[str]:
    packed: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{separator}{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = unit
    if current:
        packed.append(current)
    return packed