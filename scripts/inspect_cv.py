import sys
from pathlib import Path

from cv_analyser.chunker import chunk_sections
from cv_analyser.loaders.dispatcher import load_document
from cv_analyser.sections import split_into_sections


def main(path: Path) -> None:
    text = load_document(path)
    sections = split_into_sections(text)
    chunks = chunk_sections(sections)
    print(f"{path.name}: {len(text)} chars, {len(sections)} sections, {len(chunks)} chunks")
    for section in sections:
        print(f"  - {section.name}: {len(section.content)} chars")


if __name__ == "__main__":
    main(Path(sys.argv[1]))