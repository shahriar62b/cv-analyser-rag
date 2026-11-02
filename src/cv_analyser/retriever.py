from cv_analyser.chunker import Chunk
from cv_analyser.embedder import Embedder
from cv_analyser.vector_store import InMemoryVectorStore, SearchResult


class CVRetriever:
    def __init__(
        self, embedder: Embedder, store: InMemoryVectorStore | None = None
    ) -> None:
        self._embedder = embedder
        # "is not None" matters: an empty store is falsy because it defines __len__.
        self._store = store if store is not None else InMemoryVectorStore()

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        self._store.add(chunks, vectors)

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vector = self._embedder.embed([query])[0]
        return self._store.search(query_vector, top_k)