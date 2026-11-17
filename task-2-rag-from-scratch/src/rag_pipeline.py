"""
rag_pipeline.py - Complete End-to-End RAG Pipeline from scratch.
No LangChain, LlamaIndex, or external RAG frameworks.
"""
import os
import sys
from typing import List, Dict, Any, Optional

# Add src to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ingestion import ingest_directory, Document
from chunker import TextChunker, CONFIG_A, CONFIG_B, Chunk
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever
from generator import Generator


class RAGPipeline:
    """
    Modular, scratch-built RAG QA System.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_config: TextChunker = CONFIG_B,
        embedding_model: str = "all-MiniLM-L6-v2",
        generator_model: Optional[str] = "google/flan-t5-small",
        init_generator: bool = True
    ):
        self.data_dir = data_dir
        self.chunker = chunk_config
        self.embedding_model_name = embedding_model

        print(f"[RAG] Loading documents from: {data_dir}")
        self.documents: List[Document] = ingest_directory(data_dir)
        print(f"[RAG] Extracted {len(self.documents)} document pages/sections.")

        print(f"[RAG] Chunking with size={self.chunker.chunk_size}, overlap={self.chunker.overlap}...")
        self.chunks: List[Chunk] = self.chunker.chunk_documents(self.documents)
        print(f"[RAG] Generated {len(self.chunks)} text chunks.")

        print(f"[RAG] Initializing Embedder ({embedding_model})...")
        self.embedder = Embedder(model_name=embedding_model)

        texts = [c.text for c in self.chunks]
        print(f"[RAG] Computing embeddings for {len(texts)} chunks...")
        self.embeddings = self.embedder.embed(texts)

        print(f"[RAG] Indexing vectors into FAISS (dim={self.embedder.dim})...")
        self.vector_store = VectorStore(dim=self.embedder.dim)
        self.vector_store.add(self.embeddings)

        self.retriever = Retriever(
            embedder=self.embedder,
            store=self.vector_store,
            chunks=self.chunks
        )

        self.generator = None
        if init_generator and generator_model:
            try:
                self.generator = Generator(model_name=generator_model)
            except Exception as e:
                print(f"[RAG] Warning: Generator initialization failed ({e}). Answers will be retrieved passages.")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for query."""
        return self.retriever.retrieve(query, k=k)

    def answer_question(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        End-to-end question answering: retrieve passages, build context, generate answer.
        """
        retrieved_passages = self.retrieve(query, k=k)
        context = self.retriever.build_context(retrieved_passages)

        if self.generator:
            gen_res = self.generator.generate(query, context, retrieved_passages)
            answer = gen_res["answer"]
            sources = gen_res["sources"]
        else:
            answer = retrieved_passages[0]["text"] if retrieved_passages else "No context found."
            sources = [
                {"passage": r["text"][:200], "page": r["page"], "source": r["source"], "score": round(r["score"], 4)}
                for r in retrieved_passages[:3]
            ]

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "sources": sources,
            "all_retrieved": retrieved_passages
        }


if __name__ == "__main__":
    task2_root = os.path.dirname(current_dir)
    data_path = os.path.join(task2_root, "data")
    pipeline = RAGPipeline(data_dir=data_path, chunk_config=CONFIG_B)
    sample_q = "How many layers are in the Transformer encoder?"
    result = pipeline.answer_question(sample_q, k=3)
    print("\n--- SAMPLE QUESTION ---")
    print(f"Q: {result['query']}")
    print(f"A: {result['answer']}")
    print(f"Citations:")
    for s in result["sources"]:
        print(f"  - [{s['source']}, p.{s['page']}] (Score: {s['score']}): {s['passage'][:120]}...")

