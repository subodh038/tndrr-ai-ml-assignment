"""
generator.py - Text generation using a small seq2seq model (flan-t5-base).
Works fully on CPU with no GPU required.
"""
from typing import List


class Generator:
    """
    Wraps google/flan-t5-base for RAG-style question answering.
    Uses a text2text-generation pipeline (encoder-decoder).

    Args:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        from transformers import pipeline
        print(f"[generator] Loading {model_name} (may take a minute on first run)...")
        self.model_name = model_name
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=200,
        )
        print("[generator] Model ready.")

    def generate(self, question: str, context: str, sources: List[dict]) -> dict:
        """
        Generate an answer given a question and retrieved context.

        Args:
            question: The user's question.
            context: Concatenated retrieved passages (from Retriever.build_context).
            sources: List of source dicts from Retriever.retrieve().

        Returns:
            dict with keys:
              - answer (str): Generated answer text.
              - sources (list): Top-3 cited passages with text, page, source.
        """
        # Truncate context to ~800 chars to stay within model limits
        truncated_ctx = context[:800]
        prompt = (
            f"Answer the question based on the context below.\n\n"
            f"Context: {truncated_ctx}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        result = self.pipe(prompt)
        answer_text = result[0]["generated_text"].strip()

        cited_sources = [
            {
                "passage": s["text"][:200],
                "page": s["page"],
                "source": s["source"],
                "score": round(s.get("score", 0.0), 4),
            }
            for s in sources[:3]
        ]

        return {
            "answer": answer_text,
            "sources": cited_sources,
        }


if __name__ == "__main__":
    gen = Generator()
    ctx = "Transformers use self-attention to process sequences in parallel. BERT is a bidirectional transformer pretrained on masked language modeling."
    result = gen.generate("What is BERT?", ctx, [])
    print("Answer:", result["answer"])
