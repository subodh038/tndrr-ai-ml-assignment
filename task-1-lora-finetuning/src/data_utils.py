"""
data_utils.py
-------------
Utilities for loading, formatting, and creating HuggingFace datasets
from the NL→JSON calendar assistant dataset.
"""
import json
from typing import List, Dict, Any

try:
    from datasets import Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


INSTRUCTION = "Convert the following natural language to structured JSON."


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from a JSON file.

    Args:
        path: Path to the JSON file containing a list of example dicts,
              each with 'input' (str) and 'output' (dict) keys.

    Returns:
        List of example dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def format_for_training(example: Dict[str, Any], tokenizer=None) -> str:
    """
    Format a single example into the Alpaca-style prompt+completion string
    used for SFT training.

    Prompt format:
        ### Instruction:
        Convert the following natural language to structured JSON.

        ### Input:
        {input}

        ### Response:
        {json_output}

    Args:
        example:   Dict with 'input' (str) and 'output' (dict) keys.
        tokenizer: (Optional) tokenizer — reserved for future use with
                   special token handling.

    Returns:
        A single formatted string containing both prompt and completion.
    """
    input_text = example["input"]
    output_json = json.dumps(example["output"], ensure_ascii=False)

    return (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{output_json}"
    )


def format_prompt_only(input_text: str) -> str:
    """
    Return the prompt portion only (no completion) — used at inference time.

    Args:
        input_text: Natural language query string.

    Returns:
        Formatted prompt string ending just before the response.
    """
    return (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )


def create_hf_dataset(examples: List[Dict[str, Any]]):
    """
    Convert a list of examples to a HuggingFace Dataset object with a 'text'
    column containing the fully formatted training string for SFTTrainer.

    Args:
        examples: List of dicts with 'input' and 'output' keys.

    Returns:
        HuggingFace Dataset with columns: input, output, text.

    Raises:
        ImportError: If the `datasets` library is not installed.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "The 'datasets' library is required. Install it with: pip install datasets"
        )

    records = []
    for ex in examples:
        records.append(
            {
                "input": ex["input"],
                "output": json.dumps(ex["output"], ensure_ascii=False),
                "text": format_for_training(ex),
            }
        )

    return Dataset.from_list(records)


if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base, "dataset", "train.json")

    examples = load_dataset(train_path)
    print(f"Loaded {len(examples)} training examples")
    print("\nSample formatted example:")
    print(format_for_training(examples[0]))

    if HF_AVAILABLE:
        ds = create_hf_dataset(examples[:5])
        print(f"\nHuggingFace Dataset: {ds}")
        print(ds[0])
