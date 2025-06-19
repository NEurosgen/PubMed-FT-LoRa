import pandas as pd

from typing import List, Dict
from datasets import Dataset as HFDS
from transformers import GPT2Tokenizer


class PubMedTokenizerPipeline:
    def __init__(self, max_length: int = 1024, model_name: str = "openai-community/gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def users_fields(self, examples: Dict[str, List[str]]) -> List[str]:
        pass

    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        merged = self.users_fields(examples)
        return self.tokenizer(
            merged,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

    def __call__(self, df: pd.DataFrame):
        dataset = HFDS.from_pandas(df)
        tokenized = dataset.map(self._tokenize_function, batched=True)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
        return tokenized
    

