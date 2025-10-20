  - Create a script for tokenizer utilities:
     ```python
     from transformers import AutoTokenizer
     def load_tokenizer(model_path):
         return AutoTokenizer.from_pretrained(model_path)
     ```
