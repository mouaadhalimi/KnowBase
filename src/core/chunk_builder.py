from typing import List, Dict
import re
from semantic_text_splitter import TextSplitter
from typing import List, Dict


class ChunkBuilder:


    def __init__(self, tokenizer_model: str, chunk_size:int=500, logger=None):

        self.logger = logger
        self.splitter = TextSplitter.from_tiktoken_model(tokenizer_model,(chunk_size,chunk_size))
    

    @staticmethod
    def _clean_text(text:str)->str:
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip() 
    
    def remove_near_duplicates(self, blocks:List[Dict], windows:int = 10)->List[Dict]:

        seen = []
        cleaned = []
        for b in blocks:
            txt = re.sub(r"\s+", " ", b["text"]).strip().lower()
            if txt in seen[-windows:]:
                if self.logger:
                    self.logger.debug(f"Duplicate removed (local window): {txt[:50]}...")
                continue
            cleaned.append(b)
            seen.append(txt)
        return cleaned

    def merge_small_blocks(self, blocks: List[Dict], min_words:int=20) ->List[Dict]:
        merged, buffer = [], None
        
        for b in blocks:
            txt = b["text"].strip()
            page = b.get("page", 0)
            ents = b.get("entities", [])

            word_count = len(txt.split())

            if word_count < min_words:
                if buffer is None:
                    buffer = b.copy()
                    buffer["entities"] = ents.copy()
                else:
                    if buffer.get("page", 0) == page:
                        buffer["text"]+= " "+txt
                        existing = { (e["text"], e["label"]) for e in buffer["entities"] }
                        for e in ents:
                            if (e["text"], e["label"]) not in existing:
                                buffer["entities"].append(e)
                    else:
                        merged.append(buffer)
                        buffer = b.copy()
                        buffer["entities"] = ents.copy()
            else:
                if buffer:
                    if buffer.get("page", 0)==page:
                        buffer["text"] += " "+txt
                        existing = { (e["text"], e["label"]) for e in buffer["entities"] }
                        for e in ents:
                            if (e["text"], e["label"]) not in existing:
                                buffer["entities"].append(e)
                        merged.append(buffer)
                        buffer = None
                        
                    else:
                        merged.append(buffer)
                        merged.append(b)
                        buffer = None
                else:
                    merged.append(b)
        if buffer:
            merged.append(buffer)
        return merged
    def split_text(self, text:str) -> list[str]:
        """
        Split text into semantic chunks using the configured TextSplitter.

        Args:
            text (str): The text to split.

        Returns:
            list[str]: List of chunk strings.
        """
        return self.splitter.chunks(text)
 