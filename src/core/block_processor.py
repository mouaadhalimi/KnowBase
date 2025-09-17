

class BlockProcessor:


    def __init__(self, logger=None):
        self.logger = logger
    

    
    def remove_page_headers_footers(self, blocks: list[dict]) -> list[dict]:
        seen_header = False
        cleaned = []
        for b in blocks:
            t = b.get("type","").lower()
            if t == "page-footer":
                continue  
            if t == "page-header":
                if seen_header:
                    continue  
                seen_header = True
            cleaned.append(b)
        return cleaned
    


 