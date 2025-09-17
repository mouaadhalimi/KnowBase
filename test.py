from pathlib import Path
from src.core.utils import FileManager
from src.modules.entity_extractor import EntityExtractor
from src.modules.layout_extractor import LayoutExtractor

fm = FileManager()
lx = LayoutExtractor(fm, Path("config/config.yaml"))
ex = EntityExtractor(fm, Path("config/config.yaml"))

blocks = lx.extract(Path("data/mouad/03 - Confirmation Bias and Availability Bias.pdf"))

enriched = ex.add_entities(blocks)

for b in enriched:
    print(b["entities"])