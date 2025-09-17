from pathlib import Path
import fitz
import layoutparser as lp
from docx import Document as DocxDocument
import yaml
import io
import pytesseract
from PIL import Image
from src.core.utils import FileManager
from ultralytics import YOLO
class LayoutExtractor:
    """
    LayoutExtractor (YOLO + OCR)
    ----------------------------

    An AI-powered layout analysis module that extracts **structured content blocks**
    from documents (PDF, DOCX, TXT) for use in RAG pipelines.

    This class uses a YOLOv8 model pretrained on the
    :contentReference[oaicite:1]{index=1} dataset (via :contentReference[oaicite:2]{index=2})
    to detect visual layout elements such as titles, paragraphs, tables, lists,
    and figures from PDF pages rendered as images. For each detected block,
    it uses :contentReference[oaicite:3]{index=3} OCR to extract the actual text content.

    This approach is ideal for scanned or visually rich PDFs where text-based
    extraction (like :contentReference[oaicite:4]{index=4} `.get_text()`) fails or returns
    unordered text.

    Supported file types:
        - PDF  → Page rendered as image → YOLO layout detection → OCR text
        - DOCX → Paragraph-based layout extraction
        - TXT  → Paragraph split by double newlines

    Output format (for every block):
        {
            "type": "title" | "text" | "table" | "list" | "figure",
            "text": "Extracted text content...",
            "page": 0,
            "y": 150.0         # vertical position used for sorting
        }

    Example:
        >>> from pathlib import Path
        >>> from src.core.file_manager import FileManager
        >>> from src.modules.layout_extractor import LayoutExtractor
        >>> fm = FileManager()
        >>> lx = LayoutExtractor(fm, Path("config/config.yaml"))
        >>> blocks = lx.extract(Path("data/mouad/document.pdf"))
        >>> print(blocks[0]["type"], blocks[0]["text"][:100])

    Notes:
        - YOLO model used: `yolov8n-doclaynet.pt` (lightweight & fast)
        - OCR is done using Tesseract (ensure it's installed on the system)
        - Accuracy depends on scan quality (DPI, contrast, skew)
        - Returned blocks are sorted by page then vertical `y` position
        - Use this before chunking to preserve logical document structure

    Performance considerations:
        - Lightweight compared to :contentReference[oaicite:5]{index=5}
        - ~1–2 seconds per PDF page on CPU (yolov8n variant)
        - Suitable for production pipelines
    """

    
    def __init__(self,file_manager: FileManager, config_path:Path):
        """
        Initialize the LayoutExtractor.

        Args:
            file_manager (FileManager): Utility class for loading YAML configs.
            config_path (Path): Path to the YAML configuration file.
        """

        self.files = file_manager
        cfg = self.files.load_config(config_path)
        layout_cfg = cfg.get("layout", {})

        self.dpi = layout_cfg.get("pdf_dpi", 150)
        self.score_thresh = layout_cfg.get("score_thresh", 0.5)
        weights_path = Path("models/yolov8n-doclaynet.pt")
        self.model = YOLO(str(weights_path))
    
    def _extract_pdf(self, path:Path) -> list[dict]:
        """
        Extract structured layout blocks from a PDF using YOLOv8 + Tesseract OCR.

        Each page is rendered as an image, passed through the YOLO model to detect
        layout blocks, and then each block image is cropped and passed to
        Tesseract OCR to extract text.

        Args:
            path (Path): Path to the PDF file.

        Returns:
            list[dict]: List of blocks with type, text, page number, and y position.
        """
        doc= fitz.open(path)
        blocks = []

        for page in doc:
            pix = page.get_pixmap(dpi=self.dpi)
            img= Image.open(io.BytesIO(pix.tobytes("png")))
            results = self.model.predict(img)

            for r in results[0].boxes:
                cls_id = int(r.cls.item())
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                cropped = img.crop((x1,y1,x2,y2))
                text = pytesseract.image_to_string(cropped, lang="eng").strip()

                if text:
                    blocks.append({
                        "type": label.lower(),
                        "text": text,
                        "page":page.number,
                        "y":float(y1)
                    })
        return sorted(blocks, key=lambda b:(b["page"], b["y"]))
    


    def _extract_docx(self, path: Path) -> list[dict]:
        """
        Extract blocks from a DOCX file (paragraphs + headings).

        Args:
            path (Path): Path to the DOCX file.

        Returns:
            list[dict]: List of blocks.
        """
        doc = DocxDocument(str(path))
        blocks = []
        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue
            btype = "title" if "heading" in p.style.name.lower() else "text"
            blocks.append({
                "type": btype,
                "text": txt,
                "page": 0,
                "y": 0.0
            })
        return blocks

    
    


    def _extract_txt(self, path: Path) -> list[dict]:
        """
        Extract blocks from a TXT file (split by double newlines).

        Args:
            path (Path): Path to the TXT file.

        Returns:
            list[dict]: List of blocks.
        """
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        paras = [p.strip() for p in content.split("\n\n") if p.strip()]
        return [
            {"type": "text", "text": p, "page": 0, "y": i}
            for i, p in enumerate(paras)
        ]

    
    


    def extract(self, path: Path) -> list[dict]:
        """
        Extract structured layout blocks from a document.

        Args:
            path (Path): Path to the document.

        Returns:
            list[dict]: List of extracted blocks.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(path)
        if ext == ".docx":
            return self._extract_docx(path)
        if ext == ".txt":
            return self._extract_txt(path)
        raise ValueError(f"Unsupported file type: {ext}")    