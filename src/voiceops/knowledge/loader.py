"""Section-aware Markdown knowledge base and catalog loader."""
from pathlib import Path
import re
from typing import Dict, Any, List, Optional
import yaml
from voiceops.config import KB_DIR, CSV_DIR
from voiceops.data.loader import get_data_loader

DOC_TYPE_PRECEDENCE = {
    "policy": 5,
    "agent_guide": 4,
    "help": 3,
    "faq": 2,
    "marketing": 1,
    "meta": 0,
}

class KBChunk:
    """Represents a coherent section chunk from the knowledge base or catalog."""
    def __init__(
        self,
        chunk_id: str,
        section_id: str,
        section_heading: str,
        doc_id: str,
        title: str,
        doc_type: str,
        status: str,
        effective_from: Optional[str],
        version: Optional[str],
        language: str,
        text: str,
        source_file: str,
        chunk_index: int = 0,
        precedence_score: int = 0,
        is_product: bool = False,
        sku: Optional[str] = None
    ):
        self.chunk_id = chunk_id
        self.section_id = section_id
        self.section_heading = section_heading
        self.doc_id = doc_id
        self.title = title
        self.doc_type = doc_type
        self.status = status
        self.effective_from = effective_from
        self.version = str(version) if version is not None else None
        self.language = language
        self.text = text
        self.source_file = source_file
        self.chunk_index = chunk_index
        self.precedence_score = precedence_score
        self.is_product = is_product
        self.sku = sku

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_id": self.section_id,
            "section_heading": self.section_heading,
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "status": self.status,
            "effective_from": self.effective_from or "",
            "version": self.version or "",
            "language": self.language,
            "text": self.text,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "precedence_score": self.precedence_score,
            "is_product": self.is_product,
            "sku": self.sku or "",
        }

    @property
    def searchable_text(self) -> str:
        """Prefixed text optimized for semantic and lexical search."""
        if self.is_product:
            return f"Product Catalog > {self.title} ({self.sku})\n\n{self.text}"
        return f"{self.title} > {self.section_heading}\n\n{self.text}"

def parse_markdown_doc(filepath: Path) -> List[KBChunk]:
    """Parse a single Markdown file into section chunks with frontmatter metadata."""
    content = filepath.read_text(encoding="utf-8")
    
    # 1. Parse frontmatter
    meta: Dict[str, Any] = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            raw_meta = parts[1].strip()
            body = parts[2].strip()
            try:
                meta = yaml.safe_load(raw_meta) or {}
            except Exception:
                # Fallback manual line parse
                for line in raw_meta.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip()

    doc_id = meta.get("doc_id", filepath.stem)
    title = meta.get("title", filepath.stem)
    doc_type = meta.get("doc_type", "policy")
    status = meta.get("status", "current")
    effective_from = meta.get("effective_from")
    version = meta.get("version")
    language = meta.get("language", "en")
    
    precedence = DOC_TYPE_PRECEDENCE.get(doc_type, 1)
    if status == "superseded":
        precedence = -1 # Superseded documents get lowest score

    # 2. Extract sections using regex: ## [SECTION-ID] Heading
    pattern = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.MULTILINE)
    matches = list(pattern.finditer(body))
    chunks: List[KBChunk] = []

    if not matches:
        # Document without section markers (e.g. intro or single rule)
        clean_text = body.strip()
        if clean_text:
            chunk = KBChunk(
                chunk_id=f"{doc_id}_0",
                section_id=doc_id,
                section_heading=title,
                doc_id=doc_id,
                title=title,
                doc_type=doc_type,
                status=status,
                effective_from=effective_from,
                version=version,
                language=language,
                text=clean_text,
                source_file=filepath.name,
                chunk_index=0,
                precedence_score=precedence,
            )
            chunks.append(chunk)
        return chunks

    for i, match in enumerate(matches):
        sec_id = match.group(1).strip()
        sec_heading = match.group(2).strip()
        
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sec_text = body[start_pos:end_pos].strip()
        
        # Sub-chunk only if very long (> 2000 chars), but preserve section_id
        if len(sec_text) > 2000:
            paragraphs = sec_text.split("\n\n")
            curr_text = []
            curr_len = 0
            sub_idx = 0
            for p in paragraphs:
                if curr_len + len(p) > 1500 and curr_text:
                    sub_chunk = KBChunk(
                        chunk_id=f"{doc_id}_{sec_id}_{sub_idx}",
                        section_id=sec_id,
                        section_heading=sec_heading,
                        doc_id=doc_id,
                        title=title,
                        doc_type=doc_type,
                        status=status,
                        effective_from=effective_from,
                        version=version,
                        language=language,
                        text="\n\n".join(curr_text),
                        source_file=filepath.name,
                        chunk_index=sub_idx,
                        precedence_score=precedence,
                    )
                    chunks.append(sub_chunk)
                    curr_text = [p]
                    curr_len = len(p)
                    sub_idx += 1
                else:
                    curr_text.append(p)
                    curr_len += len(p)
            if curr_text:
                sub_chunk = KBChunk(
                    chunk_id=f"{doc_id}_{sec_id}_{sub_idx}",
                    section_id=sec_id,
                    section_heading=sec_heading,
                    doc_id=doc_id,
                    title=title,
                    doc_type=doc_type,
                    status=status,
                    effective_from=effective_from,
                    version=version,
                    language=language,
                    text="\n\n".join(curr_text),
                    source_file=filepath.name,
                    chunk_index=sub_idx,
                    precedence_score=precedence,
                )
                chunks.append(sub_chunk)
        else:
            chunk = KBChunk(
                chunk_id=f"{doc_id}_{sec_id}_0",
                section_id=sec_id,
                section_heading=sec_heading,
                doc_id=doc_id,
                title=title,
                doc_type=doc_type,
                status=status,
                effective_from=effective_from,
                version=version,
                language=language,
                text=sec_text,
                source_file=filepath.name,
                chunk_index=0,
                precedence_score=precedence,
            )
            chunks.append(chunk)

    return chunks

def load_catalog_pseudo_chunks() -> List[KBChunk]:
    """Generate pseudo-chunks for each product in products.csv (e.g. products.csv:ELE-0012)."""
    loader = get_data_loader()
    chunks: List[KBChunk] = []
    
    for sku, prod in loader.products.items():
        sec_id = f"products.csv:{sku}"
        title = prod.get("title", f"Product {sku}")
        brand = prod.get("brand", "")
        cat = prod.get("return_category", "")
        price = prod.get("price_inr", "0")
        fulfilled = prod.get("fulfilled_by", "")
        assured = prod.get("assured", False)
        warranty = prod.get("warranty_months", "0")
        desc = prod.get("description", "")
        
        text = (
            f"SKU: {sku}\n"
            f"Title: {title}\n"
            f"Brand: {brand}\n"
            f"Category: {cat}\n"
            f"Price: ₹{price}\n"
            f"Fulfilled By: {fulfilled}\n"
            f"ZentroMart Assured Badge: {'Yes' if assured else 'No'}\n"
            f"Warranty: {warranty} months brand warranty\n"
            f"Description: {desc}"
        )
        
        chunk = KBChunk(
            chunk_id=f"prod_{sku}",
            section_id=sec_id,
            section_heading=f"{title} ({sku})",
            doc_id="products.csv",
            title=title,
            doc_type="catalog",
            status="current",
            effective_from=None,
            version="1.0",
            language="en",
            text=text,
            source_file="products.csv",
            chunk_index=0,
            precedence_score=3,
            is_product=True,
            sku=sku,
        )
        chunks.append(chunk)
        
    return chunks

def load_all_kb_chunks(kb_dir: Optional[Path] = None, include_catalog: bool = True) -> List[KBChunk]:
    """Load and chunk all Markdown files in kb/ plus product catalog pseudo-chunks."""
    target_dir = Path(kb_dir) if kb_dir else KB_DIR
    all_chunks: List[KBChunk] = []
    
    for md_file in sorted(target_dir.glob("*.md")):
        doc_chunks = parse_markdown_doc(md_file)
        all_chunks.extend(doc_chunks)
        
    if include_catalog:
        catalog_chunks = load_catalog_pseudo_chunks()
        all_chunks.extend(catalog_chunks)
        
    return all_chunks
