import re
from pathlib import Path
from typing import List, Dict, Any

class DocumentChunker:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_document_title(self, text: str, fallback_filename: str) -> str:
        """Extract H1 title if present, otherwise format filename."""
        h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        return Path(fallback_filename).stem.replace("_", " ").title()

    def split_by_sections(self, text: str) -> List[Dict[str, str]]:
        """Split markdown text into logical sections based on H2/H3 headers."""
        sections = []
        lines = text.split("\n")
        current_header = "General Overview"
        current_content = []

        for line in lines:
            header_match = re.match(r"^(#{2,3})\s+(.+)$", line)
            if header_match:
                if current_content:
                    sections.append({
                        "section_title": current_header,
                        "text": "\n".join(current_content).strip()
                    })
                    current_content = []
                current_header = header_match.group(2).strip()
            else:
                current_content.append(line)

        if current_content:
            sections.append({
                "section_title": current_header,
                "text": "\n".join(current_content).strip()
            })

        return [s for s in sections if s["text"]]

    def chunk_text_sliding_window(self, text: str, section_title: str) -> List[str]:
        """Chunk text using a sliding window with overlap while preserving words."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Find nearest sentence break or newline or space
            split_at = -1
            sub = text[start:end]
            for delimiter in ["\n\n", ".\n", ". ", "\n", " "]:
                last_pos = sub.rfind(delimiter)
                if last_pos > self.chunk_size // 2:
                    split_at = start + last_pos + len(delimiter)
                    break

            if split_at == -1 or split_at <= start:
                split_at = end

            chunk = text[start:split_at].strip()
            if chunk:
                chunks.append(chunk)

            start = split_at - self.chunk_overlap
            if start < 0:
                start = 0
            if start >= text_len or split_at >= text_len:
                break

        return chunks

    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single markdown file into indexed chunks."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        file_name = file_path.name
        doc_title = self.extract_document_title(raw_text, file_name)
        sections = self.split_by_sections(raw_text)

        all_chunks = []
        chunk_idx = 0

        for sec in sections:
            sec_title = sec["section_title"]
            sec_text = sec["text"]
            text_chunks = self.chunk_text_sliding_window(sec_text, sec_title)

            for c in text_chunks:
                # Add context prefix to chunk text for better semantic matching
                enriched_text = f"[{doc_title} > {sec_title}]\n{c}"
                all_chunks.append({
                    "chunk_id": f"{file_path.stem}_chk_{chunk_idx}",
                    "source_file": file_name,
                    "document_title": doc_title,
                    "section_title": sec_title,
                    "text": enriched_text,
                    "raw_text": c,
                    "char_count": len(enriched_text)
                })
                chunk_idx += 1

        return all_chunks

    def process_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """Process all markdown files in the specified directory."""
        all_chunks = []
        md_files = sorted(list(directory_path.glob("*.md")) + list(directory_path.glob("*.txt")))
        for f in md_files:
            chunks = self.process_document(f)
            all_chunks.extend(chunks)
        return all_chunks
