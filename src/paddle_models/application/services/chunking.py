from typing import List, Dict, Any
from paddle_models.domain.models import Block, Document
from paddle_models.domain.policies.normalizers import extract_item_number, clean_text

class SemanticChunker:
    """
    Groups IR blocks into semantic entries for the knowledge base.
    """
    def chunk(self, document: Document) -> List[Dict[str, Any]]:
        entries = []
        current_entry = None
        
        # Flatten all blocks in order
        all_blocks = []
        for page in document.pages:
            for block in page.blocks:
                # Store page context in block metadata for the chunker
                block.metadata["pdfWidth"] = page.width
                block.metadata["pdfHeight"] = page.height
                all_blocks.append(block)
                
        for block in all_blocks:
            is_title = block.type == "title"
            item_no = extract_item_number(block.text) if block.type == "text" else None
            
            # Boundary detection
            should_start_new = is_title or (item_no is not None and current_entry is not None)
            
            if should_start_new or current_entry is None:
                if current_entry:
                    entries.append(current_entry)
                
                # Derive title for the new entry
                if is_title:
                    job_title = clean_text(block.text)
                elif item_no:
                    job_title = f"第 {item_no} 条"
                else:
                    job_title = "未命名章节"
                    
                current_entry = {
                    "entryId": f"{document.document_id}_{len(entries)}",
                    "jobTitle": job_title,
                    "unitName": document.metadata.get("category", "通用知识"),
                    "pageNumber": block.page_number,
                    "contentNormalized": "", # Will be filled after all blocks are added
                    "blocks": []
                }
            
            # Convert Block object to dict for JSON compatibility
            block_dict = self._block_to_dict(block)
            current_entry["blocks"].append(block_dict)
            
        if current_entry:
            entries.append(current_entry)

        # Post-chunking: fill contentNormalized
        for entry in entries:
            all_text = " ".join([b["text"] for b in entry["blocks"] if b.get("text")])
            entry["contentNormalized"] = clean_text(all_text)
            
        return entries

    def _block_to_dict(self, block: Block) -> Dict[str, Any]:
        d = {
            "type": block.type,
            "text": block.text,
            "bbox": block.bbox.to_list() if block.bbox else None,
            "pageNumber": block.page_number,
            "readingOrder": block.reading_order,
            "structureSource": block.source
        }
        if block.metadata:
            d.update(block.metadata)
        return d
