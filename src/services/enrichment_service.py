import hashlib

class EnrichmentService:
    @staticmethod
    def enrich_sparse(doc, keys):
        """Extracts and lowercases content from specified keys in a dict for sparse vectors."""
        values = []
        for key in keys:
            val = doc.get(key, "")
            if isinstance(val, list):
                values.extend([str(v).lower() for v in val])
            else:
                values.append(str(val).lower())
        return " ".join(values)

    @staticmethod
    def enrich_dense(doc, keys):
        """Markdown formatted and lowercased content for dense vectors."""
        # We prioritize productName and productDescription as core fields if they exist in keys
        name_key = "productName"
        desc_key = "productDescription"
        
        name = str(doc.get(name_key, "")).strip()
        desc = str(doc.get(desc_key, "")).strip()
        
        markdown = []
        if name:
            markdown.append(f"# {name}")
        if desc:
            markdown.append(desc)
        
        # Handle other keys
        other_keys = [k for k in keys if k not in [name_key, desc_key]]
        if other_keys:
            others = []
            for key in other_keys:
                val = doc.get(key)
                if val:
                    if isinstance(val, list):
                        val_str = ", ".join([str(v) for v in val])
                    else:
                        val_str = str(val)
                    others.append(f"{key}: {val_str}")
            
            if others:
                markdown.append("---")
                markdown.extend(others)
        
        final_text = "\n".join(markdown).lower()
        return final_text
