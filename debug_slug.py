import json
from collections import Counter

page_url_patterns = Counter()
with open('bvrith_knowledge_base/images_manifest.jsonl', encoding='utf-8') as f:
    for line in f:
        img = json.loads(line.strip())
        if img.get('status') == 'saved' and img.get('category') == 'other':
            page_url = img.get('page_url') or ''
            # Get the path segment
            path = page_url.replace('https://bvrithyderabad.edu.in','').strip('/')
            top = path.split('/')[0] if path else 'homepage'
            page_url_patterns[top] += 1

print("'other' images by top-level path:")
for path, count in sorted(page_url_patterns.items(), key=lambda x: -x[1])[:20]:
    print(f"  {count:3d}  /{path}")
