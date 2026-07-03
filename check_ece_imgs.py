import json
ece_imgs = []
with open('bvrith_knowledge_base/images_manifest.jsonl', encoding='utf-8') as f:
    for line in f:
        img = json.loads(line.strip())
        if img.get('status') == 'saved':
            page_url = img.get('page_url') or ''
            if 'electronics-and-communication' in page_url:
                ece_imgs.append(img)

print(f'ECE images in manifest: {len(ece_imgs)}')
for img in ece_imgs[:5]:
    cat = img.get('category')
    dept = img.get('department')
    name = img.get('semantic_name')
    page = (img.get('page_url') or '')[-50:]
    print(f'  cat={cat} dept={dept} name={name} page=...{page}')
