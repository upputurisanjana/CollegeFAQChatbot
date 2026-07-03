import json

with open('bvrith_knowledge_base/images_manifest.jsonl', encoding='utf-8') as f:
    images = [json.loads(l) for l in f if l.strip()]

faculty = [i for i in images if i.get('category') == 'faculty']
campus = [i for i in images if i.get('category') == 'campus']
dept = [i for i in images if i.get('category') == 'department']

print(f'Total images: {len(images)}')
print(f'Faculty images: {len(faculty)}')
print(f'Campus images: {len(campus)}')
print(f'Department images: {len(dept)}')
print()

print('Sample faculty images:')
for img in faculty[:10]:
    print(f"  Name: {img.get('semantic_name','N/A')}")
    print(f"  Dept: {img.get('department','N/A')}")
    print(f"  Status: {img.get('status','')}")
    print()
