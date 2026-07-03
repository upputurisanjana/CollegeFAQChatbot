import csv
from collections import Counter

with open('bvrith_knowledge_base/crawl_log.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

page_statuses = Counter(r['status'] for r in rows if r['type'] == 'page')
img_statuses = Counter(r['status'] for r in rows if r['type'] == 'image')
print('PAGE statuses:', dict(page_statuses))
print('IMAGE statuses:', dict(img_statuses))
print()

failed_pages = [r for r in rows if r['type'] == 'page' and r['status'] not in ('200', '301', '302', '303', '304')]
print(f'Failed/errored pages: {len(failed_pages)}')
for r in failed_pages[:30]:
    print(f"  [{r['status']}] {r['url']}")

print()
failed_images = [r for r in rows if r['type'] == 'image' and r['status'] not in ('200', '301', '302', '303', '304', 'skipped_duplicate', 'skipped_small')]
print(f'Failed/timed-out images: {len(failed_images)}')
for r in failed_images[:20]:
    print(f"  [{r['status']}] {r['url']}")
