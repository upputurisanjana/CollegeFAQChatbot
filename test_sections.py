import sys, json
sys.path.insert(0, '.')
from ingest import infer_section
from collections import Counter

chunks = []
with open('bvrith_knowledge_base/chunks.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

sections = Counter(infer_section(c) for c in chunks)
print('Section distribution (new):')
for section, count in sorted(sections.items(), key=lambda x: -x[1]):
    print(f'  {section}: {count}')

test_cases = [
    ('cse about dept',   'https://bvrithyderabad.edu.in/computer-science-and-engineering/about-the-department'),
    ('fee details',      'https://bvrithyderabad.edu.in/admission/fee-details'),
    ('faculty profile',  'https://bvrithyderabad.edu.in/computer-science-and-engineering/dr-kvn-sunitha'),
    ('placements',       'https://bvrithyderabad.edu.in/placements/placement-details'),
    ('hostel',           'https://bvrithyderabad.edu.in/admission/hostel'),
    ('about bvrit',      'https://bvrithyderabad.edu.in/about-bvrith'),
    ('IT dept',          'https://bvrithyderabad.edu.in/under-graduate/information-technology'),
    ('contact',          'https://bvrithyderabad.edu.in/contact-us'),
]
print()
print('Spot checks:')
for label, url in test_cases:
    chunk = {'source_url': url, 'breadcrumb': [], 'page_title': ''}
    print(f'  {label:20s} -> {infer_section(chunk)}')
