import sys; sys.path.insert(0, '.')
import importlib, rag, image_search
importlib.reload(image_search)
importlib.reload(rag)

# 1. Departments query
chunks = rag.retrieve(rag._expand_query('What departments are offered?'), None, 20)
print(f'Departments query: {len(chunks)} chunks retrieved')
dept_urls = [c.source_url for c in chunks if any(k in c.source_url for k in
    ['computer-science','electronics-and-comm','electrical-and-elec','information-technology','cse-artificial','under-graduate'])]
print(f'Dept-specific chunks: {len(dept_urls)}')
for u in sorted(set(dept_urls))[:8]:
    print(f'  {u[-65:]}')

# 2. Refusal detection
print()
partial = 'Based on context, BVRIT offers CSE, ECE, EEE, IT. For more info please contact admissions@bvrithyderabad.edu.in'
full = "I don't have that information. Please contact admissions@bvrithyderabad.edu.in for an authoritative answer."
print('Partial answer refused:', rag._detect_refusal(partial))
print('Full refusal detected:', rag._detect_refusal(full))

# 3. ECE typo image search
imgs = image_search.search_images('give images of any 5 ecec faculty', limit=5)
print()
print(f'ECE images (ecec typo): {len(imgs)} found')
for img in imgs:
    print(f'  {img.get("semantic_name")} | dept: {img.get("department","")[:40]}')
