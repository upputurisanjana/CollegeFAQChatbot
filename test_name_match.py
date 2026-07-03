import sys, json
sys.path.insert(0, '.')
import importlib, image_search
importlib.reload(image_search)

# Search for Dr. Aruna Rao and show full details
imgs = image_search.search_images('Dr. Aruna Rao S L Computer Science and Engineering', limit=3)
for img in imgs:
    print('semantic_name:', repr(img.get('semantic_name')))
    print('context_heading:', repr(img.get('context_heading')))
    print('category:', img.get('category'))
    print('department:', img.get('department'))
    print('page_url:', img.get('page_url','')[-60:])
    print('url:', img.get('url','')[-60:])
    print('score:', img.get('_score'))
    print()
