import os
import re

SRC = "scraped_site/pages"
DST = "scraped_site/formatted"

os.makedirs(DST, exist_ok=True)


def strip_yaml(text):
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text


def get_yaml_title(text):
    if text.startswith('---'):
        end = text.index('---', 3)
        yaml_block = text[3:end]
        for line in yaml_block.split('\n'):
            if line.startswith('title:'):
                title = line[6:].strip().strip('"').strip("'")
                return title
    return None


def fix_image_alt(text):
    def replace_img(m):
        alt = m.group(1)
        path = m.group(2)
        if alt.lower() in ('image', '', 'img'):
            stem = os.path.splitext(os.path.basename(path))[0]
            return f'![{stem}]({path})'
        return m.group(0)

    text = re.sub(r'!\[(.*?)\]\(([^)]+)\)', replace_img, text)
    return text


def format_file(content):
    yaml_title = get_yaml_title(content)
    body = strip_yaml(content)

    lines = body.split('\n')
    result = []

    start = 0
    if lines and lines[0].startswith('# ') and yaml_title:
        h1_text = lines[0][2:].strip()
        if h1_text == yaml_title or (yaml_title and h1_text in yaml_title):
            start = 1

    for line in lines[start:]:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]

        if stripped.startswith('###### '):
            text = stripped[7:].strip()
            if re.match(r'^[-–—]', text):
                result.append(f'{indent}*{text}*')
            else:
                result.append(f'{indent}**{text}**')
        elif stripped.startswith('##### '):
            result.append(f'{indent}#### {stripped[6:]}')
        elif stripped.startswith('#### '):
            result.append(f'{indent}### {stripped[5:]}')
        elif stripped.startswith('### '):
            result.append(f'{indent}## {stripped[4:]}')
        elif stripped.startswith('## '):
            result.append(f'{indent}# {stripped[3:]}')
        else:
            result.append(line)

    text = '\n'.join(result)
    text = fix_image_alt(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(l.rstrip() for l in text.split('\n'))

    return text.strip() + '\n'


failed = []
for fname in sorted(os.listdir(SRC)):
    if not fname.endswith('.md'):
        continue
    src_path = os.path.join(SRC, fname)
    dst_path = os.path.join(DST, fname)
    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            content = f.read()
        formatted = format_file(content)
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.write(formatted)
    except Exception as e:
        failed.append((fname, str(e)))

total = sum(1 for f in os.listdir(SRC) if f.endswith('.md'))
done = sum(1 for f in os.listdir(DST) if f.endswith('.md'))
print(f"Processed {done}/{total} files into {DST}/")
if failed:
    for fname, err in failed:
        print(f"  FAILED {fname}: {err}")
