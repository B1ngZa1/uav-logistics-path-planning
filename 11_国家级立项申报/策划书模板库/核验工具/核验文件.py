"""Read-only validation: JSON to stdout; never modify downloaded originals."""
from pathlib import Path
import json
import re
import zipfile
import xml.etree.ElementTree as ET
import sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')

root = Path(__file__).resolve().parent.parent
results = []
for path in sorted(root.rglob('*')):
    if path.suffix.lower() not in ('.pdf', '.docx'):
        continue
    row = {'path': path.relative_to(root).as_posix(), 'valid': False}
    try:
        if path.suffix.lower() == '.pdf':
            if path.open('rb').read(5) != b'%PDF-':
                raise ValueError('Not a PDF: possible HTML error page')
            pdf = PdfReader(path)
            row['pages'] = len(pdf.pages)
            row['encrypted'] = pdf.is_encrypted
            selected = list(range(min(14, len(pdf.pages))))
            snippets = []
            for i in selected:
                content = pdf.pages[i].extract_text() or ''
                if i < 3 or re.search(r'CONTENTS|Contents|目\s*录|copyright|licen[cs]e|版权', content):
                    snippets.append({'physical_page': i+1, 'text': content[:11000]})
            row['evidence'] = snippets
            row['metadata'] = {str(k):str(v) for k,v in (pdf.metadata or {}).items()}
        else:
            with zipfile.ZipFile(path) as z:
                doc = ET.fromstring(z.read('word/document.xml'))
                ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                paragraphs = [''.join(p.itertext()) for p in doc.findall('.//w:p',ns)]
                row['paragraphs'] = len(paragraphs)
                row['text_chars'] = sum(map(len,paragraphs))
                row['evidence'] = [{'text':'\n'.join(paragraphs)[:20000]}]
                if 'docProps/app.xml' in z.namelist():
                    app = ET.fromstring(z.read('docProps/app.xml'))
                    row['cached_pages'] = app.findtext('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Pages')
                row['pages'] = None
        row['valid'] = True
    except Exception as exc:
        row['error'] = str(exc)
    results.append(row)
payload = json.dumps(results, ensure_ascii=False, indent=2)
(Path(__file__).resolve().parent / '核验结果.json').write_text(payload, encoding='utf-8')
print(payload)
