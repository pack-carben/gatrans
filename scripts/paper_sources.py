"""Read publisher Strict OOXML source tables without modifying the originals."""
import argparse
import hashlib
import json
import posixpath
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

BASE = ('https://media.springernature.com/original/springer-static/esm/'
        'art%3A10.1038%2Fs41551-024-01312-5/MediaObjects/')


def read_tables(path):
    """Return sheet-name -> DataFrame, preserving coordinates and cached values."""
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in si.iter() if t.tag.endswith('}t'))
                       for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels = {r.attrib['Id']: r.attrib['Target']
                for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        tables = {}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).iter():
            if not sheet.tag.endswith('}sheet'):
                continue
            rid = next(v for k, v in sheet.attrib.items() if k.endswith('}id'))
            target = rels[rid]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            cells = {}
            for c in ET.fromstring(z.read(target)).iter():
                if not c.tag.endswith('}c'):
                    continue
                match = re.fullmatch(r'([A-Z]+)(\d+)', c.attrib['r'])
                col = 0
                for letter in match[1]:
                    col = col * 26 + ord(letter) - 64
                row = int(match[2]) - 1
                v = next((v.text for v in c if v.tag.endswith('}v')), None)
                if c.attrib.get('t') == 's' and v is not None:
                    v = strings[int(v)]
                elif c.attrib.get('t') == 'inlineStr':
                    v = ''.join(t.text or '' for t in c.iter() if t.tag.endswith('}t'))
                elif v is not None and c.attrib.get('t') not in ('str', 'e', 'd'):
                    v = float(v)
                if v is not None:
                    cells[row, col - 1] = v
            if cells:
                data = [[cells.get((r, c)) for c in range(max(c for _, c in cells) + 1)]
                        for r in range(max(r for r, _ in cells) + 1)]
                tables[sheet.attrib['name']] = pd.DataFrame(data)
        return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/paper_sources')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in (1, 3, 4, 5, 6, 7, 8, 10):
        name = f'41551_2024_1312_MOESM{i}_ESM.' + ('pdf' if i == 1 else 'xlsx')
        path = out / name
        if not path.exists():
            with urllib.request.urlopen(BASE + name, timeout=120) as response:
                content = response.read()
            if not content.startswith(b'%PDF' if i == 1 else b'PK'):
                raise ValueError(f'Unexpected publisher response for {name}')
            path.write_bytes(content)
        manifest.append(dict(file=name, url=BASE + name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if i != 1:
            for sheet, data in read_tables(path).items():
                target = out / 'tables' / str(i)
                target.mkdir(parents=True, exist_ok=True)
                data.to_csv(target / (re.sub(r'[^\w.-]', '_', sheet) + '.csv'), index=False, header=False)
                if args.inspect:
                    print(i, sheet, data.shape)
                    print(data.iloc[:7, :10].to_string(index=False, header=False))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
