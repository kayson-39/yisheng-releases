"""Portable glossary files, with explicit conflict handling in the UI."""
import csv,json,io
from pathlib import Path

def normalize(rows):
    out=[];seen=set()
    for row in rows:
        if not isinstance(row,(list,tuple)) or len(row)!=2:raise ValueError('术语格式应为两列：日语、中文')
        a,b=(str(x).strip() for x in row)
        if not a or not b:continue
        if (a,b) not in seen:out.append([a,b]);seen.add((a,b))
    return out

def read_glossary(path):
    p=Path(path);text=p.read_text('utf-8-sig')
    if p.suffix.lower()=='.json':
        data=json.loads(text)
        if isinstance(data,dict):
            translation=data.get('translate')
            data=data.get('glossary') if 'glossary' in data else (translation.get('glossary') if isinstance(translation,dict) else None)
        if not isinstance(data,list):raise ValueError('JSON 中未找到术语列表')
        return normalize(data)
    rows=list(csv.reader(io.StringIO(text),delimiter='\t' if p.suffix.lower()=='.tsv' else ','))
    if rows and rows[0] in (['日语','中文'],['source','target'],['jp','zh']):rows=rows[1:]
    return normalize(rows)

def merge_glossary(current,incoming,replace_conflicts=False):
    result={a:b for a,b in normalize(current)};conflicts=[]
    for a,b in normalize(incoming):
        if a in result and result[a]!=b:
            conflicts.append(a)
            if not replace_conflicts:continue
        result[a]=b
    return [[a,b] for a,b in result.items()],conflicts

def write_glossary(path,rows):
    p=Path(path);rows=normalize(rows)
    if p.suffix.lower()=='.json':p.write_text(json.dumps({'version':1,'glossary':rows},ensure_ascii=False,indent=2),'utf-8')
    else:
        with p.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f,delimiter='\t' if p.suffix.lower()=='.tsv' else ',');w.writerow(['日语','中文']);w.writerows(rows)
