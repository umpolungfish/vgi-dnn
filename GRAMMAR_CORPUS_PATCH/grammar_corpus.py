#!/usr/bin/env python3
"""Provenance-first recursive IG document and Lean indexer. Standard-library core."""
import argparse, collections, hashlib, json, re, sqlite3, sys
from pathlib import Path

DEFAULT_DOCS = '/home/mrnob0dy666/imsgct/ig-docs'
DEFAULT_LEAN = '/home/mrnob0dy666/imsgct/p4rakernel/p4ramill/Imscribing'
TEXT_EXT = {'.md','.markdown','.txt','.rst','.org','.tex','.lean','.json','.yaml','.yml','.toml','.csv','.html','.htm','.adoc','.typ'}
DOC_EXT = TEXT_EXT | {'.pdf','.docx','.odt'}
LEAN_RE = re.compile(r'^\s*(?:(?:private|protected|noncomputable|unsafe|partial|local)\s+)*(theorem|lemma|def|abbrev|instance|structure|class|inductive|coinductive|axiom|example|opaque|constant|abbreviation)\s+([\w\.\'\u0080-\uffff]+)',re.M)
IMPORT_RE = re.compile(r'^\s*import\s+(.+)$',re.M)
NS_RE = re.compile(r'^\s*namespace\s+([\w.\']+)',re.M)
TOKEN_RE = re.compile(r'(?u)\b[\w\-]{4,}\b')
HEADING_RE = re.compile(r'^\s{0,3}#{1,6}\s+(.+?)\s*$',re.M)

def decode(path):
    ext=path.suffix.lower()
    if ext in TEXT_EXT:
        b=path.read_bytes()
        return b.decode('utf-8-sig',errors='replace'), 'text'
    if ext=='.docx':
        import zipfile,xml.etree.ElementTree as ET
        with zipfile.ZipFile(path) as z:
            data=z.read('word/document.xml')
        root=ET.fromstring(data)
        ns='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        return '\n'.join(''.join(t.text or '' for t in p.iter(ns+'t')) for p in root.iter(ns+'p')), 'docx'
    if ext=='.odt':
        import zipfile,xml.etree.ElementTree as ET
        with zipfile.ZipFile(path) as z: data=z.read('content.xml')
        root=ET.fromstring(data)
        return '\n'.join(''.join(t.itertext()) for t in root.iter() if t.tag.endswith('}p')), 'odt'
    if ext=='.pdf':
        try:
            import fitz
            with fitz.open(path) as d: return '\n'.join(page.get_text() for page in d),'pdf'
        except ImportError:
            try:
                from pypdf import PdfReader
                return '\n'.join(p.extract_text() or '' for p in PdfReader(str(path)).pages),'pdf'
            except ImportError: raise RuntimeError('PDF extraction requires pymupdf or pypdf; file remains cataloged')
    raise ValueError('unsupported extension')

def init(db):
    db.executescript('''PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY, root TEXT NOT NULL, relative_path TEXT NOT NULL, path TEXT NOT NULL UNIQUE, kind TEXT NOT NULL, sha256 TEXT, bytes INTEGER, chars INTEGER, extraction TEXT, error TEXT);
    CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, start_line INTEGER NOT NULL, content TEXT NOT NULL, FOREIGN KEY(file_id) REFERENCES files(id));
    CREATE TABLE IF NOT EXISTS declarations(id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL, namespace TEXT, line INTEGER, FOREIGN KEY(file_id) REFERENCES files(id));
    CREATE TABLE IF NOT EXISTS imports(id INTEGER PRIMARY KEY,file_id INTEGER NOT NULL,module TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS mentions(id INTEGER PRIMARY KEY,file_id INTEGER NOT NULL,term TEXT NOT NULL,context TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS ix_declarations_name ON declarations(name);
    CREATE INDEX IF NOT EXISTS ix_imports_module ON imports(module);
    CREATE INDEX IF NOT EXISTS ix_mentions_term ON mentions(term);
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(content, tokenize='unicode61');''')

def chunks(text,limit=3000):
    lines=text.splitlines(keepends=True); buf=''; start=1; idx=0
    for line_number,line in enumerate(lines,1):
        if buf and len(buf)+len(line)>limit:
            yield idx,start,buf; idx+=1;buf='';start=line_number
        if len(line)>limit:
            for k in range(0,len(line),limit):
                if buf: yield idx,start,buf;idx+=1;buf=''
                yield idx,line_number,line[k:k+limit];idx+=1
            start=line_number+1
        else: buf+=line
    if buf:yield idx,start,buf

def index(roots,db_path,max_bytes=40_000_000):
    db_path=Path(db_path).expanduser().resolve();db_path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(str(db_path));init(db)
    db.execute('DELETE FROM chunk_search');
    for t in ['mentions','imports','declarations','chunks','files']:db.execute('DELETE FROM '+t)
    counts=collections.Counter(); errors=[]
    paths=[]
    for label,root in roots:
        root=Path(root).expanduser().resolve()
        if not root.is_dir():
            errors.append({'root':str(root),'error':'directory does not exist'});continue
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.is_symlink() or any(part in {'.git','.lake','node_modules','target','.venv','venv','__pycache__'} for part in p.parts): continue
            if p.suffix.lower() in DOC_EXT:paths.append((label,root,p))
    try:
        for label,root,p in paths:
            size=p.stat().st_size;kind='lean' if p.suffix.lower()=='.lean' else 'document'
            error=None;text='';mode=''; digest=hashlib.sha256()
            with p.open('rb') as stream:
                for b in iter(lambda:stream.read(1024*1024),b''):digest.update(b)
            if size>max_bytes:error='exceeds max byte size';counts['skipped_oversize']+=1
            else:
                try:text,mode=decode(p)
                except Exception as exc:error=f'{type(exc).__name__}: {exc}';counts['extraction_errors']+=1
            cur=db.execute('INSERT INTO files(root,relative_path,path,kind,sha256,bytes,chars,extraction,error) VALUES(?,?,?,?,?,?,?,?,?)',(label,str(p.relative_to(root)),str(p),kind,digest.hexdigest(),size,len(text),mode,error))
            fid=cur.lastrowid;counts[kind]+=1
            if text:
                for ordinal,line,part in chunks(text):
                    cur=db.execute('INSERT INTO chunks(file_id,ordinal,start_line,content) VALUES(?,?,?,?)',(fid,ordinal,line,part))
                    db.execute('INSERT INTO chunk_search(rowid,content) VALUES(?,?)',(cur.lastrowid,part));counts['chunks']+=1
                if kind=='lean':
                    for m in LEAN_RE.finditer(text):
                        ln=text.count('\n',0,m.start())+1
                        namespaces=[(x.start(),x.group(1)) for x in NS_RE.finditer(text) if x.start()<m.start()]
                        ns=namespaces[-1][1] if namespaces else ''
                        db.execute('INSERT INTO declarations(file_id,kind,name,namespace,line) VALUES(?,?,?,?,?)',(fid,m.group(1),m.group(2),ns,ln));counts['declarations']+=1
                    for m in IMPORT_RE.finditer(text):
                        for token in m.group(1).split():
                            if token.startswith('--'):break
                            db.execute('INSERT INTO imports(file_id,module) VALUES(?,?)',(fid,token));counts['imports']+=1
                else:
                    for term in set(TOKEN_RE.findall(p.stem.casefold())):
                        if len(term)>=5:
                            db.execute('INSERT INTO mentions(file_id,term,context) VALUES(?,?,?)',(fid,term,'filename'));counts['filename_terms']+=1
            if counts['document']+counts['lean'] and (counts['document']+counts['lean'])%500==0:db.commit()
        db.commit()
        summary={'roots':[{ 'label':label,'path':str(root)} for label,root in roots], 'counts':dict(counts),'indexed_files':len(paths),'errors':errors,'database':str(db_path)}
        return summary
    finally:db.close()

def search(db_path,term,limit):
    db=sqlite3.connect(db_path)
    query=' '.join(x for x in re.findall(r'[\w]+',term) if x)
    if not query:return []
    return [{'path':r[0],'line':r[1],'excerpt':r[2][:500]} for r in db.execute('SELECT files.path,chunks.start_line,chunks.content FROM chunk_search JOIN chunks ON chunks.id=chunk_search.rowid JOIN files ON files.id=chunks.file_id WHERE chunk_search MATCH ? LIMIT ?',(' OR '.join(query.split()),limit))]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--docs',default=DEFAULT_DOCS);ap.add_argument('--lean',default=DEFAULT_LEAN);ap.add_argument('--db',default='grammar_corpus.sqlite');ap.add_argument('--report',default='grammar_corpus_report.json');ap.add_argument('--max-bytes',type=int,default=40_000_000);ap.add_argument('--search');ap.add_argument('--limit',type=int,default=10)
    a=ap.parse_args()
    if a.search:
        print(json.dumps(search(a.db,a.search,a.limit),indent=2,ensure_ascii=False));return
    out=index([('ig_docs',a.docs),('lean',a.lean)],a.db,a.max_bytes)
    Path(a.report).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('GRAMMAR_CORPUS',json.dumps(out,ensure_ascii=False))
    if out['errors']:sys.exit(2)
if __name__=='__main__':main()
