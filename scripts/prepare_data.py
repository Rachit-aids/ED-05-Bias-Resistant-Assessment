from __future__ import annotations
import argparse, re
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import normalize_label, save_normalized

LABEL_WORDS = {'CORRECT':'correct','CONTRADICTORY':'contradictory','INCORRECT':'incorrect'}

def lname(tag):
    return tag.split('}',1)[-1].lower()

def text_of(el):
    return ' '.join(''.join(el.itertext()).split()) if el is not None else ''

def parse_xml(path):
    root=ET.parse(path).getroot()
    rows=[]
    # The archive has evolved across mirrors; use semantic element/attribute names rather than fixed positions.
    for el in root.iter():
        if lname(el.tag) not in {'question','item','problem','q'}:
            continue
        attrs={k.lower().split('}')[-1]:v for k,v in el.attrib.items()}
        qid=attrs.get('id') or attrs.get('question_id') or attrs.get('questionid') or ''
        qtext=''
        refs=[]
        answers=[]
        for child in el.iter():
            n=lname(child.tag)
            txt=text_of(child)
            if n in {'question','questiontext','prompt'} and txt and not qtext:
                qtext=txt
            elif n in {'referenceanswer','reference','answer'}:
                a=child.attrib
                role=' '.join(str(v).lower() for v in a.values())
                if n!='answer' or 'student' not in role:
                    if txt: refs.append(txt)
            elif n in {'studentanswer','response','studentresponse'}:
                if txt: answers.append((txt, child.attrib))
        # If nested response records exist, extract labels from their attributes or descendants.
        if answers:
            for ans, a in answers:
                label=''
                for k,v in a.items():
                    if str(v).upper() in LABEL_WORDS:
                        label=LABEL_WORDS[str(v).upper()]
                if not label:
                    for x in el.iter():
                        if text_of(x)==ans:
                            for k,v in x.attrib.items():
                                if str(v).upper() in LABEL_WORDS: label=LABEL_WORDS[str(v).upper()]
                if label and refs:
                    for ref in refs[:1]:
                        rows.append({'question_id':qid,'question':qtext,'reference_answer':ref,'student_answer':ans,'label':label})
    return rows

def parse_csv(path):
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    lower={c.lower():c for c in df.columns}
    aliases={
      'question_id':['question_id','questionid','id'],
      'question':['question','question_text','prompt'],
      'reference_answer':['reference_answer','reference answer','answer','ref_answer'],
      'student_answer':['student_answer','student answer','response','answer_text'],
      'label':['label','student_answer_label','student label','gold']}
    mapping={}
    for target, opts in aliases.items():
        for o in opts:
            if o.lower() in lower: mapping[target]=lower[o.lower()]; break
    missing=[x for x in aliases if x not in mapping]
    if missing: raise ValueError(f'{path}: cannot map columns {missing}; found {list(df.columns)}')
    out=[]
    for _,r in df.iterrows():
        try: lab=normalize_label(r[mapping['label']])
        except ValueError: continue
        out.append({'question_id':r[mapping['question_id']],'question':r[mapping['question']],'reference_answer':r[mapping['reference_answer']],'student_answer':r[mapping['student_answer']],'label':lab})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-dir',default='data/raw')
    ap.add_argument('--out-dir',default='data/processed')
    ap.add_argument('--input',default=None,help='Optional normalized CSV/TSV/CSV mirror')
    args=ap.parse_args(); raw=Path(args.raw_dir); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    files=[Path(args.input)] if args.input else [p for p in raw.rglob('*') if p.suffix.lower() in {'.xml','.csv','.tsv'}]
    rows=[]
    for f in files:
        try:
            rows += parse_xml(f) if f.suffix.lower()=='.xml' else parse_csv(f)
        except Exception as e:
            print(f'Skipping {f}: {e}')
    if not rows: raise SystemExit('No usable three-way rows found. Inspect data/raw or pass --input to a normalized CSV.')
    df=pd.DataFrame(rows).drop_duplicates(subset=['question_id','reference_answer','student_answer'])
    # Preserve an existing split column if available in normalized inputs; otherwise make a grouped train/dev split.
    qids=sorted(df.question_id.astype(str).unique())
    import random
    rng=random.Random(42); rng.shuffle(qids)
    cut=max(1,int(len(qids)*0.15)); dev_q=set(qids[:cut])
    df['split']=['dev' if str(x) in dev_q else 'train' for x in df.question_id]
    train=df[df.split=='train'].drop(columns='split'); dev=df[df.split=='dev'].drop(columns='split')
    train.to_csv(out/'train.csv',index=False); dev.to_csv(out/'dev.csv',index=False)
    df.to_csv(out/'all.csv',index=False)
    print(f'rows={len(df)} train={len(train)} dev={len(dev)} questions={len(qids)}')

if __name__=='__main__': main()
