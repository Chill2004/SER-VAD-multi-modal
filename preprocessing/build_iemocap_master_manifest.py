from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
TARGETS=['valence','arousal','dominance']
def normalize_vad(x): return ((x-1.0)/4.0)*2.0-1.0
def parse_utt(u):
    m=re.match(r'^(Ses\d{2})([FM])_(impro\d+\w*|script\d+_?\d*\w*)_([FM])(\d+)$',u)
    if not m: return None
    s,o,d,sp,t=m.groups(); return {'session':s,'dialog_owner_gender':o,'dialogue_name':d,'dialogue_id':f'{s}{o}_{d}','dialogue_type':'impro' if d.startswith('impro') else 'script','speaker_gender':sp,'speaker_id':f'{s}{sp}','turn_index':int(t)}
def wavs(root):
    rows=[]
    for sd in sorted(root.glob('Session[1-5]')):
        wr=sd/'sentences'/'wav'
        for dd in sorted(wr.iterdir()):
            if not dd.is_dir(): continue
            for w in sorted(dd.glob('*.wav')):
                meta=parse_utt(w.stem)
                if meta is None: raise ValueError(w.stem)
                rows.append({'utterance_id':w.stem,'wav_path':str(w.resolve()),'wav_relpath':w.relative_to(root).as_posix(),**meta})
    df=pd.DataFrame(rows)
    if df.empty or df['utterance_id'].duplicated().any(): raise ValueError('Missing/duplicate WAV utterances')
    return df
def evals(root):
    rows=[]
    pat=re.compile(r'^\[(.*?)\]\s+(\S+)\s+(\S+)\s+\[([\d., ]+)\]')
    for sd in sorted(root.glob('Session[1-5]')):
        for p in sorted((sd/'dialog'/'EmoEvaluation').glob('*.txt')):
            for line in p.read_text(errors='replace').splitlines():
                m=pat.match(line.strip())
                if not m: continue
                try: v=[float(x.strip()) for x in m.group(4).split(',')]
                except ValueError: continue
                if len(v)!=3: continue
                tm=re.match(r'([\d.]+)\s*-\s*([\d.]+)',m.group(1)); a=float(tm.group(1)) if tm else None; b=float(tm.group(2)) if tm else None
                rows.append({'utterance_id':m.group(2),'emotion_cat':m.group(3),'valence':v[0],'arousal':v[1],'dominance':v[2],'start_time':a,'end_time':b,'duration_sec':(b-a) if a is not None else None})
    return pd.DataFrame(rows).drop_duplicates('utterance_id',keep='first')
def transcripts(root):
    rows=[]; pat=re.compile(r'^(\S+)\s+\[([\d.]+)\s*-\s*([\d.]+)\]:\s*(.*)$')
    for sd in sorted(root.glob('Session[1-5]')):
        d=sd/'dialog'/'transcriptions'
        if not d.exists(): d=sd/'dialog'/'Transcriptions'
        for p in sorted(d.glob('*.txt')):
            for line in p.read_text(errors='replace').splitlines():
                m=pat.match(line.strip())
                if m: rows.append({'utterance_id':m.group(1),'transcript_raw':m.group(4).strip()})
    return pd.DataFrame(rows).drop_duplicates('utterance_id',keep='first')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset_root',type=Path,required=True); ap.add_argument('--output_csv',type=Path,required=True); a=ap.parse_args()
    df=wavs(a.dataset_root).merge(evals(a.dataset_root),on='utterance_id',how='left').merge(transcripts(a.dataset_root),on='utterance_id',how='left')
    for c in TARGETS: df[f'{c}_norm']=normalize_vad(df[c])
    df['fold_loso']=df['session'].str.extract(r'Ses(\d{2})')[0].astype(int); df['has_vad_label']=df[TARGETS].notna().all(axis=1); df['has_transcript']=df['transcript_raw'].notna()
    a.output_csv.parent.mkdir(parents=True,exist_ok=True); df.sort_values('utterance_id').to_csv(a.output_csv,index=False,encoding='utf-8-sig'); print('Rows:',len(df),'Saved:',a.output_csv)
if __name__=='__main__': main()
