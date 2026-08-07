from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
TARGET=['valence_norm','arousal_norm','dominance_norm']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest_csv',type=Path,required=True); ap.add_argument('--out_dir',type=Path,required=True); ap.add_argument('--test_session',default='Ses05'); a=ap.parse_args()
    df=pd.read_csv(a.manifest_csv).dropna(subset=TARGET); train=df[df.session!=a.test_session].reset_index(drop=True); test=df[df.session==a.test_session].reset_index(drop=True)
    if train.empty or test.empty: raise ValueError('Empty split')
    a.out_dir.mkdir(parents=True,exist_ok=True); train.to_csv(a.out_dir/'train.csv',index=False,encoding='utf-8-sig'); test.to_csv(a.out_dir/'test.csv',index=False,encoding='utf-8-sig')
    info={'test_session':a.test_session,'n_train':len(train),'n_test':len(test),'train_sessions':sorted(train.session.unique().tolist()),'test_sessions':sorted(test.session.unique().tolist()),'note':'Validation is created later by build_audio_aug_manifests.py.'}; (a.out_dir/'split_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8'); print(info)
if __name__=='__main__': main()
