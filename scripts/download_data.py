from datasets import load_dataset  
from pathlib import Path 

RAW = Path ( "data/raw" ) ; RAW.mkdir ( parents = True, exist_ok = True ) 

def get ( repo, name=None, conds = None ) : 
    try : 
        ds = load_dataset ( repo, name ) if name else load_dataset  ( repo ) 
        print ( f"OK {repo}{'/' + name  if name else ''} : {ds}" ) 
        return ds 
    except Exception as e : 
        print ( f"FAIL {repo} : {e}" ) 

raid = get ( "liamdugan/raid" ) 
semeval = get ( "YAIFS/semEval2024-Task8" ) 
hc3 = get ( "Hello-SimpleAI/HC3" ) 
m4 = get ( "nebula-program/M4") 

wiki = get ("Salesforce/wikitext", "wikitext-103-raw-v1" ) 
owt = get ("Skylion007/openwebtext", "plain_text") 

raid.save_to_disk ( RAW / "raid" ) 
semeval.save_to_disk ( RAW / "semeval2024-task8" ) 
hc3.save_to_disk ( RAW/ "hc3" ) 
wiki.save_to_disk ( RAW / "wikitext103" ) 

