from datasets import load_dataset  
from pathlib import Path 

RAW = Path ( "data/raw" ) ; RAW.mkdir ( parents = True, exist_ok = True ) 
# what does this thing mean, i understand that it is calling some fundtion path and creates an object and assigns the label RAW to its nam . and then form it we call some kind of method that sets something, perhaper this is a dict or someting. 

def get ( repo, name=None, conds = None ) : 
    try : 
        ds = load_dataset ( repo, name ) if name else load_dataset  ( repo ) 
        # so we assign the value of load_dataset ( repo, name ) if the name is present, and if not then we just load the one of the repo ( what is the repo ? ) 
        print ( f"OK {repo}{'/' + name  if name else ''} : {ds}" ) 
        # this is just printing something like if the name is present then we do something like OK reponame/ (name of the thing or somethign ( what does name mean) 
        return ds 
    except Exception as e : 
        print ( f"FAIL {repo} : {e}" ) 

# so this function : loads the dataset and returns it. wait do WE give it a nam  hmmm. this is itneresting. so then if there is an error then it returns something the fail and the repo and the exceptionname for this. 


raid = get ( "liamdugan/raid" ) 
semeval = get ( "YAIFS/semEval2024-Task8" ) 
hc3 = get ( "Hello-SimpleAI/HC3" ) 
m4 = get ( "nebula-program/M4") 
# so then for thsi we call the function that we just made, and we put in the reponame. 

wiki = get ("Salesforce/wikitext", "wikitext-103-raw-v1" ) 
owt = get ("Skylion007/openwebtext", "plain_text") 
# so for this we are fetching it, but we also have the name specified or something. hmmm so for this we have the load_dataset thing. so the load_dataset can either take something that has a repo and a name or something ( i assume that it is a function that ether takes a repo and a name or ust arepo, and if it just takes a repo then it means that the dataset is the repo, and if it takes the name, then it could be some monorepo that contains lots and lots of other smaller dataset repos or something. 

raid.save_to_disk ( RAW / "raid" ) 
semeval.save_to_disk ( RAW / "semeval2024-task8" ) 
hc3.save_to_disk ( RAW/ "hc3" ) 
wiki.save_to_disk ( RAW / "wikitext103" ) 



