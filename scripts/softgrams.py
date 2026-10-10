import argparse, json 
from pathlib import Path 
import re 

def splitclauses ( text  : str ) -> list[tuple[str, int, int ] ]:

    if not text : 
        return [] 
    
    
    breakre = re.compile ( r"[.!?;:]+") 

    clauses = [] 
    start = 0 
    for m in breakre.finditer ( text ) : 
        end = m.end () 
        clauses.append (( text[start:end], start, end ))  
        start = end 

    if start < len ( text ) : 
        clauses.append (( text [start:], start, len ( text ) )) 

    return clauses 

