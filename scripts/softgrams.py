import argparse, json 
from pathlib import Path 
import re 

def splitclauses ( text  : str ) -> list[tuple[str, int, int ] ]: #what the fuck is this thing, ive never seen things like ths, is it just specifying the types or something, is it not possible for us just to leave it out or osmething. 

    if not text : 
        return [] 
    
    # so this is, if it is not of type string ? that is the only logical operation that can be done onto this unassigned var. and hten it is alos specifying that it is a list of at ype of a string int it, or it is a list of typlse that hav string int it ( is this machine versus nonmachien ( i thought that we needed machinegen, aiassisted and human or something, so surely that should be three int values or somethign ) 
    
    breakre = re.compile ( r"[.!?;:]+") 

    clauses = [] 
    start = 0 
    for m in breakre.finditer ( text ) : 
        end = m.end () 
        clauses.append (( text[start:end], start, end ))  
        # for m in breakre.finditer ( text ) i.e it is calling a method on the string and then g
        start = end 

    if start < len ( text ) : 
        clauses.append (( text [start:], start, len ( text ) )) 
            # what is this doing: it is iterating throuhg thes things. so i am guessing taht breakre = re.compile does something like greak them apart using these as the markers or something ? these are something like the clause markers. 

            # and then we have the clause list that starts empty, and start = 0 and hten for m in breakre.finditer( text <) what does this thing do ? and hten end = m.end () this is FOR every single one of the clauses in the thing right? and then we append hmmmmm we append to clauses something like the text bwetween tehse two points or something with the hmmm wait append takes the of thing, the starting index, and the step right ? how does thsi thing work, and then we make the start the end ? oh so this skips to hte next part of the clause or something. and then if the start is less than the length of the text, then we append it or something idke bro
    return clauses 

