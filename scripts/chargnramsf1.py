from collections import Counter


def make_grams(s: str, n: int = 3) -> Counter:
    s = s.lower()
    if not s:
        return Counter()
    if len(s) < n:
        return Counter([s])
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def f1_from_grams(ga: Counter, gb: Counter) -> float:
    shared = sum((ga & gb).values())
    total = sum(ga.values()) + sum(gb.values())
    return 2.0 * shared / total if total else 0.0 

def charngramf1 ( a : str, b : str, n : int = 3 ) -> float : 
# this takes in two strings and an int of three ( what does this int do ? what is it used for > oh shit like is it saying that uhhhhhh wait is it taking three things like start stop and length or somethin ? what is it doing ? 
    a = a.lower() 
    b = b.lower() 

    if not a or not b : 
        return 0.0 
   # if there are no characters, then it is likely 0 percent human or something. or 0 percent similarity 

    def grams ( s : str ) -> Counter : 
        if len ( s ) < n : 
            return Counter ( [s] ) 
        return Counter ( s [i : i + n ] for i in range ( len ( s ) - n + 1 ) ) 
    # if hte lenght of the string is less than n ? what the fuck, are we doing tokens of size three or somethign ? waht is this thing, if a string is shorter, then it is a gram or something. return the counter of the list containng s hmmm wait is [s] the list of the string s or something. 
    # then if not then it returns the counter of the string frmom the index to the index _ n for i in rnge ( len ( s ) - n + 1 ) ) whihc shows that it should go from the 0 1 2 or somethign so multiples of 3 from index 0 or something which is n - 1 or something hmm i thnk taht is the case
    ga = grams ( a ) 
    gb = grams ( b ) 
    shared = sum (( ga & gb ).values () ) 
    # so this ands the two together or somethign ? doe sthis compare the srings or somethign? and then calls the .values() method on it, and then it takes ths um of the values or something, so im guessing taht it puts the values() between zero and one . 
    # wait it takes the grams or something. 
    # wiat im not sure that i particulaly understand what the grams thing does, not what counter does or something hmm. 

    if shared == 0 : 
        return 0.0 
    return 2.0 * shared / (sum(ga.values()) + sum(gb.values()))  
