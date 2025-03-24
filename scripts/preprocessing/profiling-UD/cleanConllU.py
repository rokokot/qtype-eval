import codecs
import sys
import re

def main(fileU):
    InputU=codecs.open(fileU, "r", "utf8")
    out=codecs.open(fileU.rstrip('.conllu')+'_cleaned.conllu', 'w', 'utf-8')
    for l in InputU:
        if l=="\n":
            out.write('\n')
            continue
        lS=l.split("\t")
        if not("." in lS[0]):
            if len(lS) > 3:
                lS[8] = "_"
                l = "\t".join(lS)
            #l = re.sub(r"\d+\.\d+.*?\t", "_\t", l)
            #print l.encode("utf8"),
            out.write(l)

main(sys.argv[1])
