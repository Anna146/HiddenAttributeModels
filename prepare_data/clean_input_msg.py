import re

input = "data/raw/professions.txt"
output = "data/raw/professions_txt.txt"

with open(input, "r") as f_in, open(output, "w") as f_out:
    for line in f_in:
        line = line.strip().split("\t")
        txt = line[-1].lower()

        pattern = re.compile('#[\w]*')
        txt = pattern.sub(' ', txt)
        pattern = re.compile('[\w]*@[\w]*')
        txt = pattern.sub(' ', txt)
        txt = re.sub(r'https?:\/\/.*[\r\n]*', '', txt)

        pattern = re.compile('([^\s\w\']|_)+|\d|\t|\n')
        txt = pattern.sub(' ', txt)
        pattern = re.compile('\s+')
        txt = pattern.sub(" ", txt)
        pattern = re.compile('\.+')
        txt = pattern.sub(".", txt)

        post_len = len([x for x in txt.split(" ") if len(x.strip().strip(".")) > 1])
        if post_len > 100 or post_len < 10:
            continue
        f_out.write("%s\t%s\t%s\n" % (line[0], line[1], txt))
