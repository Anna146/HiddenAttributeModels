#!/usr/bin/env python
import gzip
import json
import re
import sys

patterns = [
    (re.compile("#[\w]*"), " "),
    (re.compile("[\w]*@[\w]*"), " "),
    (re.compile(r"https?:\/\/.*[\r\n]*"), ""),
    (re.compile("([^\s\w']|_)+|\d|\t|\n"), " "),
    (re.compile("\s+"), " "),
    (re.compile("\.+"), "."),
]

ids = {}
for line in gzip.open("reddit_ham_postids.txt.gz"):
    predicate, uid, label, postid = line.strip().split("\t")
    ids[postid] = ",".join([predicate, uid, label])

for line in sys.stdin:
    try:
        js = json.loads(line)
    except:
        continue
    if not isinstance(js, dict):
        continue

    if "id" not in js or js["id"] not in ids:
        continue

    txt = (js.get("selftext", "") + " " + js.get("body", "") + " " + js.get("title", "")).lower()

    for pattern, substitution in patterns:
        txt = pattern.sub(substitution, txt)

    post_len = len([x for x in txt.split(" ") if len(x.strip().strip(".")) > 1])
    if post_len > 100 or post_len < 10:
        continue

    print("%s\t%s" % (ids[js["id"]], txt))
