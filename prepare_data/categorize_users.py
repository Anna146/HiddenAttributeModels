from collections import defaultdict
import random
import sys
import os

################# Check if the post was used for labeling ###########################################################

syn_dict = eval(open("data/prof_synonyms.txt", "r").read())
syn_list = list(syn_dict.keys()) + sum(syn_dict.values(), [])
male_list = ["man", "male", "boy", "husband", "father", "brother"]
female_list = ["woman", "female", "girl", "lady", "wife", "mother", "sister"]
married_list = ["married", "engaged", "dating", "boyfriend", "spouse", "girlfriend", "fiancee", "lover", "partner", "wife", "husband"]
unmarried_list = ["single", "divorsed", "widow", "spouseless", "celibate", "unwed", "fancy-free"]


def is_iama(line, predicate):
    if predicate == "profession":
        impos = min([line.find(p) for p in ["i am", "i'm"] if p in line] + [1_000_000_000])
        if impos != 1_000_000_000:
            for itm in syn_list:
                if line.find(" " + itm + " ", impos + 4, impos + len(itm) + 8) != -1:
                    return True
        return False
    elif predicate == "gender":
        iam_pos = min([line.find(p) for p in ["i am a", "i'm a"] if p in line] + [sys.maxsize])
        if iam_pos != sys.maxsize:
            for itm in male_list + female_list:
                if line.find(" " + itm + " ", iam_pos + 3, iam_pos + len(itm) + 12) != -1:
                    return True
        return False
    elif predicate == "age":
        ## check "born in"
        born_pos = line.find("i was born in ")
        if born_pos != -1:
            if_pos = line.find("if ", min(0, born_pos - 20, born_pos))
            if if_pos == -1:
                born_year = "".join(x for x in line[born_pos + 9 : born_pos + 20].split(" ") if x.isdigit())
                try:
                    born_year = int(born_year)
                    if born_year > 1910 and born_year < 2015:
                        return True
                except:
                    pass
        ## check "years old"
        iam_pos = min([line.find(p) for p in ["i am", "i'm"] if p in line] + [sys.maxsize])
        if iam_pos != sys.maxsize:
            years_pos = line.find("years old", iam_pos + 4, iam_pos + 20)
            if years_pos != -1:
                curr_age = "".join(x for x in line[iam_pos:years_pos].split(" ") if x.isdigit())
                try:
                    curr_age = int(curr_age)
                    if curr_age > 5 and curr_age < 100:
                        return True
                except:
                    pass
        return False
    elif predicate == "family":
        iam_pos = min([line.find(p) for p in ["i am ", "i have a"] if p in line] + [sys.maxsize])
        iam_not_pos = min([line.find(p) for p in ["i am not", "i have no"] if p in line] + [sys.maxsize])
        if (iam_not_pos != sys.maxsize) or (iam_pos != sys.maxsize):
            for itm in married_list + unmarried_list:
                if line.find(" " + itm + " ", iam_pos + 3, iam_pos + len(itm) + 12) != -1:
                    return True
        return False
    else:
        raise Exception("Unknown predicate")


#####################################  Parameters    ############################################################

age_map = {(0, 13): "child", (14, 23): "teenager", (24, 45): "adult", (46, 65): "middle-aged", (66, 100): "senior"}
black_list = ["assistant", "tv presenter", "explorer", "politician", "detective", "movie director", "housewife", "activist"]
personals = ["I", "you", "he", "she", "we", "they", "me", "you", "him", "her", "us", "them", "my", "your", "our", "their", "his", "her"]
threshold = 100  # max users per predicate
threshold_total = 6000  # max users

#################################################################################################################


def age_to_label(age):
    age = int(age)
    for rang, age_name in age_map.items():
        if rang[0] < age <= rang[1]:
            return age_name
    raise Exception("bad age")


def classify_users(predicate, inp_file):
    test_file = "data/reddit/whitelists/test_" + predicate + ".txt"
    train_dir = "data/reddit/whitelists/train_" + predicate + "/"

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    prof_dict = defaultdict(lambda: dict())
    cnt = 0
    all_users = []

    with open(inp_file, "r") as f_in:
        curr_auth = ""
        curr_prep = 0
        curr_tot = 0
        curr_prof = ""
        has_iama = False
        for line in f_in:
            try:
                keys, txt = line.rstrip().split("\t")
                this_pred, auth, prof = keys.split(",")
                if this_pred != predicate:
                    continue
                if predicate == "age":
                    prof = age_to_label(prof)
                has_iama = has_iama or is_iama(txt, predicate)
            except:
                continue
            if auth != curr_auth:
                cnt += 1
                if curr_auth != "" and has_iama:
                    prof_dict[curr_prof][curr_auth] = curr_prep * 1.0 / curr_tot
                    all_users.append(curr_auth)
                curr_prof = prof
                curr_prep = 0
                curr_tot = 0
                curr_auth = auth
                has_iama = False
            else:
                curr_prep += int(any([p in txt.split(" ") for p in personals]))
                curr_tot += 1

    random.shuffle(all_users)
    all_users = all_users[:threshold_total]

    tot_ct = 0
    with open(test_file, "w") as f_out:
        for proff, auths in prof_dict.items():
            auths = dict(x for x in auths.items() if x[0] in all_users)
            if predicate == "profession" and proff in black_list:
                continue
            with open(train_dir + proff + ".txt", "w") as f_train:
                if len(auths) < threshold:
                    to_write = list(auths.keys())
                else:
                    auth_sorted = sorted(auths.items(), key=lambda x: x[1], reverse=True)[:threshold]
                    to_write = [x[0] for x in auth_sorted]
                for au in to_write:
                    if random.random() > 0.09:
                        f_train.write(au + "\n")
                    else:
                        f_out.write(au + "\n")
                tot_ct += len(to_write)


for predicate in ["profession", "age", "gender", "family"]:
    print("processing", predicate)
    classify_users(predicate, inp_file="data/raw/posts.txt")
