import re
from nltk.corpus import stopwords
import csv
import sys
import os


DEFAULT_PADDING_TOKEN = "@@PADDING@@"

sys.path.insert(0, "utils")
from model_statistics import *

vocab_path = "data/vocab.txt"
with open(vocab_path, "r") as f:
    vocab = dict((line.strip(), index) for index, line in enumerate(f.readlines()))
unkn_ind = len(vocab)

stop_words = set(stopwords.words("english"))


def text_to_bow(inp):
    inp = inp.lower().strip().split()
    inp = [re.sub(r"[^a-zA-Z]", "", x) for x in inp]
    return [x for x in inp if len(x) > 1 and x not in stop_words]


def embed_user(inp_label, train_file, act_file, batch_size=1, character_id=0, in_words=False):

    max_utterance_len = 100
    max_char_len = 100

    with open(train_file, "a") as f_train:
        # define csv writers
        writer_train = csv.writer(f_train, lineterminator="\n")
        loc_lens = []

        character_words = []
        utt_count = 0
        curr_utt = ""
        sent_array = []
        for line in act_file:
            if line == None or len(line) == 0:
                continue
            line = line.strip()
            if line[0].isalpha():
                curr_words = line.lower().strip()
                utt_count += 1
                curr_utt += " " + curr_words
                if utt_count == batch_size:
                    sent_array.append(curr_utt.replace(",", ""))
                    curr_utt = text_to_bow(curr_utt.strip())
                    curr_utt = curr_utt[:max_utterance_len]
                    if not in_words:
                        curr_utt = [vocab[x] if x in vocab else unkn_ind for x in curr_utt]
                        while curr_utt[0] == unkn_ind:
                            curr_utt = curr_utt[1:]
                        assert len(curr_utt) > 0
                        loc_lens.append(len(curr_utt))
                        if len(curr_utt) < max_utterance_len:
                            curr_utt.extend([unkn_ind] * (max_utterance_len - len(curr_utt)))
                    else:
                        curr_utt.extend([DEFAULT_PADDING_TOKEN] * (max_utterance_len - len(curr_utt)))
                    character_words.append(curr_utt)
                    curr_utt = ""
                    utt_count = 0

        r_len = len(character_words)
        if r_len > 10:
            character_words = character_words[:max_char_len]
            if max_char_len > r_len:
                if not in_words:
                    character_words.extend([[unkn_ind] * max_utterance_len] * (max_char_len - r_len))
                else:
                    character_words.extend([[DEFAULT_PADDING_TOKEN] * max_utterance_len] * (max_char_len - r_len))
            writer_train.writerow([character_id, inp_label] + [item for sublist in character_words for item in sublist] + sent_array)
    return r_len


################# Check if the post was used for labeling ###########################################################

syn_dict = eval(open("data/prof_synonyms.txt", "r").read())
syn_list = list(syn_dict.keys()) + sum(syn_dict.values(), [])
male_list = ["man", "male", "boy", "husband", "father", "brother"]
female_list = ["woman", "female", "girl", "lady", "wife", "mother", "sister"]
married_list = ["married", "engaged", "dating", "boyfriend", "spouse", "girlfriend", "fiancee", "lover", "partner", "wife", "husband"]
unmarried_list = ["single", "divorsed", "widow", "spouseless", "celibate", "unwed", "fancy-free"]


def is_iama(line, predicate):
    if predicate == "profession":
        impos = min([line.find(p) for p in ["i am", "i'm"] if p in line] + [sys.maxsize])
        if impos != sys.maxsize:
            for itm in syn_list:
                if line.find(" " + itm + " ", impos + 4, impos + len(itm) + 8) != -1:
                    return itm
    elif predicate == "gender":
        iam_pos = min([line.find(p) for p in ["i am a", "i'm a"] if p in line] + [sys.maxsize])
        if iam_pos != sys.maxsize:
            for itm in male_list + female_list:
                if line.find(" " + itm + " ", iam_pos + 3, iam_pos + len(itm) + 12) != -1:
                    return itm
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
                        return str(born_year)
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
                        return str(curr_age)
                except:
                    pass
    elif predicate == "family":
        iam_pos = min([line.find(p) for p in ["i am ", "i have a"] if p in line] + [sys.maxsize])
        iam_not_pos = min([line.find(p) for p in ["i am not", "i have no"] if p in line] + [sys.maxsize])
        if (iam_not_pos != sys.maxsize) or (iam_pos != sys.maxsize):
            for itm in married_list + unmarried_list:
                if line.find(" " + itm + " ", iam_pos + 3, iam_pos + len(itm) + 12) != -1:
                    return itm
    else:
        raise Exception("Unknown predicate")
    return ""


######################### Parameters  ######################################################

age_map = {(0, 13): "child", (14, 23): "teenager", (24, 45): "adult", (46, 65): "middle-aged", (66, 100): "senior"}
predicate_file = "data/profession_list.txt"


#####################################################################################


def age_to_label(age):
    age = int(age)
    for rang, age_name in age_map.items():
        if rang[0] < age <= rang[1]:
            return age_name
    raise Exception("bad age")


def create_sample(predicate, inp_file):
    in_folder = "data/reddit/whitelists/train_" + predicate + "/"
    in_test = "data/reddit/whitelists/test_" + predicate + ".txt"

    train_folder = "data/reddit/train_" + predicate + "/"
    test_file = "data/reddit/test_" + predicate + ".txt"

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if predicate == "profession":
        predicate_dict, predicate_list = load_predicates(predicate_file)
    if predicate == "age":
        predicate_list = ["child", "teenager", "adult", "middle-aged", "senior"]
    if predicate == "gender":
        predicate_list = ["f", "m"]
    if predicate == "family":
        predicate_list = ["y", "n"]

    # Clear contents
    with open(test_file, "w") as f:
        pass
    for f_name in os.listdir(train_folder):
        with open(train_folder + f_name, "w"):
            pass

    # Load whitelists
    category_whitelists = dict()
    for f_name in os.listdir(in_folder):
        with open(in_folder + f_name, "r") as f_list:
            category_whitelists[f_name[:-4]] = set(line.strip() for line in f_list)
    with open(in_test, "r") as f_list:
        test_whitelist = set(line.strip() for line in f_list)

    with open(inp_file, "r") as f_in:
        texts = []
        curr_char = ""
        curr_prof = -1
        curr_prof_name = ""
        char_count = 0
        has_ims = None
        for line in f_in:
            # hack to convert each line from the hadoop output (one file with all predicates)
            # to the expected input (3 columns + predicate specified by file)
            keys, txt = line.rstrip().split("\t")
            this_pred, auth, prof = keys.split(",")
            if this_pred != predicate:
                continue
            line = "%s\t%s\t%s\n" % (auth, prof, txt)

            iama = is_iama(line, predicate)
            if iama != "":
                has_ims = line.strip().split("\t")[2]
                continue
            line = line.strip().split("\t")
            if len(line) < 3:
                continue
            if line[0] == curr_char:
                if predicate == "age":
                    try:
                        line[1] = age_to_label(line[1])
                    except:
                        continue
                texts.append(line[2])
                if curr_prof == -1 and line[1] in predicate_list:
                    curr_prof_name = line[1]
                    curr_prof = predicate_list.index(line[1])
            else:
                if curr_char != "" and curr_prof != -1 and curr_prof_name in category_whitelists:
                    res_file = None
                    if curr_char in test_whitelist and has_ims != None:
                        res_file = test_file
                    elif curr_char in category_whitelists[curr_prof_name]:
                        res_file = train_folder + curr_prof_name + ".txt"
                    if res_file != None:
                        embed_user(inp_label=curr_prof, train_file=res_file, act_file=texts, batch_size=4, character_id=char_count, in_words=False)
                        char_count += 1
                        has_ims = None
                curr_char = line[0]
                try:
                    curr_prof_name = line[1]
                    curr_prof = predicate_list.index(line[1])
                except:
                    curr_prof = -1
                texts = [line[2]]


for predicate in ["profession", "age", "gender", "family"]:
    print("processing", predicate)
    create_sample(predicate, inp_file="data/raw/posts.txt")
