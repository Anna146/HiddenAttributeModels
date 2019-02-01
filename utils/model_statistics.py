from collections import defaultdict
import numpy as np
import random
from sklearn.metrics import roc_auc_score
random.seed(33)

DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


def chunks(l, n):
    n = max(1, n)
    return list(l[i:i+n] for i in range(0, len(l), n))


def load_predicates(filepath):
    predicates = []
    with open(filepath, "r") as f_trip:
        for line in f_trip:
            predicates.append(line.lower().strip())
    return dict((x[0], x[1]) for x in enumerate(predicates)), predicates


def compute_MRR_per_prof(filepath, offset = 0):
    prof_dict = defaultdict(lambda: [0.0,0])
    predicates, _ = load_predicates("/home/tigunova/PycharmProjects/movie_experiment/raw_data/profession_list.txt")
    big_count = 0
    big_MRR = 0
    with open(filepath, "r") as f_in:
        for line in f_in:
            fields = line.split("\t")
            answ = fields[0 + offset]
            score = 0
            for i in enumerate(fields[1 + offset:]):
                pred_num = i[1].split(",")[0][1:]
                if pred_num == answ:
                    score = 1.0 / (i[0] + 1)
            prof_dict[answ][0] += score
            prof_dict[answ][1] += 1
        for prof, stats in prof_dict.items():
            big_count += 1
            big_MRR += float(stats[0] / stats[1])
    print("MACR0 MRR " + str(big_MRR / big_count))
    return big_MRR / big_count


def compute_MRR_per_character(filepath, confusion_file = "/data/confusion_matrix.txt", return_rrs = False):
    old_answ = ""
    curr_dict = defaultdict(float)
    big_sum = 0
    big_count = 0
    # for confusion matrix
    y_true = []
    y_pred = []
    char_RRs = dict()
    char_hits = dict()

    with open(filepath, "r") as f_in, open(confusion_file, "a") as f_out:
        curr_char = "-1"
        for line in f_in:
            fields = line.strip().split("\t")
            character = fields[0]
            curr_answ = fields[1]
            if old_answ == "":
                old_answ = curr_answ
            if character != curr_char:
                if len(curr_dict) != 0:
                    predictions = sorted(curr_dict.items(), key=lambda x: x[1], reverse=True)
                    y_true.append(int(old_answ))
                    y_pred.append(int(predictions[0][0]))
                    y_true.append(int(old_answ))
                    y_pred.append(int(predictions[1][0]))
                    for pred in enumerate(predictions):
                        if pred[0] == 0:
                            char_hits[curr_char] = 1 if old_answ == pred[1][0] else 0
                        if old_answ == pred[1][0]:
                            big_sum += 1.0 / (pred[0] + 1)
                            char_RRs[character] = 1.0 / (pred[0] + 1)
                            break
                    big_count += 1
                curr_char = character
                curr_dict = defaultdict(float)
            for i in enumerate(fields[2:]):
                pred_num = i[1].split(",")[0][1:]
                curr_dict[pred_num] += float(i[1].split(",")[1][:-1])
            old_answ = curr_answ
        if len(curr_dict) > 0:
            predictions = sorted(curr_dict.items(), key=lambda x: x[1], reverse=True)
            y_true.append(int(old_answ))
            y_pred.append(int(predictions[0][0]))
            y_true.append(int(old_answ))
            y_pred.append(int(predictions[1][0]))
            for pred in enumerate(predictions):
                if pred[0] == 0:
                    char_hits[curr_char] = 1 if old_answ == pred[1][0] else 0
                if old_answ == pred[1][0]:
                    big_sum += 1.0 / (pred[0] + 1)
                    char_RRs[curr_char] = 1.0 / (pred[0] + 1)
                    break
            big_count += 1
        f_out.write(" ".join(str(x) for x in y_true) + "\n")
        f_out.write(" ".join(str(x) for x in y_pred) + "\n")
    if return_rrs:
        return big_sum / big_count, char_RRs, char_hits
    return big_sum / big_count


def compute_auroc(filepath, offset = 0):
    y_true = []
    y_probs = []
    with open(filepath, "r") as f_in:
        for line in f_in:
            fields = line.split("\t")
            cur_probs = [0 for x in range(len(fields[1 + offset:]))]
            cur_true = [0 for x in range(len(fields[1 + offset:]))]
            answ = int(fields[0 + offset])
            cur_true[answ] = 1
            for i in fields[1 + offset:]:
                pred_num = int(i.split(",")[0][1:])
                cur_probs[pred_num] = float(i.split(",")[1][:-1].strip(")"))
            y_probs.append(cur_probs)
            y_true.append(cur_true)
    return (roc_auc_score(np.array(y_true).transpose(), np.array(y_probs).transpose(), average="micro"), roc_auc_score(np.array(y_true).transpose(), np.array(y_probs).transpose(), average="macro"))


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from scipy import interp

def compute_multi_auroc(filepath, offset = 0):
    y_true = []
    y_probs = []
    class_num = 0
    with open(filepath, "r") as f_in:
        for line in f_in:
            fields = line.split("\t")
            cur_probs = [0 for x in range(len(fields[1 + offset:]))]
            class_num = len(fields[1 + offset:])
            answ = int(fields[0 + offset])
            for i in fields[1 + offset:]:
                pred_num = int(i.split(",")[0][1:])
                cur_probs[pred_num] = float(i.split(",")[1][:-1].strip(")"))
            y_probs.append(cur_probs)
            y_true.append(answ)
    y_probs = np.array(y_probs)

    # Binarize the output
    y_true = label_binarize(y_true, classes=[i for i in range(class_num)])
    n_classes = y_true.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        if np.any(y_true[:, i]):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr.keys()]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in fpr.keys():
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(fpr)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc["macro"]


from sklearn.metrics import accuracy_score

def compute_accuracy(filepath, offset=0):
    big_count = 0
    y_true = []
    y_pred = []
    with open(filepath, "r") as f_in:
        for line in f_in:
            fields = line.split("\t")
            answ = fields[0 + offset]
            big_count += 1
            y_true.append(int(answ))
            y_pred.append(int(fields[1+offset].split(",")[0][1:]))
    return accuracy_score(y_true, y_pred)

from scipy.stats import ttest_rel
import statsmodels.api as sm

def significance_MRR(file1, file2):
    _, d1, h1 = compute_MRR_per_character(file1, "/home/tigunova/out.txt", True)
    _, d2, h2 = compute_MRR_per_character(file2, "/home/tigunova/out.txt", True)
    inter_keys = set(d1.keys()).intersection(set(d2.keys()))
    y1 = []
    y2 = []
    for k, v in d1.items():
        if k in inter_keys:
            y1.append(v)
            y2.append(d2[k])
    # construct table
    ww = len([x for x in inter_keys if h1[x] == 0 and h2[x] == 0])
    wc = len([x for x in inter_keys if h1[x] == 1 and h2[x] == 0])
    cw = len([x for x in inter_keys if h1[x] == 0 and h2[x] == 1])
    cc = len([x for x in inter_keys if h1[x] == 1 and h2[x] == 1])
    table = [[ww, wc], [cw, cc]]
    print("AUROC significance " + str(sm.stats.mcnemar(table, exact=True, correction=True)))
    print("MRR significance " + str(ttest_rel(y1, y2)))
    return ttest_rel(y1, y2)


def compute_stats_dump(test_file, dump_dir):
    confusion_file = "output_data/confusion_matrix.txt"
    macro = compute_MRR_per_prof(test_file, 1)
    #macro = compute_MFR_per_prof(test_file, 1)
    mrr_character = compute_MRR_per_character(test_file, confusion_file=confusion_file)
    print("Micro MRR: " + str(mrr_character))
    print("Auroc " + str(compute_auroc(test_file, 1)))
    print("MACRO Auroc " + str(compute_multi_auroc(test_file, 1)))
    print("accuracy " + str(compute_accuracy(test_file, 1)))
    #print("ndcg " + str(compute_ndcg_macro(test_file, 1)))
    with open(dump_dir, "w") as dump_file:
        dump_file.write("Character MRR: " + str(mrr_character) + "\n")
        dump_file.write("Macro MRR: " + str(macro) + "\n")
        dump_file.write("Auroc " + str(compute_auroc(test_file, 1)))




#################################### DATA LOADERS  #################################################################

# expecting format [char_id, true_label, [word_indexes], [sentences]]
def indexes_gen(filepath, batch_size, word_num, utter_num, dtype=int, k = 0, mode = "test", folds = 1):
    with open(filepath, "r") as f_in:
        counter = 1
        batch_X = np.empty(shape=(batch_size, utter_num * word_num), dtype=dtype)
        while True:
            batch_y = []
            batch_sentences = []
            batch_charids = []
            i = 0
            while i < batch_size:
                data = f_in.readline().strip()
                if data == "":
                    raise StopIteration()
                rel = (counter - k) % folds
                if (mode == "train" and rel != 0) or (mode == "test" and rel == 0):
                    data = data.split(",")
                    batch_X[i] = data[2:2 + word_num * utter_num]
                    batch_y.append(int(data[1]))
                    batch_charids.append(data[0])
                    batch_sentences.append(data[2 + word_num * utter_num:])
                    i += 1
                counter += 1
            yield np.array(batch_X), np.array(batch_y), np.array(batch_charids), np.array(batch_sentences)

# expecting format [char_id, true_label, [words], [sentences]]
def words_gen(filepath, batch_size, word_num, utter_num, dtype=int, k = 0, mode = "test", folds = 1):
    with open(filepath, "r") as f_in:
        counter = 1
        while True:
            batch_X = []
            batch_y = []
            batch_sentences = []
            batch_charids = []
            i = 0
            while i < batch_size:
                data = f_in.readline().strip()
                if data == "":
                    raise StopIteration()
                rel = (counter - k) % folds
                if (mode == "train" and rel != 0) or (mode == "test" and rel == 0):
                    data = data.split(",")
                    slice = data[2:2 + word_num * utter_num]#batch_X = np.append(batch_X, np.array([data[2:2+word_num*utter_num]]).astype(int), axis=0)
                    slice = chunks(slice, word_num)
                    slice = [list(filter(lambda a: a != DEFAULT_PADDING_TOKEN, x)) for x in slice]
                    batch_X.extend(slice)
                    batch_y.append(int(data[1]))
                    batch_charids.append(data[0])
                    batch_sentences.append(data[2 + word_num * utter_num:])
                    i += 1
                counter += 1
            yield batch_X, np.array(batch_y), np.array(batch_charids), np.array(batch_sentences)
