# add some readme don't be lazy

# Data format

The dataset for each attribute consists of a training folder (/train_predicate/) and a text file (/test_predicate.txt). Inside training folder are the training instances in separate files for each attribute value, in test file the instances are from all values.

The format of the train and test file is the following:
\[instance_id,correct_label_id,comma-separated word indixes,plain text of the instance\]

The indexing of labels is according to the indexing of lines in predicae_list.txt file
The indexing of words is according to the /vocabulary.txt file and their respective embeddings are in /weights.txt file

All word indixes arrays are padded, the padding index is the last word + 1 in the vocabulary. The word indexes arrays are 40 * 40 size for MovieChatt and 100 * 100 for Reddit and PersonaChat (utterance_len * subject_len)
