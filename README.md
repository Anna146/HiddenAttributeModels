# Reddit Data
Our Reddit corpus was created using all comments and submissions made between Jan 2006 and October 2018. 

The file data/raw/postids.txt.gz is in the format [predicate \t author_id \t predicate_value \t message_id]. Get the message texts yourself, the text to be taken is 'selftext' + 'body' + 'title' from api's json. You can either crawl only the post ids we used using the Reddit API, or you can obtain all the reddit data from another source, such as https://files.pushshift.io/reddit/

A Hadoop script for automatically extracting the needed messages and cleaning them is available in prepare_data/hadoop/. It expects to find reddit_comments and reddit_submission is in the user's home directory. If you opt to extract the messages yourself rather than using Hadoop, you will need to run prepare_data/clean_input_msg.py to clean the messages' text. 

Then run prepare_data/categorize_users.py to split all users into train files (one for each predicate) and test file.

Then run prepare_data/create_input_samples.py to get the vocabulary indexing of all the messages corresponding to the authors from the split above.

# Data format

The dataset for each attribute consists of a training folder (/train_predicate/) and a text file (/test_predicate.txt). Inside training folder are the training instances in separate files for each attribute value, in test file the instances are from all values.

The format of the train and test file is the following:
\[instance_id,correct_label_id,comma-separated word indixes,plain text of the instance\]

The indexing of labels is according to the indexing of lines in predicae_list.txt file
The indexing of words is according to the /vocabulary.txt file and their respective embeddings are in /weights.txt file

All word indixes arrays are padded, the padding index is the last word + 1 in the vocabulary. The word indexes arrays are 40 * 40 size for MovieChatt and 100 * 100 for Reddit and PersonaChat (utterance_len * subject_len)

# Training and testing
how?
