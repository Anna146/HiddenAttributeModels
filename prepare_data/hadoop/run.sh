#!/bin/bash
source /home/d5hadoop/new/d5hadoop_env.sh
hadoop fs -put data/raw/postids.txt.gz reddit_ham_postids.txt.gz

hadoop jar $CDH_MR2_HOME/hadoop-streaming.jar -file prepare_data/hadoop/mapper.py -mapper prepare_data/hadoop/mapper.py -file prepare_data/hadoop/reducer.py -reducer prepare_data/hadoop/reducer.py -cacheFile 'reddit_ham_postids.txt.gz#reddit_ham_postids.txt.gz' -input 'reddit_*/*' -output reddit_ham_data && \
hadoop fs -cat reddit_ham_data/part-00000 > data/raw/posts.txt && \
echo success


