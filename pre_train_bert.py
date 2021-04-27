import tokenizers
import os
import sys
import json
import nltk
import random
import tokenizers
import logging
import tensorflow as tf
import sentencepiece as spm

from glob import glob
from google.colab import auth, drive
from tensorflow.keras.utils import Progbar

sys.path.append("bert")

from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

auth.authenticate_user()
  
# configure logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s :  %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
log.handlers = [sh]

if 'COLAB_TPU_ADDR' in os.environ:
  log.info("Using TPU runtime")
  USE_TPU = True
  TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']

  with tf.Session(TPU_ADDRESS) as session:
    log.info('TPU address is ' + TPU_ADDRESS)
    # Upload credentials to TPU.
    with open('/content/adc.json', 'r') as f:
      auth_info = json.load(f)
    tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
    
else:
  log.warning('Not connected to TPU runtime')
  USE_TPU = False

import nltk
regex_tokenizer = nltk.RegexpTokenizer("\w+")

def normalize_text(text):
  # lowercase text
  text = str(text).lower()
  # remove punktuation symbols
  text = " ".join(regex_tokenizer.tokenize(text))
  return text

def count_lines(filename):
  count = 0
  with open(filename) as fi:
    for line in fi:
      count += 1
  return count

RAW_DATA_FPATH = "myfile.txt"
PRC_DATA_FPATH = "proc_dataset.txt" 

# apply normalization to the dataset
# this will take a minute or two

total_lines = count_lines(RAW_DATA_FPATH)
bar = Progbar(total_lines)

with open(RAW_DATA_FPATH) as fi:
  with open(PRC_DATA_FPATH, "w") as fo:
    for l in fi:
      fo.write(normalize_text(l)+"\n")
      bar.add(1)

import sentencepiece as spm
PRC_DATA_FPATH = "proc_dataset.txt"

MODEL_PREFIX = "tokenizer" 
VOC_SIZE = 32000 
SUBSAMPLE_SIZE = 1280000 
NUM_PLACEHOLDERS = 256 

SPM_COMMAND = ('--input={} --model_prefix={} '
               '--vocab_size={} --input_sentence_size={} '
               '--shuffle_input_sentence=true ' 
               '--bos_id=-1 --eos_id=-1').format(
               PRC_DATA_FPATH, MODEL_PREFIX, 
               VOC_SIZE - NUM_PLACEHOLDERS, SUBSAMPLE_SIZE)

spm.SentencePieceTrainer.Train(SPM_COMMAND)

def read_sentencepiece_vocab(filepath):
  voc = []
  with open(filepath) as fi:
    for line in fi:
      voc.append(line.split("\t")[0])
  voc = voc[1:]
  return voc

snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))

def parse_sentencepiece_token(token):
    if token.startswith('_'):
        return token[1:]
    else:
        return "##" + token

bert_vocab = list(map(parse_sentencepiece_token,snt_vocab))
token_sub = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
bert_vocab = token_sub + bert_vocab
bert_vocab += ["[UNUSED_{}]".format(i) for i in range(VOC_SIZE - len(bert_vocab))]
VOC_FNAME = "vocab.txt" 

with open(VOC_FNAME, "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")

bert_tokenizer = tokenization.FullTokenizer(VOC_FNAME)

!mkdir ./shards
!split -a 4 -l 256000 -d $PRC_DATA_FPATH ./shards/shard_

MAX_SEQ_LENGTH = 128 
MASKED_LM_PROB = 0.15 
MAX_PREDICTIONS = 20 
DO_LOWER_CASE = True 
PROCESSES = 2 
PRETRAINING_DIR = "pretraining_data"

XARGS_CMD = ("ls ./shards/ | "
             "xargs -n 1 -P {} -I{} "
             "python3 bert/create_pretraining_data.py "
             "--input_file=./shards/{} "
             "--output_file={}/{}.tfrecord "
             "--vocab_file={} "
             "--do_lower_case={} "
             "--max_predictions_per_seq={} "
             "--max_seq_length={} "
             "--masked_lm_prob={} "
             "--random_seed=1234 "
             "--dupe_factor=5")

XARGS_CMD = XARGS_CMD.format(PROCESSES, '{}', '{}', PRETRAINING_DIR, '{}', 
                             VOC_FNAME, DO_LOWER_CASE, 
                             MAX_PREDICTIONS, MAX_SEQ_LENGTH, MASKED_LM_PROB)

tf.gfile.MkDir(PRETRAINING_DIR)
!$XARGS_CMD

BUCKET_NAME = "another_qa_bucket" 
MODEL_DIR = "bert_model_wiki" 
tf.gfile.MkDir(MODEL_DIR)

if not BUCKET_NAME:
  log.warning("WARNING: BUCKET_NAME is not set. "
              "You will not be able to train the model.")

# use this for BERT-base

bert_base_config = {
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": VOC_SIZE
}

with open("{}/bert_config.json".format(MODEL_DIR), "w") as fo:
  json.dump(bert_base_config, fo, indent=2)
  
with open("{}/{}".format(MODEL_DIR, VOC_FNAME), "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")

if BUCKET_NAME:
  !gsutil -m cp -r $MODEL_DIR $PRETRAINING_DIR gs://$BUCKET_NAME

!python bert/run_pretraining.py \
    --input_file gs://another_qa_bucket/pretraining_data/*.tfrecord \
    --output_dir gs://another_qa_bucket/bert_model_wiki/model/ \
    --do_train True \
    --do_eval True \
    --bert_config_file gs://another_qa_bucket/bert_model_wiki/bert_config.json \
    --train_batch_size 32 \
    --max_seq_length 128 \
    --masked_lm_prob 0.15\
    --max_predictions_per_seq 20 \
    --num_train_steps 1000000 \
    --num_warmup_steps 10 \
    --learning_rate 2e-5 \
    --use_tpu True \
    --tpu_name $TPU_NAME


!python bert/run_squad.py \
  --vocab_file gs://another_qa_bucket/bert_model_wiki/vocab.txt \
  --bert_config_file gs://another_qa_bucket/bert_model_wiki/bert_config.json \
  --init_checkpoint gs://another_qa_bucket/bert_model_wiki/model/model.ckpt-1000000 \
  --do_train=True \
  --train_file /content/train-v1.1.json \
  --do_predict=True \
  --predict_file /content/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --doc_stride 128 \
  --output_dir gs://another_qa_bucket/bert_model_wiki/

!python /content/bi-att-flow/squad/evaluate-v1.1.py /content/dev-v1.1.json /content/bert_model_wiki_predictions.json
