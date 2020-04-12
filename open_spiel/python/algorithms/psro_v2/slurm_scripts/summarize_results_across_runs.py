"""
This file parse stdout.txt files in the tree rooted at root_result.
It reads out data[keys] (keys given as params) and combine them into a combined tensorflow event file. tfevent file takes too long a time to read, so choose stdout.txt instead
The presumbed file structure is
--root_result
  --exp1
    --stdout.txt
    --nfg
    --log
      --tfevent.file
  --exp2
    --stdout.txt
    --nfg
    --log
      --tfevent.file
  After running this script, a following structure will be created
  --summary
    --exp1_name_abbreviated
      --tfevent.file
    --exp2_name_abbreviated
      --tfevent.file
"""
import os
import shutil
import functools
import tensorflow as tf
from tensorboardX import SummaryWriter
print = functools.partial(print, flush=True)

ROOT_DIR = './root_result'
SUMMARY_DIR = ROOT_DIR+'/summary'
IDENTIFYER = ['arslr','arsn'] # change dependent on the parameter tuned
TAG = ['exp','player0_exp','player1_exp']

if os.path.exists(SUMMARY_DIR):
    shutil.rmtree(SUMMARY_DIR, ignore_errors=True)
else:
    os.makedirs(SUMMARY_DIR)

for exp_dir_name in os.listdir(ROOT_DIR):
    if os.path.isfile(ROOT_DIR+'/'+exp_dir_name) or exp_dir_name=='summary':
        continue
    print('parsing',end='...')
    # extract experiment related params from folder name for ease of display
    #identifier_li = exp_dir_name.split('_')
    #exp_id = ''
    #for identifier in IDENTIFYER:
    #    i = identifier_li.index(identifier)
    #    exp_id = '_'+identifier+'_'+identifier_li[i+1]
    #exp_id = exp_id[1:]
    exp_id = exp_dir_name
    writer = SummaryWriter(logdir=SUMMARY_DIR+'/'+exp_id)
    log_folder = ROOT_DIR+'/'+exp_dir_name+'/log/'
    events_file = log_folder + os.listdir(log_folder)[0]
    for e in tf.train.summary_iterator(events_file):
        for v in e.summary.value:
            if v.tag in TAG:
                writer.add_scalar(v.tag,v.simple_value,e.step)
    writer.close()
    print(exp_dir_name,'finished parsing')
