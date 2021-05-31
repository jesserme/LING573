#!/bin/sh
/home2/jesserme/spanish_emotions/py_3.9/bin/python3 emotion_detection_primary.py /home2/jesserme/src/D4/primary/dev.tsv /home2/jesserme/src/D4/primary/train.tsv 
/home2/jesserme/spanish_emotions/py_3.9/bin/python3 emotion_detection_primary_eval.py /home2/jesserme/src/D4/primary/dev.tsv /home2/jesserme/src/D4/primary/train.tsv /home2/jesserme/src/D4/primary/eval.tsv