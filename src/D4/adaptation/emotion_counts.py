#!/usr/bin/python3

import csv
import collections

dic = collections.defaultdict(lambda: 0)

with open("train_nor_811.csv", "r") as in_f:

    csv_reader = csv.reader(in_f)

    for row in csv_reader:

        text_id, emotion, text = row

        dic[emotion] += 1

print(dic)

