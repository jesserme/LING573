#!/usr/bin/python3

stop_words = []

with open("vietnamese-stopwords.txt", "r") as in_f:
    for line in in_f:
        stop_words.append(line.strip())

print(repr(stop_words))


