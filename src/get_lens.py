import numpy

def load_data(input_filename):
    '''
    Read the TSV file `input_filename' and return a map of
    the form:
    'tweet_id': tuple([event, tweet, offensive, emotion])
    '''

    tweets = {}

    with open(input_filename, "r", encoding="utf-8") as input_file:

        seen_header = False


        for line in input_file:

            #skip header
            if (not seen_header ):
                seen_header = True
                continue

            line = line.strip()

            tweet_id, event, tweet, offensive, emotion = line.split("\t")

            tweets[tweet_id] = tuple([event, tweet, offensive, emotion])

    return tweets

data = load_data("train.tsv")

lens = []

for tweet in data:
    lens.append(len(data[tweet][1]))

arr = numpy.array(lens)
print("Mean: {0}; Std dev: {1}; Max: {2}; Min: {3}".format(arr.mean(), arr.std(), max(lens), min(lens)))

short, avg, long = 0, 0, 0;

for seen_len in lens:
    if (seen_len < 120):
        short += 1
    elif (seen_len < 180 ):
        avg += 1
    else:
        long += 1

print("Short: {0}; Avg: {1}; Long: {2}".format(short, avg, long))

