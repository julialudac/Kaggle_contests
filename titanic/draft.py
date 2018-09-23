import csv


with open("train.csv") as f:
    reader = csv.reader(f)
    #print(reader)
    reader.__next__()
    for row in reader:
        print(row)