import csv

dataset = []
with open("./valdata_nocnd.csv", "r") as df:
    reader = csv.reader(df)
    dataset = list(reader)

bad_samples = list(filter(lambda x: abs(float(x[4])-0) == 0, dataset[1:]))
bad_samples[42]



