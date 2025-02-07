import csv
from sklearn.model_selection import train_test_split


data, labels = [], []
with open('data/data.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(csvreader)  # Skip the header (id,main,target,context,score)

    for row in csvreader:
        data.append([row[0], row[1], row[2], row[3]])
        labels.append(float(row[4]))


# Split the data into train, validation, and test sets (80% train, 10% validation, 10% test)
train_data, val_test_data, train_labels, val_test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
val_data, test_data, val_labels, test_labels = train_test_split(
    val_test_data, val_test_labels, test_size=0.5, random_state=42
)


# Specify the file name
filename = "data/test_data.csv"

new_data = []
for texts, label in zip(test_data, test_labels):
    new_data.append([texts[0], texts[1], texts[2], texts[3], label])


# Open the CSV file in write mode and write the data
with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(new_data)