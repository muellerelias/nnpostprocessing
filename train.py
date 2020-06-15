import tensorflow
import dataset.read_csv_files as provider

data = provider.read_csv('ecmwf_PF_03_240.csv')
print(data[0])
print(provider.add_label_dictionary(data[0]))
