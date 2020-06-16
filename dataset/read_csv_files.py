import csv
import os 

path='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias'

def read_csv(name):
    list = []
    with open(os.path.join(path, name ), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            list.append(row)
    return list[1:]

def read_csv_with_label(name):
    list = []
    with open(os.path.join(path, name ), newline='') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            list.append(row)
    return list[1:]

def add_label_dictionary(listitem):
    list = []
    list.append({"id":listitem[0]})
    list.append({"init_date":listitem[1]})
    list.append({"country":listitem[2]})
    list.append({"T2M":listitem[3]})
    list.append({"U10M":listitem[4]})
    list.append({"V10M":listitem[5]})
    list.append({"RTOT":listitem[6]})
    list.append({"gh_500":listitem[7]})
    list.append({"t_850":listitem[8]})
    list.append({"u_300":listitem[9]})
    list.append({"u_500":listitem[10]})
    list.append({"u_850":listitem[11]})
    list.append({"v_300":listitem[12]})
    list.append({"v_500":listitem[13]})
    list.append({"v_850":listitem[14]})
    list.append({"q_850":listitem[15]})
    list.append({"msl":listitem[16]})
    list.append({"10u":listitem[17]})
    list.append({"10v":listitem[18]})
    list.append({"tp":listitem[19]})
    list.append({"slhf":listitem[20]})
    list.append({"sshf":listitem[21]})
    list.append({"ssr":listitem[22]})
    list.append({"2t":listitem[23]})
    list.append({"sm20":listitem[24]})
    list.append({"tcw":listitem[25]})
    list.append({"AT":listitem[26]})
    list.append({"ZO":listitem[27]})
    list.append({"ZOEA":listitem[28]})
    list.append({"AR":listitem[29]})
    list.append({"ZOWE":listitem[30]})
    list.append({"BL":listitem[31]})
    list.append({"GL":listitem[32]})
    return list

def filterCountry(data, country):
    return list(filter(lambda x: True if x[2]==country else False, data))

def main():
    read_csv('ecmwf_PF_03_240.csv')

if __name__ == "__main__":
    main()