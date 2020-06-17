import csv
import os
from itertools import groupby
from datetime import datetime, date, timedelta  

path='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias'

def read_csv(name):
    list = []
    with open(os.path.join(path, name ), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        rownr = 0
        for row in spamreader:
            if rownr is not 0:
                for i in range(len(row)):
                    if i==0:
                        row[i]=int(row[i])
                    if i not in [0,1,2]:
                        row[i]=float(row[i])
                list.append(row)
            rownr+=1
    return list[1:]

def read_csv_with_title(name):
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

def filter_country(data, country):
    return list(filter(lambda x: True if x[2]==country else False, data))

def get_labes_by_country(data, id, days):
    countries=[]
    for key, group in groupby(sorted(data, key=lambda x: x[2]), lambda x: x[2]):
        countries.append(list(group))
    resultlist = []
    for country in countries:
        result_country = get_labes(country, id, days)
        resultlist.append(result_country)
    return resultlist

def get_labes(data, id, days):
    start = datetime.strptime(data[id][1], '%Y-%m-%d').date()
    dates = []
    date = start
    for i in range(days):
        dates.append(date.strftime("%Y-%m-%d"))
        date = date +timedelta(days=1)
    filterddata = list(filter(lambda x: True if x[1] in dates else False, data))
    T2M = 0 
    U10M = 0
    V10M = 0 
    RTOT = 0
    for i in  filterddata:
        T2M  += i[3]
        U10M += i[4]
        V10M += i[5]
        RTOT += i[6]
    return [i[2], start.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"),  T2M/len(filterddata), U10M/len(filterddata), V10M/len(filterddata), RTOT]

def main():
    read_csv('ecmwf_PF_03_240.csv')

if __name__ == "__main__":
    main()