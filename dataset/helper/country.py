from datetime import date, datetime, timedelta
import math

def convert_country(name):
    countries = [{"name":'Portugal', "id": 1},{"name":'Spain', "id":2},{"name":'France', "id":3},{"name":'Ireland', "id":4},{"name":'United Kingdom', "id":5},{"name":'Belgium', "id":6},{"name":'Netherlands', "id":7},{"name":'Germany', "id":8},{"name":'Switzerland', "id":9},{"name":'Austria', "id":10},{"name":'Italy', "id":11},{"name":'Czechia', "id":12},{"name":'Poland', "id":13},{"name":'Denmark', "id":14},{"name":'Norway', "id":15},{"name":'Sweden', "id":16},{"name":'Finland', "id":17},{"name":'Hungary', "id":19},{"name":'Croatia', "id":20},{"name":'Romania', "id":21},{"name":'Bulgaria', "id":22},{"name":'Greece', "id":23}]
    for country in countries:
        if name.upper() == country["name"].upper():
            return country["id"]
    return 'not in the list of countries'
