from datetime import date, datetime, timedelta
import math

def convert_date(date, type='linear'):
    formated_date = datetime.strptime(date, '%Y-%m-%d').date()
    day = formated_date.timetuple().tm_yday -1
    all = datetime(formated_date.year, 12, 31).timetuple().tm_yday -1
    if type == 'linear':
        return abs((2*day/all) -1)
    if type == 'sinus':
        return abs(round(math.sin((day/all)*math.pi), 15))

print(convert_date('2009-12-01', 'sinus'))