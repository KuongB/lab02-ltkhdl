
import numpy as np
import csv
import os

DATA_PATH = '../data/raw/AB_NYC_2019.csv'

def load_and_convert():
    print(f"Reading {DATA_PATH}...")
    with open(DATA_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)
    
    print(f"Loaded {len(data_rows)} rows.")
    
    # Helper for floats (handles empty string)
    def parse_float(x):
        return float(x) if x != '' else np.nan
        
    # Helper for ints (handles empty string - defaulting to 0 or -1? 
    # Usually availability/reviews are 0 if missing, ids shouldn't be missing)
    def parse_int(x):
        return int(x) if x != '' else 0

    # Helper for date
    def parse_date(x):
        return np.datetime64(x) if x != '' else np.datetime64('NaT')

    # Column mapping (by index since order is fixed in CSV usually)
    # ['id', 'name', 'host_id', 'host_name', 'neighbourhood_group', 
    #  'neighbourhood', 'latitude', 'longitude', 'room_type', 'price', 
    #  'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 
    #  'calculated_host_listings_count', 'availability_365']
    
    # Define types
    # Strings as object to avoid truncation of long names
    # Categories as Unicode fixed width? object is safer and fine for strings.
    # Numbers as int32/float32 or 64.
    
    dt = np.dtype([
        ('id', np.int64),
        ('name', object),
        ('host_id', np.int64),
        ('host_name', object),
        ('neighbourhood_group', 'U20'), # Low cardinality
        ('neighbourhood', object), # Can be long?
        ('latitude', np.float64),
        ('longitude', np.float64),
        ('room_type', 'U20'),
        ('price', np.int32),
        ('minimum_nights', np.int32),
        ('number_of_reviews', np.int32),
        ('last_review', 'M8[D]'),
        ('reviews_per_month', np.float64),
        ('calculated_host_listings_count', np.int32),
        ('availability_365', np.int32)
    ])
    
    # Transformation loop
    converted_rows = []
    
    for i, row in enumerate(data_rows):
        try:
            # row indices
            # 0: id
            # 1: name
            # 2: host_id
            # 3: host_name
            # 4: neighbourhood_group
            # 5: neighbourhood
            # 6: lat
            # 7: lon
            # 8: room_type
            # 9: price
            # 10: min_nights
            # 11: reviews
            # 12: last_review
            # 13: reviews_pm
            # 14: calc_host
            # 15: avail
            
            # Simple direct mapping
            record = (
                int(row[0]),             # id
                row[1],                  # name
                int(row[2]),             # host_id
                row[3],                  # host_name
                row[4],                  # neighbourhood_group
                row[5],                  # neighbourhood
                float(row[6]),           # lat
                float(row[7]),           # lon
                row[8],                  # room_type
                int(row[9]),             # price
                int(row[10]),            # min_nights
                int(row[11]),            # reviews
                parse_date(row[12]),     # last_review
                parse_float(row[13]),    # reviews_pm
                int(row[14]),            # calc_host
                int(row[15])             # avail
            )
            converted_rows.append(record)
        except Exception as e:
            print(f"Error converting row {i}: {row}")
            print(e)
            break
            
    print("Converting to structured array...")
    structured_data = np.array(converted_rows, dtype=dt)
    
    print("Success!")
    print(structured_data[:3])
    print(structured_data.dtype)
    print("Price mean:", np.mean(structured_data['price']))
    print("Reviews/month mean (nanmean):", np.nanmean(structured_data['reviews_per_month']))

if __name__ == "__main__":
    load_and_convert()
