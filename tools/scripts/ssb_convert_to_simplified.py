import os

nations = """ALGERIA
ARGENTINA
BRAZIL
CANADA
EGYPT
ETHIOPIA
FRANCE
GERMANY
INDIA
INDONESIA
IRAN
IRAQ
JAPAN
JORDAN
KENYA
MOROCCO
MOZAMBIQUE
PERU
CHINA
ROMANIA
SAUDI ARABIA
VIETNAM
RUSSIA
UNITED KINGDOM
UNITED STATES"""

# 25 nations
nations = nations.split('\n')


import argparse

# 5 regions
regions = """AFRICA
AMERICA
ASIA
EUROPE
MIDDLE EAST"""
regions = regions.split('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'convert')
    parser.add_argument('data_directory', type=str, help='Data Directory')
    args = parser.parse_args()

    data_dir = args.data_directory
    # process suppliers
    suppliers_path = data_dir + 'supplier.tbl'
    suppliers_new_path = data_dir + 'supplier.tbl.simple'
    lines = open(suppliers_path).readlines()
    o = []
    for line in lines:
        try:
            parts = line.split('|')
            parts[4] = str(nations.index(parts[4]))
            parts[5] = str(regions.index(parts[5]))
            parts[3] = str(int(parts[4]) * 10 + int(parts[3][-1]))
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(suppliers_new_path,'w')
    for line in o:
        f.write(line)
    f.close()
    os.remove(suppliers_path)
    os.rename(suppliers_new_path, suppliers_path)

    # process customers
    customers_path = data_dir + 'customer.tbl'
    customers_new_path = data_dir + 'customer.tbl.simple'
    lines = open(customers_path).readlines()
    o = []
    for line in lines:
        try:
            parts = line.split('|')
            parts[4] = str(nations.index(parts[4]))
            parts[5] = str(regions.index(parts[5]))
            parts[3] = str(int(parts[4]) * 10 + int(parts[3][-1]))
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(customers_new_path,'w')
    for line in o:
        f.write(line)
    f.close()
    os.remove(customers_path)
    os.rename(customers_new_path, customers_path)

    # process parts
    parts_path = data_dir + 'part.tbl'
    parts_new_path = data_dir + 'part.tbl.simple'
    lines = open(parts_path).readlines()
    o = []
    for line in lines:
        try:
            parts = line.split('|')
            parts[2] = int(parts[2].split('#')[-1]) - 1
            parts[3] = parts[2] * 5 + ((int(parts[3].split('#')[-1]) % 10) - 1)
            parts[4] = parts[3] * 40 + ((int(parts[4].split('#')[-1][2:])) - 1)
            parts[2] = str(parts[2])
            parts[3] = str(parts[3])
            parts[4] = str(parts[4])
            o.append('|'.join(parts))
        except:
            print(line)
            break

    f = open(parts_new_path,'w')
    for line in o:
        f.write(line)
    f.close()
    os.remove(parts_path)
    os.rename(parts_new_path, parts_path)
