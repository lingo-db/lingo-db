import re

# File paths
file1 = './include/lingodb/compiler/Dialect/util/UtilOps.td'
file2 = './src/execution/baseline/BaselineBackend.cpp'

# Extract strings from the first file
with open(file1, 'r') as f:
    content1 = f.read()
list1 = re.findall(r'^def (\w+)', content1, re.MULTILINE)

# Extract strings from the second file
with open(file2, 'r') as f:
    content2 = f.read()
list2 = re.findall(r'dialect::util::(\w+)', content2)

# Find missing strings
missing = [s for s in list1 if s not in list2]

# Output missing strings
print("Uncovered ops from UtilOps.td:\n")
for s in missing:
    print(s)

coverage = len(list1) - len(missing)
coverage_percent = (coverage / len(list1)) * 100
print(f"\nTotal coverage: {coverage} out of {len(list1)} ({coverage_percent:.2f}%)")
