import random

data = 5 * randn(10000) + 50
q25, q75 = percentile(data, 25), percentile(data, 75)
iqr = q75 - q25
print(iqr)