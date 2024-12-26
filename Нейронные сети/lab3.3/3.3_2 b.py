import random

def rand_gen():
    random.seed(132, version=2)
    print(random.uniform(0, 1))

for _ in range(4):
    rand_gen()