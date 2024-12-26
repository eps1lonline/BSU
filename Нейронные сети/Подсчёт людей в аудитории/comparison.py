import matplotlib.pyplot as plt

result1 = []
with open("result1.txt", "r") as file:
    lines = file.readlines()
    for line in lines[1:]:
        parts = line.strip().split('\t')
        result1.append(parts[0])

result2 = []
with open("result2.txt", "r") as file:
    lines = file.readlines()
    for line in lines[1:]:
        parts = line.strip().split('\t')
        result2.append(parts[0])

common_ids = list(set(result1) & set(result2))
result1_count = len(result1)
result2_count = len(result2)

x_labels = ['DNN', 'YOLOv8']
y_values = [result1_count, result2_count]

plt.bar(x_labels, y_values, color=['yellow', 'orange'])
plt.ylabel('people_count')
plt.title('comparison')

plt.yticks(range(0, max(y_values) + 1, 1))

plt.savefig('graph.png')

print(f'end')