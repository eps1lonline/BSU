import copy

def parse_coordinates(coord_str):
    x, y = coord_str.strip('()').split(';')
    return float(x), float(y)

def manhattan_distance(coord1, coord2):
    x1, y1 = parse_coordinates(coord1)
    x2, y2 = parse_coordinates(coord2)
    # print("|", x2, " - ", x1, "| + |", y2, " - ", y1, "| = ", (abs(x2 - x1) + abs(y2 - y1)))
    return abs(x2 - x1) + abs(y2 - y1)


ptr = []
with open("C:/Users/nikit/Desktop/Координаты_домов.txt", "r", encoding='utf-8') as file:
    for line in file:
        ptr.append(line)
# print(ptr)


points = copy.deepcopy(ptr)
for count, i in enumerate(points):
    if (i == "#\n"):
        points.pop(count)
        points.pop(count)


for count, i in enumerate(points):
    points[count] = i.split('~', 1)[1].strip()
# print(points)


massSum = []
for i in range(0, len(points)):
    sum = 0
    # print(points[i])
    for j in range(0, len(points)):
        if i != j:
            sum += manhattan_distance(points[i], points[j])
    massSum.append(sum)
    # print("sum =", sum, "\n")
# print("massSum:", massSum)


min = massSum[0]
for count, i in enumerate(massSum):
    if i < min:
        min = i
        indexMin = count
# print("min =", min)
# print("indexMin =", indexMin)
# print(points[indexMin], "\tdistance =", min)


# print(ptr)
for count, i in enumerate(ptr):
    if points[indexMin] in i:
        print("Оптимальное расстояние до любого дома:\n", ptr[count - int(ptr[count].split('~')[0])], ptr[count], end="")
        break