import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

try:
    x1, y1, x2, y2 = map(float, input("Enter x1, y1, x2, y2: ").split())
    print("Distance:", distance(x1, y1, x2, y2))
except ValueError:
    print("Please enter valid numbers.")
except Exception as e:
    print(f"An error occurred: {e}")