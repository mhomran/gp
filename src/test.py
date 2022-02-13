class obj:
    def __init__(self, salary):
        self.salary = salary


x = obj(10)
y = obj(20)
x.count = 100
y.count = 200
ut = [x, y]

# ut.sort(key=lambda x: x.count, reverse=True)

for x in ut:
    print(x.salary)
    print(x.count)
    print("---")
