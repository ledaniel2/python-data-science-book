class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
 
    def display(self):
        print(f'Point: ({self.x}, {self.y})')
        
# Create a Point object
point1 = Point(3, 4)

# Access the attributes
print(point1.x)  # Output: 3
print(point1.y)  # Output: 4

# Call the methods
point1.move(1, 2)
point1.display()  # Output: Point: (4, 6)
class ColoredPoint(Point):
    def __init__(self, x, y, color):
        super().__init__(x, y)
        self.color = color
 
    def display(self):
        print(f"Colored Point: ({self.x}, {self.y}), Color: {self.color}")

# Create a ColoredPoint object
point2 = ColoredPoint(5, 7, "red")

# Access the attributes and methods
point2.move(2, 3)  # This method is inherited
point2.display()   # Output: Colored Point: (7, 10), Color: red
