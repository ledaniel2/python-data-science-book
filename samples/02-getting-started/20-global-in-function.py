x = 10

def modify_x():
    global x
    x = x / 2

print(x)
modify_x()
print(x)
