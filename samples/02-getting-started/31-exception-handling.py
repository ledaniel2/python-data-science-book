try:
    result = 10 / 0
except ZeroDivisionError:  # Can use simply 'except:' to catch all types of exception
    print("Oops! You tried to divide by zero.")
finally:                   # This clause is optional
    print("This will always be executed.")
