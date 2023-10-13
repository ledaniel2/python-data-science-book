# Chapter 2: Getting Started with Python

Now that you have a basic understanding of data science and the role of Python, it's time to get your hands dirty with some actual programming! In this chapter, we will walk you through the process of installing Python and setting up your environment, ensuring that you have all the necessary tools to start writing and running Python programs.

Once your environment is set up, we will uncover the basics of the Python programming language, introducing fundamental concepts such as variables, data types, operators, and control structures. As you become more comfortable with these concepts, we will progress to more advanced ones, including functions, classes, and docstrings, which are essential for modularizing and reusing your code. This will serve as a strong foundation for the rest of the book, as we explore some of Python's many libraries and their relevance to the world of data science. 

We will be covering a lot of ground, so if you're new to Python, you may want to experiment with the code examples as you progress. If you already know some Python and just need a refresher, feel free to skim through this chapter. Don't worry, everything is straightforward, and we'll guide you step by step. So, let's embark on your Python programming journey!

Our learning goals for this chapter are:

 * Learn how to install Python and set up a development environment
 * Understand the fundamentals of Python programming, including data types, variables, operators, loops, and conditionals
 * Gain knowledge of functions, modules, and packages to better organize, structure and reuse your code
 * Acquire familiarity with more advanced Python concepts, which will be useful when utilizing data science libraries

## 2.1: Running a Python Program

Python is a high-level, general-purpose programming language that was created by Guido van Rossum in the late 1980s. The development of Python was started in December 1989, and its first version, Python 0.9.0, was released in February 1991. The language was designed with an emphasis on code readability, simplicity, and ease of use.

Python is named after the British comedy group Monty Python, and its official documentation often contains references and jokes related to their work. Guido van Rossum wanted to create a language that would appeal to both professional programmers and hobbyists, so he sought inspiration from the group's humor and unconventional approach.

The Python interpreter is a program that reads and executes Python code. It is an essential component of the Python ecosystem and enables you to run Python scripts interactively or from a file. When you run a Python script, the interpreter compiles the source code into bytecode, which is an intermediate representation of the code that is optimized for execution. This bytecode is then executed by the Python Virtual Machine (PVM), which is part of the interpreter.

To speed up the execution of Python scripts, the interpreter caches the compiled bytecode in files with the `.pyc` extension. These files are stored in the `__pycache__` directory, which is located in the same folder as the original `.py` files on most systems, or in `C:\Users\%USERNAME%\AppData\Local\Temp` on Windows.

When you run a script, the interpreter first checks if there is a valid `.pyc` file in the `__pycache__` directory. If one is found, the interpreter will use the cached bytecode to execute the script, skipping the compilation step. If no `.pyc` file is found or if the source code has been modified since the last run, the interpreter will compile the script and create or update the corresponding `.pyc` file.

Using compiled bytecode files can significantly improve the startup time of your Python scripts, especially for large projects. However, it is important to note that the performance improvements are limited to the initial loading of the script, and the actual runtime performance remains unchanged.

### Downloading and Installing Python

To get started with Python, you first need to download and install it on your computer. Follow these steps:

 1. Visit the official Python website at https://www.python.org/
 2. Click on the 'Downloads' tab.
 3. Choose the appropriate version for your operating system (Windows, macOS, or Linux). You can also download the latest stable version directly by clicking on the download button.
 3. Run the downloaded installer and follow the on-screen instructions. Be sure to check the box that says "Add Python to PATH" during the installation process, as this will make it easier to run Python from the command line.

Alternatively, on operating systems with a package manager, such as Linux, it may be better to install Python using either the command-line or GUI-based package manager. Consult your distribution's documentation for details on how to do this. Do check the version of Python provided against the current stable version available from the official website.

### Using the Python Interpreter

Once you have installed Python, you can start the Python interpreter by opening a terminal or command window and typing `python` (or `python3` on some systems). You should see the Python version number and a prompt (`>>>`) where you can type Python commands. To exit the interpreter, type `exit()` or press Ctrl + Z, then Enter (Windows), or Ctrl + D (macOS/Linux).

For editing Python scripts, you can use any text editor you prefer, such as Notepad++ (Windows), Sublime Text (macOS/Windows/Linux), or Visual Studio Code (macOS/Windows/Linux). Save your Python code with a `.py` file extension.

To run a Python script from the command line, navigate to the folder containing the script using the `cd` command and type `python your_script_name.py` (or `python3 your_script_name.py` on some systems). Use `-i` immediately *before* the script name to enter an interactive Python environment after your script has run, allowing you to examine global variables or call individual functions, for example.

It is important to recognize a significant distinction in Python's block structure (explained later in this chapter) when inputting code into the interactive Python interpreter as opposed to writing a Python script. In the interpreter, an empty line always terminates the current block, regardless of the indentation level. Conversely, within a Python script, an empty line between lines with equal indentation is both permissible and quite common. In real-world scenarios, this implies that directly pasting Python scripts into the interpreter may not always work as intended, although adding a single space character to any blank lines usually suffices; a better solution is to save the script and pass it with the `-i` option to the interpreter.

### Setting Up the Environment

Now that Python is installed, it's time to set up the environment. It's a good practice to create a virtual environment for each Python project to avoid conflicts between package versions. We'll be using `venv` (a built-in Python module) to create virtual environments.

To set up and use a virtual environment, follow these steps:

 1. Open a terminal or command window.
 2. Navigate to your project folder using the `cd` command.
 3. Run `python -m venv your_environment_name` (or `python3 -m venv your_environment_name` on some systems) to create a new virtual environment (this command may take some time to complete). Note that this step only needs to be performed once.
 4. Activate the virtual environment by running `source your_environment_name/bin/activate` (macOS/Linux) or `your_environment_name\Scripts\activate` (Windows) as a command in the terminal or command window.
 5. Install any required packages using `pip install package_name`. To exit the virtual environment, type `deactivate`.

Once the virtual environment is activated, you'll see the environment name in parentheses before the command prompt. For example:

```bash
(DataScienceProject) C:\Users\user\Documents\PythonProjects\DataScienceProject>
```

Congratulations! You've successfully installed Python and set up a virtual environment for your data science project. Now let's learn how to use the interpreter by testing it with some small Python programs.

## 2.2: Python Basics

We'll now explore the fundamentals of Python programming, introducing data types, variables, operators, loops, conditionals, block structure and more. These concepts form the foundation of your Python programming journey, so let's dive in!

### Types

Python has several built-in data types that allow you to work with different kinds of data. Some of the most common data types we will meet are:

| Data Type              | Python Type | Description                                          | Examples                             |
|------------------------|-------------|------------------------------------------------------|--------------------------------------|
| Integers               | `int`         | Whole Numbers (up to 64-bit)                         | 0, -1000, 23                         |
| Floating-Point Numbers | `float`       | Decimal Numbers (up to about 15 significant figures) | 3.14, 1.23e20                        |
| Booleans               | `bool`        | Binary Decision Value                                | True, False                          |
| Strings                | `str`         | Text                                                 | 'Hello'                              |
| Lists                  | `list`        | Ordered, Mutable Collections                         | [1, 2.3, False], []                  |
| Tuples                 | `tuple`       | Ordered, Immutable Collections                       | (1, 2.3, False), ()                  |
| Dictionaries           | `dict`        | Key-Value Pairs                                      | {'key1': 'value1', 'key2': 'value2'} |

### Variables

Variables are used to store data in a program. They are given a name, and you can assign a value to them using the assignment operator (=). The name must begin with a letter or underscore, and contain other letters, numbers or underscores (this is a slight simplification). The value can be any valid Python expression, such as a number, a string, or the result of an arithmetic operation.

Here are some examples of variables showing the use of descriptive variable names:

```python
x = 5
y = 3.2
greeting = 'Hello, Python!'
is_happy = True
```

Variables can be reassigned to any number of times, including with a different type of expression to before:

```python
x = "I'm a string, now!"
```

### Operators

Operators are symbols that perform operations on combinations of values and variables. Python has several types of operators, including arithmetic, comparison, and logical operators.

 1. Arithmetic operators: Perform mathematical operations.

```python
x = 5
y = 2

# Addition
result = x + y   # result = 7

# Subtraction
result = x - y   # result = 3

# Multiplication
result = x * y   # result = 10

# Division
result = x / y   # result = 2.5

# Floor division
result = x // y  # result = 2

# Modulus
result = x % y   # result = 1

# Exponentiation
result = x ** y  # result = 25

# Negation
result = -x      # result = -5
```

 2. Comparison operators: Compare values and return a boolean result (`True` or `False`).

```python
x = 5
y = 2

# Equal to
result = x == y  # result = False

# Not equal to
result = x != y  # result = True

# Greater than
result = x > y   # result = True

# Less than
result = x < y   # result = False

# Greater than or equal to
result = x >= y  # result = True

# Less than or equal to
result = x <= y  # result = False
```

 3. Logical operators: Perform logical operations (AND, OR, NOT) on boolean values.

```python
x = True
y = False

# Logical AND
result = x and y  # result = False

# Logical OR
result = x or y   # result = True

# Logical NOT
result = not x    # result = False
```

### Loops

Loops are used to execute a block of code repeatedly. Python has two types of loops: for and while.

 1. `for` loop: Executes a block of code a fixed number of times, iterating over a sequence (e.g., a Python list or a range object).

```python
# Loop through a range
for i in range(5):
    print(i)
```

This will output:

```plaintext
0
1
2
3
4
```

```python
# Loop through a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

This will output:

```plaintext
apple
banana
cherry
```

 2. `while` loop: Executes a block of code as long as a given condition evaluates to `True`.

```python
countdown = 5
while countdown >= 0:
    print(countdown, end=' ')  # Separate output with a space character
    countdown = countdown - 1
print()  # Start a new line
```

This will output:

```plaintext
5 4 3 2 1 0
```

### Conditionals

Conditionals are used to execute a block of code only if a specific condition is met. Python has three conditional statements: if, elif, and else.

 1. `if`: Executes a block of code if the specified condition evaluates to `True`.

```python
x = 5

if x > 3:
    print("x is greater than 3")
```

This will output:

```plaintext
x is greater than 3
```

 2. `elif`: Can be used after an `if` statement to check for additional conditions if the previous condition(s) evaluated to `False`.

```python
x = 5

if x > 10:
    print("x is greater than 10")
elif x > 3:
    print("x is greater than 3 but not greater than 10")
```

This will output:

```plaintext
x is greater than 3 but not greater than 10
```

 3. `else`: Executes a block of code if none of the previous conditions evaluated to `True`.

```python
x = 2

if x > 10:
    print("x is greater than 10")
elif x > 3:
    print("x is greater than 3 but not greater than 10")
else:
    print("x is not greater than 3 or 10")
```

This will output:

```plaintext
x is not greater than 3 or 10
```

### Strings

Strings are sequences of characters enclosed in single or double quotes:

```python
text = 'Hello, World!'
```

This book tries to standardize on the former, but there really is no difference unless you need to have a single or double quote as part of the string itself.

Raw strings begin and end with either `"""` or `'''` and can span multiple lines, respecting the indentation of the end of opening triple-quote:

```python
text2 = """''""''""''''"""
text3 = """
           This isn't displayed as indented, but has a blank line both before and after.
        """
```

Some useful string methods (functions which operate on `str` variables) are:

 * `len(text)`: Number of characters in variable `text`
 * `text.upper()`: Convert variable `text` to uppercase
 * `text.lower()`: Convert variable `text` to lowercase
 * `text.strip()`: Remove whitespace from the beginning and end of variable `text`
 * `text.replace(old, new)`: Replace all occurrences (if any) of old with new
 * `text.split(separator)`: Split the string into a list based on separator

Formatted strings allow output of variables, values or expressions enclosed in braces (`{` and `}`) as part of their contents, for example:

```python
x = 1.2
y = True
print(f'x has value {x}, while y is {y}')
```

### Lists

Lists are ordered, mutable collections of items, which are often all be of the same type, but this is not mandatory:

```python
numbers = [1, 2, 3, 4, 5]
```

Some common list operations:

 * `len(numbers)`: Length of the list
 * `numbers.append(6)`: Add an item to the end
 * `numbers.insert(0, 0)`: Insert an item at a specific index
 * `numbers.remove(3)`: Remove the first occurrence of an item
 * `numbers.pop()`: Remove and return the last item
 * `numbers.index(4)`: Find the index of the first occurrence of an item

List slicing:

 * `numbers[start:end]`: Get a sublist from start to end (exclusive of end)
 * `numbers[start:end:step]`: Get a sublist with a step


### Block Structure

Python uses indentation (exclusively) to define code blocks:

```python
print('Please enter a number: ')
x = int(input())

if x > 0:
    print("x is positive")
    print("Another line inside the if block")
else:
    print("x is non-positive")

print("This line is outside the if-else block")
```

Each level of indentation indicates a new block of code. In the interpreter, the prompt changes to `...` when entry of code is within a block, and entering a blank line at this prompt always closes the current block. Consistent indentation of source files (usually with multiples of four spaces) is crucial for readability and avoiding syntax errors.

Variables can be assigned or reassigned within a block, and will maintain their visibility (local to a function or global, see later in this chapter).

### Comments

Comments in Python are used to explain code, making it more readable. There are two types of comments:

Single-line comments: Start with a # symbol.

```python
# This is a single-line comment
x = 5  # This is an inline comment
```

Multi-line comments: Enclosed between triple quotes (either single or double).

```python
"""
This is a multi-line comment.
It can span multiple lines.
"""
```

Comments are ignored by the compiler (except for docstrings, see later in this chapter). They can contain unused Python code (this is called "commented out code"), expected output, other values, variable names or natural language. Comments are usually written in English if the code is intended to be shared or distributed. The most frequent reader of your comments is probably yourself, so get ahead of the game and document code which is non-obvious to the casual observer. You'll thank yourself for it!

## 2.3: Functions, Modules, and Packages

The first Python programs you will write are likely to fit into a single source file, or may even be simply part of an interactive session with the Python interpreter. This approach doesn't scale though, and soon you'll want to achieve modularity and code reuse in your projects.

### Functions

Functions are blocks of reusable code that perform a specific task. They allow you to break your code into smaller, more manageable pieces, making it easier to read and maintain. Functions can optionally take input parameters, known as arguments, and return a value.

To define a function, you use the `def` keyword, followed by the function name and a pair of parentheses with any input parameters. The function body is then indented and can include a `return` statement to return a value.

Here's an example of a simple function that adds two numbers:

```python
def add_numbers(a, b):
    result = a + b
    return result

# The next line calls the function
sum_result = add_numbers(5, 3)

print(sum_result)
```

This will output:

```plaintext
8
```

The `def` keyword starts an indented block which must be ended by a blank line, or the interpreter will issue an error message when it tries to compile it. Variables which are assigned to within a function are local by default, to override this and reference a global variable, use the `global` keyword.

```python
x = 10

def modify_x():
    global x
    x = x / 2

print(x)
modify_x()
print(x)
```

This will output:

```plaintext
10
5.0
```

### Modules

Modules are files containing Python code, usually a collection of functions, classes, and variables. They help you organize your code into separate, related components, making it more manageable and reusable. You can import a module into another Python script using the `import` statement. This allows you to use the functions, classes, and variables defined in the module.

Here's an example of how to create and use a module:

 1. Create a new file called `math_operations.py` with the following content:

```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b
```

 2. In another file `test_math.py`, import the `math_operations` module and use its functions:

```python
import math_operations

result = math_operations.add(5, 3)
print(result)
result = math_operations.subtract(5, 3)
print(result)
```

This will output:

```plaintext
8
2
```

Also possible is to use a renamed import:

```python
import math_operations as Math
result = Math.add(20, 30)
```

Or even specify exactly which parts of the module to use:

```python
from math_operations import subtract
result = subtract(20, 10)
```

### Packages

Packages are a way to organize related modules into a single directory structure. A package is simply a directory containing an `__init__.py` file (which may be empty) and one or more module files.

Here's an example of how to create and use a package:

 1. Create a directory named `my_package`.
 2. Inside `my_package`, create an empty file named `__init__.py`.
 3. Inside `my_package`, create a file named `string_operations.py` with the following content:

```python
def concatenate(a, b):
    return a + b

def reverse(s):
    return s[::-1]
```

 4. In another file `test_string.py`, import the `string_operations` module from the `my_package` package and use its functions:

```python
from my_package import string_operations

result = string_operations.concatenate('Hello, ', 'World!')
print(result)

result = string_operations.reverse('Python')
print(result)
```

This will output:

```plaintext
Hello, World!
nohtyP
```

### Docstrings

Docstrings are used to provide documentation for functions, classes, and modules. They are written as a string literal enclosed in triple quotes (either single or double) and placed immediately after the function, class, or module definition. There is no fixed format of a docstring, but you should aim to be as verbose as you think is necessary. The following example provides some idea of a typical outline:

```python
def greet(name):
    """
    Return a greeting message for the given name.

    Args:
        name (str): The name to greet.

    Returns:
        str: The greeting message.
    """
    return f'Hello, {name}!'
```

In summary, these concepts of functions, modules and packages play a crucial role in structuring and organizing Python code, making it more reusable, maintainable, and scalable. Your data science projects will have better reach and reception if you learn to use these facilities effectively.

## 2.4: Advanced Python

We'll now move on to some more advanced features of Python: classes, inheritance, lambdas, list comprehensions, named and default parameters, decorators, exception handling and unit testing. Some of the libraries we will introduce later in the book rely on client code using these Python features, which is the reason for discussing them here. You'll soon have a solid understanding of these advanced concepts, enabling you to write more efficient and cleaner Python code.

### Classes

A class is a blueprint for creating objects, which are instances of the class. Objects have attributes (data) and methods (functions) associated with them. To define a class, you use the `class` keyword, followed by the class name and a colon. The whole of the class body is then indented and can include attributes and methods.

Python classes can have both "normal" and "special" methods, the latter of which are named between double underscores, for example `__init__`. Here's an example of a simple class representing a point in two-dimensional space:

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
 
    def display(self):
        print(f'Point: ({self.x}, {self.y})')
```

In this example, the `__init__` method is called the constructor. It initializes the object's attributes when a new object is created. The self parameter represents the instance of the class and is used to access the object's attributes and methods.

To create an object (an "instance of a class"), you call the class name followed by a pair of parentheses with any required input parameters.

```python
# Create a Point object
point1 = Point(3, 4)
```

Notice that the `Point()` constructor is indistinguishable from function call syntax. For this reason, the Python convention is to give all class names an initial capital letter, and use all lowercase letters for function names.

To access an object's attributes or methods, you use the dot notation followed by the attribute or method name.

```python
# Access the attributes
print(point1.x)  # Output: 3
print(point1.y)  # Output: 4

# Call the methods
point1.move(1, 2)
point1.display()  # Output: Point: (4, 6)
```

### Inheritance

Inheritance is a way of creating a new class that is a modified version of an existing class. The new class, called the subclass, inherits the attributes and methods of the existing class, which is called the superclass.

Here's an example of a class ColoredPoint that inherits from the Point class and adds a color attribute:

```python
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
```

In this example, the `super().__init__(x, y)` line calls the constructor of the superclass (Point) to initialize the inherited `x` and `y` attributes.

### Lambdas

Lambda functions, also known as anonymous functions, or simply, lambdas, allow you to create small, simple, one-time-use functions without the need for a formal `def` statement. They are particularly useful for short operations or when you need to pass a function as an argument to another function. Here's an example of how to create a lambda function to square a number:

```python
# Define a lambda function to square a number
square = lambda x: x**2

# Use the lambda function
result = square(5)
print(result)  # Output: 25
```

In this example, we defined a lambda function to square a number and assign it to the variable `square`. Then, we use the lambda function by invoking (passing a value to) it. Lambda functions can also be passed to other functions as a parameter, without the need to assign them to a variable first.

### List Comprehensions

List comprehensions are a concise way to create lists in Python. They provide a more readable and efficient alternative to using `for` loops and the `append()` method. Here's an example of using list comprehension to create a list of squared numbers:

```python
# Using a for loop
squared_numbers = []
for i in range(1, 11):
    squared_numbers.append(i**2)
print(squared_numbers)
```

```python
# Using list comprehension
squared_numbers = [i**2 for i in range(1, 11)]
print(squared_numbers)
```

Both methods create the same list of squared numbers, but the list comprehension is more concise and easier to read. Some of the code examples later in this book use list comprehensions, so you should aim to become able to translate between the two forms.

### Named and Default Parameters

Named parameters, also known as keyword arguments, allow you to specify the name of a parameter when calling a function. This makes your code more readable and less prone to errors, as the order of the arguments doesn't matter when using named parameters.

Default parameters are each given a fallback value to use in case one isn't explicitly provided by the caller. All default parameters must appear at the end of the parameter list, after any the non-default parameters. Default values are provided after an equals `=` sign for each default parameter in the parameter list.

Here's an example of using both named and default parameters:

```python
def print_student_info(name, grade, age='not known'):
    print(f'Name: {name}, Grade: {grade}, Age: {age}')

# Using positional arguments
print_student_info('Bob', '10th Grade')

# Using named parameters
print_student_info(grade='10th Grade', name='Bob')
```

Both function calls produce the same output, but the second call uses named parameters, making the code more readable and less error-prone, as the order of the arguments does not matter. In each call the `age` parameter is not provided, and so receives the default value.

### Decorators

Decorators are a way to modify or extend the behavior of a function without changing its code. A decorator is a function that takes another function as input and returns a new function that usually extends or modifies the input function's behavior.

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

This will output:

```plaintext
Something is happening before the function is called.
Hello!
Something is happening after the function is called.
```

Decorators can be employed to perform logging, timing, exception trapping, or debugging, as in the example above.

### Generator Functions

Python's `yield` keyword is used with functions which can be called repeatedly, and which preserve their state between calls. These functions are known as "generator functions", as they can be used to generate sequences without using a list.

Here's an example of how to use the `yield` statement in Python to create a generator function that generates Fibonacci numbers indefinitely. The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1.

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Create a generator object
fib_gen = fibonacci()

# Print the first 10 Fibonacci numbers
for _ in range(10):
    print(next(fib_gen), end=' ')

print()
```

This will output:

```plaintext
0 1 1 2 3 5 8 13 21 34
```

In this example, the `fibonacci()` function is a generator function that generates an infinite sequence of Fibonacci numbers. The `yield` statement is used to produce the next Fibonacci number in the sequence each time the generator's `next()` function is called. The generator function allows us to generate Fibonacci numbers on-the-fly without having to store the entire sequence in memory.

You can call `next(fib_gen)` as many times as you want, and it will keep generating Fibonacci numbers indefinitely. However, be careful not to call it from within an infinite loop without any break condition, as it will run forever and consume system resources.

### Exception Handling

Exception handling in Python enables you to manage errors gracefully and continue the execution of your program. You can use the `try`, `except`, `finally`, and `raise` keywords to handle exceptions.

Example of exception handling:

```python
try:
    result = 10 / 0
except ZeroDivisionError:  # Can use simply 'except:' to catch all types of exception
    print("Oops! You tried to divide by zero.")
finally:                   # This clause is optional
    print("This will always be executed.")
```

### Unit Testing

Unit testing in Python is essential to ensure that your code is working correctly. The `unittest` module provides a framework for creating and running tests.

Here's an example of a simple unit test using the `unittest` module:

```python
# my_module.py
def add(a, b):
    return a + b
```

```python
# test_my_module.py
import unittest
from my_module import add

class TestMyModule(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

if __name__ == "__main__":
    unittest.main()
```

To run the tests, execute the `test_my_module.py` script, and you will get the following output:

```plaintext
.
----------------------------------------------------------------------
Ran 1 test in 0.001s

OK
```

In summary, these advanced Python concepts will enhance your programming skills and allow you to write more efficient, maintainable, and robust code. As you explore data science with Python, you'll find that these facilities of Python play a significant role in various tasks, such as data manipulation, analysis, and modeling.
