def print_student_info(name, grade, age='not known'):
    print(f'Name: {name}, Grade: {grade}, Age: {age}')

# Using positional arguments
print_student_info('Bob', '10th Grade')

# Using named parameters
print_student_info(grade='10th Grade', name='Bob')
