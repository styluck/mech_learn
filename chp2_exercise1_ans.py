# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:17:31 2024

@author: lich5
"""
#%% 利用生成器构建fibonacci数列
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Using the generator
fib = fibonacci(5)

# Print Fibonacci numbers
for number in fib:
    print(number)


#%% 利用生成器编写一个无穷长度的等差数列
def infinite_counter(start=0):
    while True:
        yield start
        start += 1

# Using the infinite generator
counter = infinite_counter()

# Print the first 5 values
for _ in range(5):
    print(next(counter))
    
#%% 编写一个用于计时的装饰器
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)
        end_time = time.time()    # Record end time
        print(f"Function '{func.__name__}' executed in {end_time - start_time} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(2)

# Call the function
slow_function()