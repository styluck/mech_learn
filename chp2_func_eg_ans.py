# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 23:25:09 2024

@author: 6
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:49:55 2024

@author: 6
"""

#%% 利用生成器得到一个数列的平方
def square_nums(nums:list):
    for i in nums:
        yield (i*i)
        
def square_nums(nums:int):
    for i in range(nums):
        yield (i*i)

nums = [1, 2, 3, 4, 5]
snums = square_nums(nums)
print(next(snums))

#%% 利用生成器来构建无穷长度的数列
def infinite_counter(start=0):
    while True:
        yield start
        start += 1

# Using the infinite generator
counter = infinite_counter()

# To manually get the next value from a generator, 
# you can use the next() function. This function resumes 
# the generator from where it last yielded a value and 
# gets the next one.

# Print the first 5 values
for _ in range(5):
    print(next(counter))



#%% *args 
def my_function(*args):
    for arg in args:
        print(arg)

my_function(1, 2, 3)
my_function("a", "b", "c")

def sum_numbers(*args):
    return sum(args)

print(sum_numbers(1, 2, 3))  # Output: 6
print(sum_numbers(10, 20, 30, 40))  # Output: 100

def make_pizza(size, *toppings, crust="regular"):
    print(f"\nMaking a {size}-inch pizza with {crust} crust.")
    if toppings:
        print("Toppings:")
        for topping in toppings:
            print(f"- {topping}")

# Calling the function
make_pizza(12, "pepperoni", "mushrooms", crust="thin")

#%% 和 **kwargs

def my_function(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

my_function(name="Alice", age=25, city="New York")
my_function(**{'name':"Alice", 'age':25, 'city':"New York"})

def configure_settings(**kwargs):
    if "debug" in kwargs:
        print(f"Debug mode is {'on' if kwargs['debug'] else 'off'}")
    if "theme" in kwargs:
        print(f"Selected theme: {kwargs['theme']}")

configure_settings(debug=True, theme="dark")

# 两个混在一起用
def my_function(*args, **kwargs):
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)

my_function(1, 2, 3, name="Alice", age=25)

#%% 解包
def my_function(name, age, city):
    print(f"{name} is {age} years old and lives in {city}.")

info = {"name": "Alice", "age": 25, "city": "New York"}
my_function(**info)  # Unpacking the dictionary




#%% 上下文管理器
class MyContextManager:
    def __enter__(self):
        print("Entering the context...")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context...")
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return True  # Suppresses exceptions if True

# Using the custom context manager
with MyContextManager() as manager:
    print("Inside the context")
    # Uncomment the line below to raise an exception and see how it's handled
    # raise ValueError("Something went wrong!")

print("Outside the context")

#%% 发生错误时，仍然能够自行退出
class MyContextManager:
    def __enter__(self):
        print("Acquiring resource")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"Exception caught: {exc_value}")
        print("Releasing resource")
        return True  # Suppress exception

with MyContextManager() as manager:
    print("Inside the context")
    raise ValueError("Error occurred")  # This will trigger the __exit__ method

print("Program continues")

#%% 如何使用一个装饰器
def my_decorator(func):
    def wrapper(*args, **kwargs):
        # Code to execute before calling the original function
        print("Something before the function")

        # Call the original function
        result = func(*args, **kwargs)

        # Code to execute after calling the original function
        print("Something after the function")

        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# Equivalent to: say_hello = my_decorator(say_hello)
say_hello()

#%% 利用装饰器将一个函数重复n次
def repeat(n):  # n is the argument passed to the decorator
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)  # Calls the function 3 times
def foo():
    print("Hello!")

# Call the function
foo()

# 多个装饰器
def decorator1(func):
    def wrapper(*args, **kwargs):
        print("Decorator 1")
        return func(*args, **kwargs)
    return wrapper

def decorator2(func):
    def wrapper(*args, **kwargs):
        print("Decorator 2")
        return func(*args, **kwargs)
    return wrapper

@decorator1
@decorator2
def my_function():
    print("Inside my_function")

my_function()



