import random

# Assuming variables_list is your list of variables
# and excluding_var is the variable you want to exclude
variables_list = ['var1', 'var2', 'var3', 'var4', 'var5']
excluding_var = 'var3'

# Select three random variables from the variables_list excluding the excluding_var
random_vars = [var for var in random.sample(variables_list, k=3) if var != excluding_var]

print("Random variables:", random_vars)
