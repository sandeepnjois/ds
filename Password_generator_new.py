## PASSWORD GENERATOR PROJECT ##

import random as rn
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

print("Welcome to the PyPassword Generator!")
nr_letters= int(input("How many letters would you like in your password?\n")) 
nr_symbols = int(input(f"How many symbols would you like?\n"))
nr_numbers = int(input(f"How many numbers would you like?\n"))

#Eazy Level - Order not randomised:
#e.g. 4 letter, 2 symbol, 2 number = JduE&!91

# Method 1
# Sequential password
p_letters = ''
for letter in range(0, nr_letters):
    p_letters += letters[rn.randint(0, 51)]
    
p_numbers = ''
for number in range(0, nr_numbers):
    p_numbers += numbers[rn.randint(0, 9)]
    
p_symbols = ''
for symbol in range(0, nr_symbols):
    p_symbols += symbols[rn.randint(0, 8)] 
    
pw = p_letters+p_symbols+p_numbers

str_var = list(pw)
rn.shuffle(str_var)
password_shuffled = ''.join(str_var)


print(f'Your password is {password_shuffled}')
#Hard Level - Order of characters randomised:
#e.g. 4 letter, 2 symbol, 2 number = g^2jk8&P

