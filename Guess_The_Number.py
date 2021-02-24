### NUMBER GUESSING GAME ###

print("Welcome to the 'NUMBER GUESSING GAME!")
level = int(input("Choose level- \nType: '1' for Easy, '2' for Moderate or '3' for Hard: "))
import random as rn
number = rn.randint(1,100)

if level == 1:
    p = 15
elif level == 2:
    p = 10
elif level == 3:
    p = 5

def guess_num():
    i = p
    while i > 0:
        print(f"You have {i} attempts to guess")
        global guess
        guess = int(input('Guess a number between 1 and 100: '))
        if guess > number:
            print('Too high')
        elif guess < number:
            print('Too low')
        elif guess == number:
            i = 0
            print('You guessed it right! Thanks for playing! ')
        i = i - 1
        
        
guess_num()
print("Game over!")
    
    


