## ROCK, PAPER, SCISSORS ##
rock = '''
    _______
---'   ____)
      (_____)
      (_____)
      (____)
---.__(___)
'''

paper = '''
    _______
---'   ____)____
          ______)
          _______)
         _______)
---.__________)
'''

scissors = '''
    _______
---'   ____)____
          ______)
       __________)
      (____)
---.__(___)
'''
# yc = your choice
yc = int(input("Type:\n 0 for 'ROCK'\n 1 for 'PAPER'\n 2 for 'SCISSORS'\n"))


print('You chose:')
if yc >= 3 or yc <0:
    print('You entered an invalid number. Try again.')
else:
    if yc == 0:
        print(rock)
    elif yc == 1:
        print(paper)
    else:
        print(scissors)
    import random as rn
    
    # cc = computer' choice
    cc = rn.randint(0, 2)
    
    print('Computer chose:')
    if cc == 0:
        print(rock)
    elif cc == 1:
        print(paper)
    else:
        print(scissors)
        
    chance = yc * 10 + cc
    if (chance == 2) or (chance == 10) or (chance == 21):
        print('You WIN!')
    elif (yc == cc):
        print('Same choice by both. Retry!')
    else:
        print('You LOSE!')