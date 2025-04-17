import random

triad = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
bot_options = {1: "rock", 2: "paper", 3: "scissors"}
player_points = 0
bot_points = 0

while True:
    print("===")
    player_choice = input("Your pick is:   ").lower()
    bot_choice = bot_options[random.randint(1,3)]

    if player_choice.lower() == "exit":
        break
    
    print(f"The Bot Picked: {bot_choice}")

    if triad[player_choice] == bot_choice:
        player_points += 1
        print("\nPlayer Won!")
    elif player_choice == bot_choice:
        print("\nIt's a Tie!")
    else:
        bot_points += 1
        print("\nPlayer Lost!")

    print(f"Score: Player-{player_points} Bot-{bot_points}")