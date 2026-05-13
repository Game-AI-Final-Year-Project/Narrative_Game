import torch
import torch.nn as nn
import torch.nn.functional as F
from generation import generate, tokenizer, model
from tokenizer import Tokenizer
from model import TransformerModel
import time
import sys

def slow_print(text):

    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.02)

    print()

def startscreen():
    ### Title screen

    while True: 
        slow_print("Frozen Palace: AI Version")
        slow_print("START?")
        slow_print("(Press (1) for Start!)")
        slow_print("(Press (2) for Help!)")
        slow_print("(Press (3) for Quit!)")
        startChoice = input("> ")
    
        if (startChoice == "1"):
            adventure_game()
        elif (startChoice == "2"):
            show_help()
        elif (startChoice == "3"):
            slow_print("Goodbye! Press Ctrl+C to exit!")
        else:
            slow_print("Invalid Choice")

def show_help():

    slow_print("\n=== HELP ===")
    slow_print("Type the number of your choice.")
    slow_print("Explore the Frozen Palace.")
    slow_print("Survive and uncover the mystery of the missing persons.\n")

    input("Press Enter to return...")

def adventure_game():

    ### Printing a welcome message
    slow_print("You have arrived in the Frozen Palace.")
    slow_print("After hearing of many civilians never returning after visiting, you've decided to investigate.")
    slow_print("As the mayor of the nearby town, you have a duty to.")
    slow_print("With 2 adventurers in tow, you set off!")

    ### First choice
    slow_print("After passing through the gate into the courtyard...")
    slow_print("...there are 3 paths. Ahead, left and right.")
    slow_print("(Type [Ahead], [Left], or [Right] to choose)")

    ### Prompting player with a choice
    roomChoice = input("> ")

    if (roomChoice == "Ahead" or "ahead"):
    
        hasLight = False

        slow_print("After instructing the two adventurers to go through the ")
        slow_print("Going in through the main door, you find yourself in the main hall.")
        slow_print("The hall in front of you and the foyer are empty.")
        slow_print("There are no signs of life. You notice a single candle and lighter on the console table.")
        slow_print("Take them?")
        slow_print("(1. Yes)")
        slow_print("(2. No)")

        ### Choice for player (ahead path)
        choice = input("> ")
        if (choice == "1"):
            hasLight = True
            slow_print("You grabbed the candle and lighter!")

        elif(choice == "2"):
            hasLight = False
            slow_print("You left them behind as you travelled on.")

        else:
            slow_print("Invalid choice. Please pick Yes or No")
    

        if hasLight:
            prompt = f"""
            LOCATION: The hall of the Frozen Palace. 
            STATE: torch=true
            ACTION: Walk forward with candle and lighter, searching for missing civilians.
            RESULT:
            """
            result = generate(model, tokenizer, prompt)
            slow_print(result)
        else:
            prompt = f"""
            LOCATION: The hall of the Frozen Palace. 
            STATE: torch=false
            ACTION: Walk forward fumbling in the darkness, searching for missing civilians.
            RESULT:
            """
            result = generate(model, tokenizer, prompt)
            slow_print(result)

    
        slow_print(result)
    

    elif(roomChoice == "Left"):
        slow_print("On the left side of the courtyard is a side door leading to an observatory.")
        slow_print("Though there are windows, they're pitch black.")
        slow_print("Due to this, you call one adventurer over to investigate with you.")
        slow_print("Enter?")
        slow_print("(1. Yes)")
        slow_print("(2. No)")

        ### First choice for player on Left route
        choice = input("> ")
        if (choice == "1"):
            slow_print("")
    
        elif(choice == "2"):
            slow_print("")
    
        else:
            slow_print("Invalid choice. Please pick Yes or No")


    elif(roomChoice == "Right"):
        slow_print("The right side of the courtyard leads to a kitchen.")
        slow_print("You tell both adventures to search the front and left side of the palace.")
        slow_print("")

    else:
        slow_print("Invalid choice. Please enter Ahead, Left or Right")

startscreen()
adventure_game()