import torch
import torch.nn as nn
import torch.nn.functional as F
from generation import generate, tokenizer, model
from tokenizer import Tokenizer
from model import TransformerModel

def startscreen():
    ### Title screen

    while True: 
        print("Frozen Palace: Non-AI Version")
        print("START?")
        print("(Press (1) for Start!)")
        print("(Press (2) for Quit!)")
        startChoice = input("> ")
    
        if (startChoice == "1"):
            adventure_game()
        elif (startChoice == "2"):
            print("Goodbye!")
            break
        else:
            print("Invalid Choice")

def adventure_game():

    ### Printing a welcome message
    print("You have arrived in the Frozen Palace.")
    print("After hearing of many civilians never returning after visiting, you've decided to investigate.")
    print("As the mayor of the nearby town, you have a duty to.")
    print("With 2 adventurers in tow, you set off!")

    ### First choice
    print("After passing through the gate into the courtyard...")
    print("...there are 3 paths. Ahead, left and right.")


    ### Prompting player with a choice
    roomChoice = input("> ")

    if (roomChoice == "Ahead"):
    
        hasLight = False
    
        print("Going in through the main door, you find yourself in the main hall.")
        print("The hall in front of you and the foyer are empty.")
        print("There are no signs of life. You notice a single candle and lighter on the console table.")
        print("Take them?")
        print("(1. Yes)")
        print("(2. No)")

        ### Choice for player (ahead path)
        choice = input("> ")
        if (choice == "1"):
            hasLight = True
            print("You grabbed the candle and lighter!")

        elif(choice == "2"):
            hasLight = False
            print("You left them behind as you travelled on.")

        else:
            print("Invalid choice. Please pick Yes or No")
    

        if hasLight:
            prompt = f"""
            LOCATION: The hall of the Frozen Palace. 
            STATE: torch=true
            ACTION: Walk forward with candle and lighter, searching for missing civilians.
            RESULT:
            """
            result = generate(model, tokenizer, prompt)
            print(result)
        else:
            prompt = f"""
            LOCATION: The hall of the Frozen Palace. 
            STATE: torch=false
            ACTION: Walk forward fumbling in the darkness, searching for missing civilians.
            RESULT:
            """
            result = generate(model, tokenizer, prompt)
            print(result)

    
        print(result)
    

    elif(roomChoice == "Left"):
        print("On the left side of the courtyard is a side door leading to an observatory.")
        print("Though there are windows, they're pitch black.")
        print("Enter?")
        print("(1. Yes)")
        print("(2. No)")

        ### First choice for player on Left route
        choice = input("> ")
        if (choice == "1"):
            print("")
    
        elif(choice == "2"):
            print("")
    
        else:
            print("Invalid choice. Please pick Yes or No")


    elif(roomChoice == "Right"):
        print("")
        print("")
        print("")

    else:
        print("Invalid choice. Please enter Ahead, Left or Right")

adventure_game()