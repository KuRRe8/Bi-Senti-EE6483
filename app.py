import os
import subprocess
import time
import configparser

def setup_env():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_file_directory)

    config = configparser.ConfigParser()
    config.read(os.path.join("config", "runtime.ini"))
    config['general']['project_root'] = current_file_directory
    with open(os.path.join("config", "runtime.ini"), 'w') as configfile:
        config.write(configfile)
    
    if not os.path.exists("temp"):
        os.makedirs("temp")

def run_jupyter():
    print("Start Jupyter...")
    subprocess.Popen(["start", "cmd", "/k", "jupyter", "notebook", os.path.join("src","jupyter","full.ipynb")], shell=True)

def training():
    print("Running Program 2...")
    # Add your program 2 code here
    subprocess.Popen(["start", "cmd", "/k", "python"], shell=True)

def inference():
    print("Running Program 3...")
    # Add your program 3 code here
    subprocess.Popen(["start", "cmd", "/k", "python program3.py"], shell=True)

def frontend():
    print("Running Program 4...")
    # Add your program 4 code here
    subprocess.Popen(["start", "cmd", "/k", "python program4.py"], shell=True)

def main():

    setup_env()
    
    while True:
        print("\nWelcome to the EE6483 group project reception:")
        print("1. Open ipynb file(providing an interactive manner to find out our thought process)")
        print("2. Training routine(training the model, specified in config/runtime.ini)")
        print("3. Inference routine(get the prediction result)")
        print("4. Frontend(an Qt application to provide a user-friendly interface)")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            run_jupyter()
        elif choice == '2':
            training()
        elif choice == '3':
            inference()
        elif choice == '4':
            frontend()
        elif choice == '5':
            print("Thank you for using our program. See you next time!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(2)

if __name__ == "__main__":
    main()