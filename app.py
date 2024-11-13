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

def results():
    choice = input("Show the result url press 1, show directly press 2: ")
    if choice == '1':
        print("submit\\out\\submission.csv")
        time.sleep(3)
    elif choice == '2':
        csv_path = os.path.join("submit", "out", "submission.csv")
        subprocess.Popen(["start", "cmd", "/k", "python", "-c", f"import pandas as pd; print(pd.read_csv('{csv_path}'))"], shell=True)
    else:
        print("Invalid choice. Please try again.")
        time.sleep(2)

def gradio_frontend():
    print("Opening Gradio...")
    gr_path = os.path.join("src", "front", "gradio_front", "get_sentiment_box.py")
    subprocess.Popen(["start", "cmd", "/k", f"python {gr_path}"], shell=True)

def qt_frontend():
    print("Opening Qt...")
    qt_path = os.path.join("src", "front", "qt_front", "__main__.py")
    subprocess.Popen(["start", "cmd", "/k", f"python {qt_path}"], shell=True)

def main():

    setup_env()
    
    while True:
        print("\nWelcome to the EE6483 group project reception:")
        print("1. Open ipynb file(providing an interactive manner to find out our thought process)")
        print("2. Check for results from finetuned model(csv file)")
        print("3. Frontend(gradio lightweight web UI)")
        print("4. Frontend(an Qt application to provide a user-friendly interface)")
        print("5. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            run_jupyter()
        elif choice == '2':
            results()
        elif choice == '3':
            gradio_frontend()
        elif choice == '4':
            qt_frontend()
        elif choice == '5':
            print("Thank you for using our program. See you next time!")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(2)

if __name__ == "__main__":
    main()