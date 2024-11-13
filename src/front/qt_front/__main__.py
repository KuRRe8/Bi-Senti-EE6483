import os
import app

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    print(os.getcwd())
    app.main()
