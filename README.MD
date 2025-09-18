# created a ML Project folder

# Creating a virtual environment
# in windows is
python --version to check python version
To clear the terminal we use "clear" command
python -m venv mlpro to create a virtual environment

# Activate the virtual environment
<!-- ./ven_name/Scripts/activate -->
./mlpro/Scripts/activate
# incase you get an error while activating Virtual Environment
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser


pwd - present working directory
cd - to move to a specific directory
ls - to list all files
cd .. to come out of the current directory

# to install all libraries from requirements file
cd backend # enter the backend
pip install -r requirements.txt # install
cd .. # come out of backend

cd frontend # enter the frontend
pip install -r requirements.txt # install

# git init
# git add backend/ frontend/ README.MD
# git commit -m "v1"

git remote add origin https://github.com/mujeebr/Ml_Project_Bidar.git
git branch -M main
git push -u origin main


# from second time

git add modified files
# git commit -m "v1"
git push -u origin main