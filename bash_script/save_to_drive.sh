#!/bin/bash
cd /content/$1
# This script copies the contents of the FysikCamp2023 folder to the Google Drive folder
# Change the year here
source_folder="/content/"$1


if [ ! -d "/content/drive/MyDrive/Colab Notebooks" ]; then
  echo "Du har glemt at gemme i Google Drive. Så tryk CTRL + S for gemme i Google Drive, og kør derefter denne celle igen."
  exit 1
fi


echo "Copying main folder to Google Drive"
# Folder to exclude from copying
excluded_files=(".gitignore" "README.md" "HER.ipynb" "LICENSE" "requirements.txt" ".DS_Store" ".git")
#Change the year here
destination_folder="/content/drive/MyDrive/Colab Notebooks/"$1
# Copy folders and contents excluding the specified folder and files, without overwriting existing files
rsync -av --exclude="bash_script" --exclude=".git" --exclude-from=<(printf "%s\n" "${excluded_files[@]}") --ignore-existing "$source_folder/" "$destination_folder/"

echo "Finished creating main folder"


echo "Copying source folder to Google Drive"
excluded_files=(".gitignore" "README.md")
#Change the year here
destination_folder="/content/drive/MyDrive/Colab Notebooks/"$1"/source"
# Copy folders and contents excluding the specified folder and files
rsync -av --exclude="bash_script" --exclude=".git" --exclude-from=<(printf "%s\n" "${excluded_files[@]}") "$source_folder/" "$destination_folder/"

echo "Finished creating source folder"


echo "Finished"