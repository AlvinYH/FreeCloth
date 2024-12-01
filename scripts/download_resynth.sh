#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~_-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
urld () { [[ "${1}" ]] || return 1; : "${1//+/ }"; echo -e "${_//%/\\x}"; }

read -p "Please enter the desired download option (packed / scans):" option
while [[ $option != 'packed' ]] && [[ $option != 'scans' ]]
do
    read -p "Please choose between 'packed' and 'scans': " option 
done 

read -p "Username:" username
read -p "Password:" password
username=$(urle $username)
password=$(urle $password)

mkdir -p "data/resynth"

printf "\n\nDownloading all $option data to the folder $PWD/data/resynth.\n\n"

# download data for 5 subjects
wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=pop&sfile=$option/rp_anna_posed_001.tar.gz" -O "data/resynth/rp_anna_posed_001.tar.gz"
wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=pop&sfile=$option/rp_beatrice_posed_025.tar.gz" -O "data/resynth/rp_beatrice_posed_025.tar.gz"
wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=pop&sfile=$option/rp_christine_posed_027.tar.gz" -O "data/resynth/rp_christine_posed_027.tar.gz"
wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=pop&sfile=$option/rp_felice_posed_004.tar.gz" -O "data/resynth/rp_felice_posed_004.tar.gz"
wget --post-data "username=$username&password=$password" "https://download.is.tue.mpg.de/download.php?domain=pop&sfile=$option/rp_janna_posed_032.tar.gz" -O "data/resynth/rp_janna_posed_032.tar.gz"

echo Unzipping the data...

FILES="data/resynth/*.tar.gz"
for f in $FILES
do
  tar -xvf $f
  rm $f
done

# Move the data to the correct folder
mv is/cluster/qma/work/SCALE2POP/data/resynth/packed/* data/resynth/
rm -rf is