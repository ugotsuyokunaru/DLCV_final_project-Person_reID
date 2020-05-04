# Make dataset directory
if ! [ -d $1 ]; then
    mkdir $1
fi

# Download market-1501 dataset
if ! [ -f "$1/Market-1501-v15.09.15.zip" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B8-rUzbwVRk0c054eEozWG9COHM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B8-rUzbwVRk0c054eEozWG9COHM" -O "$1/Market-1501-v15.09.15.zip" && rm -rf /tmp/cookies.txt
fi

# Remove the downloaded zip file
unzip "$1/Market-1501-v15.09.15.zip" -d $1

rm ./$1/Market-1501-v15.09.15.zip
