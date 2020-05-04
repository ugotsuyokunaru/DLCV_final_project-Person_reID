# Make dataset directory
if ! [ -d "./models" ]; then
    mkdir models    
fi

# Get resnet50.pth
if ! [ -f "./models/resnet50.pth" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YqeqpSSS_QHvEzIcsjot-pp5jQDZrHk8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18O78hXzBioW8igG9WZMX5W9spqEAysOE" -O "./models/resnet50.pth" && rm -rf /tmp/cookies.txt
fi

# Get classifier.pth
if ! [ -f "./models/classifier.pth" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YqeqpSSS_QHvEzIcsjot-pp5jQDZrHk8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Oc5WH7MJe4Xa9qzh9_MXIF-zBHUxIacd" -O "./models/classifier.pth" && rm -rf /tmp/cookies.txt
fi
