# Make dataset directory
if ! [ -d $1 ]; then
    mkdir $1
fi

# Download training set
if ! [ -f "$1/train.zip" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YqeqpSSS_QHvEzIcsjot-pp5jQDZrHk8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YqeqpSSS_QHvEzIcsjot-pp5jQDZrHk8" -O "$1/train.zip" && rm -rf /tmp/cookies.txt
fi

# Download validation set
if ! [ -f "$1/val.zip" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1E63El9CKvm0YYntNp0M4uxIasgzzkXeD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E63El9CKvm0YYntNp0M4uxIasgzzkXeD" -O "$1/val.zip" && rm -rf /tmp/cookies.txt
fi

# Download test set
if ! [ -f "$1/test.zip" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dSzOCMHrMcSCcl0RSZQvhiFyYdVW9X0Q' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dSzOCMHrMcSCcl0RSZQvhiFyYdVW9X0Q" -O "$1/test.zip" && rm -rf /tmp/cookies.txt
fi

# Download Ground Truth
if ! [ -f "$1/val_GT.json" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Zr3B9e7Ra67nI9rFJ4JV-wXhJIUCEKHW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Zr3B9e7Ra67nI9rFJ4JV-wXhJIUCEKHW" -O "$1/val_GT.json" && rm -rf /tmp/cookies.txt
fi

if ! [ -f "$1/train_GT.json" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1L9IUHuqB6g1zlj81r7p8FMh091IVWOUJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1L9IUHuqB6g1zlj81r7p8FMh091IVWOUJ" -O "$1/train_GT.json" && rm -rf /tmp/cookies.txt
fi

# Download eval.py
if ! [ -f "./eval.py" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Fd-HDBK459ZrfF9YQRIstEXw3L-28IKS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Fd-HDBK459ZrfF9YQRIstEXw3L-28IKS" -O ./eval.py && rm -rf /tmp/cookies.txt
fi

# Unzip and remove the downloaded zip file
unzip "$1/train.zip" -d $1
unzip "$1/val.zip" -d $1
unzip "$1/test.zip" -d $1

rm ./$1/train.zip
rm ./$1/val.zip
rm ./$1/test.zip
