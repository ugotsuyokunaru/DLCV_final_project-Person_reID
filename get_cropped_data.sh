################################################################################################
# get test_resize.zip
file_id="1641vXY5DtlVWeKol6M4PtVkfdCnN7328"
file_name="test_resize.zip"

# first stage to get the warning html
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$file_id" > /tmp/intermezzo.html

# second stage to extract the download link from html above
download_link=$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')
curl -L -b /tmp/cookies "https://drive.google.com$download_link" > $file_name
################################################################################################


################################################################################################
# get Resize_val.tar.gz
file_id="1zGVd42yeKgs4Hl-15yM7VUSXqY5BYR39"
file_name="Resize_val.tar.gz"
    
# first stage to get the warning html
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$file_id" > /tmp/intermezzo.html

# second stage to extract the download link from html above
download_link=$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')
curl -L -b /tmp/cookies "https://drive.google.com$download_link" > $file_name
################################################################################################


################################################################################################
# get Resize_train.tar.gz
file_id="168wyQ2EgwnuTVCwzAkglRnIvY4vQWKxj"
file_name="Resize_train.tar.gz"
    
# first stage to get the warning html
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$file_id" > /tmp/intermezzo.html

# second stage to extract the download link from html above
download_link=$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')
curl -L -b /tmp/cookies "https://drive.google.com$download_link" > $file_name
################################################################################################

mkdir IMDb_resize

mv ./Resize_train.tar.gz ./IMDb_resize/Resize_train.tar.gz
mv ./Resize_val.tar.gz ./IMDb_resize/Resize_val.tar.gz
mv ./test_resize.zip ./IMDb_resize/test_resize.zip

# Unzip and remove the downloaded zip file
unzip "IMDb_resize/test_resize.zip" -d ./IMDb_resize
tar -xvzf "IMDb_resize/Resize_val.tar.gz" -C ./IMDb_resize
tar -xvzf "IMDb_resize/Resize_train.tar.gz" -C ./IMDb_resize

rm ./IMDb_resize/Resize_train.tar.gz
rm ./IMDb_resize/Resize_val.tar.gz
rm ./IMDb_resize/test_resize.zip

mv ./IMDb_resize/test_resize ./IMDb_resize/test
mv ./IMDb_resize/media/disk1/EdwardLee/dataset/IMDb_Resize/train/ ./IMDb_resize/train
mv ./IMDb_resize/media/disk1/EdwardLee/dataset/IMDb_Resize/val/ ./IMDb_resize/val
rm -r ./IMDb_resize/media/


# Download Ground Truth
if ! [ -f "IMDb_resize/val_GT.json" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Zr3B9e7Ra67nI9rFJ4JV-wXhJIUCEKHW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Zr3B9e7Ra67nI9rFJ4JV-wXhJIUCEKHW" -O "IMDb_resize/val_GT.json" && rm -rf /tmp/cookies.txt
fi

if ! [ -f "IMDb_resize/train_GT.json" ]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1L9IUHuqB6g1zlj81r7p8FMh091IVWOUJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1L9IUHuqB6g1zlj81r7p8FMh091IVWOUJ" -O "IMDb_resize/train_GT.json" && rm -rf /tmp/cookies.txt
fi