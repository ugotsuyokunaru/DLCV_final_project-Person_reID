# Make dataset directory
if ! [ -d "./pretrain" ]; then
    mkdir pretrain
fi


if ! [ -f "./pretrain/resnet50_ft_weight.pkl" ]; then
    # Download dataset from Dropbox
    wget https://www.dropbox.com/s/6tv01ngob44ihsr/resnet50_ft_weight.zip
    
    # Unzip the downloaded zip file
    unzip ./resnet50_ft_weight.zip

    # Remove the downloaded zip file
    rm ./resnet50_ft_weight.zip

    # move to target folder
    mv resnet50_ft_weight.pkl ./pretrain/
fi