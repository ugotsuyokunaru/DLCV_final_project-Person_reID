import torch
import torch.nn as nn

criterion = nn.TripletMarginLoss(margin=1)

def triplet_loss(inputs, labels, cast_num):
    '''
        Calculate Triplet Loss by feature input
        
        inputs: tensor        _(num_cast + num_cand) x 2048
        labels: tensor(int)   _(num_cast + num_cand) x 1
        cast_num: int         _ number of casts
    '''
    bs = inputs.size()[0]
    candid_num = bs-cast_num
    x_p = inputs[:cast_num]     # positive : casts
    x_a = inputs[cast_num:]     # anchor   : candidates (including "others" )
    
    print("labels :", labels)
    print("cast_num :", cast_num)
    print("anchors   : (candidates) labels[cast_num:] :", labels[cast_num:])
    print("positives : (casts)      labels[:cast_num] :", labels[:cast_num])

    loss = 0.0
    for i in range(candid_num):
        cand_label = labels[i+cast_num]
        num_p = labels.index(cand_label)
        print('[{} vs {}]'.format(i+cast_num, num_p))
        num_n = int(torch.randint(0, cast_num, (1,)))   # num_n : range (0 ~ cast_num-1)
        
        if cand_label==labels[num_n]: 
            # if get same label, pick "other" as x_n
            # if get same label, pick "the last cast" as x_n
            num_n = cast_num
            print('\n\n\n\n"others" = {}(ind {})'.format(labels[num_n], num_n))
            
        x_n = inputs[num_n]
        print('tri_loss a(cand) = {}, p(cast) = {}, n(n) = {}\n'.format(cand_label, labels[num_p], labels[num_n]))
        loss += criterion(x_a[i],x_p[num_p-cast_num],x_n)
    loss /= candid_num
        
    return loss
