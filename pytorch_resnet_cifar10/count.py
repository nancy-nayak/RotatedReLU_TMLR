# ResNet110_identitymapping


import numpy as np


# # ####  ResNet20 #############################################################################
# _list20 = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 10]
# _listrrelu = [16, 16, 16, 16, 16, 16, 15, 32, 31, 32, 32, 32, 32, 64, 64, 64, 64, 64, 40, 10]
# # _listrrelutwostep =  [16, 16, 15, 16, 15, 14, 15, 30, 32, 31, 30, 26, 31, 61, 61, 57, 61, 61, 64, 10]
# # _listrrelu = _listrrelutwostep
# # _listrrelutwostep_f = [16, 16, 15, 16, 16, 13, 15, 32, 27, 30, 28, 27, 29, 58, 61, 56, 62, 61, 64, 10]
# # _listrrelu = _listrrelutwostep_f
# # gamma=0.1 CIFAR10

# # ## Network slimming
# # k = 3
# # h_out_h_0 = 32
# # h_out_w_0 = 32
# # h_out_h_1 = 16
# # h_out_w_1 = 16
# # h_out_h_2 = 8
# # h_out_w_2 = 8

# flop20 = 0
# memory20 = 0
# flop = 0
# memory = 0

# # For resnet20
# for i in range(len(_listrrelu)):
#     if i==0:
#         flop20 += 2*_list20[i]*3*3*3*32*32
#         memory20 += _list20[i]*3*3*3
#         flop += 2*_listrrelu[i]*3*3*3*32*32
#         memory += _listrrelu[i]*3*3*3
    

#     if i>0 and i<=6:
#         flop20 += 2*_list20[i]*_list20[i-1]*3*3*32*32
#         memory20 += _list20[i]*_list20[i-1]*3*3
#         flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*32*32
#         memory += _listrrelu[i]*_listrrelu[i-1]*3*3

    
#     if i>6 and i<=12:
#         flop20 += 2*_list20[i]*_list20[i-1]*3*3*16*16
#         memory20 += _list20[i]*_list20[i-1]*3*3
#         flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*16*16
#         memory += _listrrelu[i]*_listrrelu[i-1]*3*3
    
#     if i>12 and i<=18:
        
#         flop20 += 2*_list20[i]*_list20[i-1]*3*3*8*8
#         memory20 += _list20[i]*_list20[i-1]*3*3
#         flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*8*8
#         memory += _listrrelu[i]*_listrrelu[i-1]*3*3

#     if i==19:
#         flop20 += 2*_list20[i]*_list20[i-1]
#         memory20 += _list20[i]*_list20[i-1]
#         flop += 2*_list20[i]*_listrrelu[i-1]
#         memory += _list20[i]*_listrrelu[i-1]

# print(flop20, flop, memory20, memory)
# exit()



# ####  ResNet56 #############################################################################
# _list56 = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 100]
# # _listrrelu = [16, 6, 4, 7, 5, 11, 8, 10, 7, 12, 11, 14, 13, 14, 14, 13, 11, 15, 11, 32, 21, 32, 22, 32, 21, 32, 24, 32, 28, 32, 32, 32, 31, 32, 31, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 64, 64, 64, 64, 64, 34, 10]
# _listrrelu = [16, 13, 5, 9, 3, 6, 3, 9, 3, 0, 4, 11, 6, 7, 5, 11, 8, 14, 11, 32, 16, 32, 23, 32, 27, 32, 27, 32, 31, 32, 30, 32, 31, 32, 30, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 100]
# # _listrrelu = [16, 13, 5, 9, 3, 6, 3, 9, 3, 0, 4, 11, 6, 7, 5, 11, 8, 14, 11, 32, 16, 32, 23, 32, 27, 32, 27, 32, 31, 32, 30, 32, 31, 32, 30, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 100]
# # print(len(_list56), len(_listrrelu))
# # _listrrelutwostep = [16, 16, 16, 15, 16, 14, 16, 16, 15, 16, 16, 15, 16, 16, 16, 14, 15, 16, 13, 32, 29, 25, 31, 31, 30, 29, 30, 31, 27, 29, 32, 31, 30, 29, 29, 29, 31, 58, 61, 58, 60, 58, 60, 58, 58, 60, 61, 59, 58, 55, 59, 55, 62, 57, 64, 10]
# # _listrrelu = _listrrelutwostep
# # _listrrelutwostep_f = [16, 16, 16, 14, 15, 14, 16, 15, 16, 14, 16, 16, 15, 15, 14, 13, 16, 16, 13, 29, 29, 31, 30, 29, 29, 28, 28, 31, 32, 29, 31, 29, 29, 27, 31, 30, 26, 58, 58, 60, 60, 58, 56, 61, 62, 59, 58, 57, 59, 56, 61, 60, 63, 58, 64, 10]
# # _listrrelu = _listrrelutwostep_f
# # _listrrelu = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
# # # gamma=0.04


# ## Network slimming
# k = 3
# h_out_h_0 = 32
# h_out_w_0 = 32
# h_out_h_1 = 16
# h_out_w_1 = 16
# h_out_h_2 = 8
# h_out_w_2 = 8

# flop56 = 0
# memory56 = 0
# flop = 0
# memory = 0

# # For resnet110
# for i in range(len(_listrrelu)):
#     if i==0:
#         flop56 += 2*_list56[i]*3*3*3*32*32
#         memory56 += _list56[i]*3*3*3
#         flop += 2*_listrrelu[i]*3*3*3*32*32
#         memory += _listrrelu[i]*3*3*3
#         print("------------------------------------------")
#         print("input to conv", 3, "Output from conv", _list56[i])
#         print("input to conv", 3, "Output from conv", _listrrelu[i])
   

#     if i>0 and i<=18:
#         flop56 += 2*_list56[i]*_list56[i-1]*3*3*32*32
#         memory56 += _list56[i]*_list56[i-1]*3*3
#         if i%2!=0:
#             # print(i, _list56[i-1], _listrrelu[i])
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*32*32
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#             print("------------------------------------------")
#             print("ReLU: input to conv", _list56[i-1], "Output from conv", _list56[i])
#             print("RReLU: input to conv", _list56[i-1], "Output from conv", _listrrelu[i])

#         else:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*32*32
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#             print("------------------------------------------")
#             print("ReLU: input to conv", _list56[i-1], "Output from conv", _list56[i])
#             print("RReLU: input to conv", _listrrelu[i-1], "Output from conv", _listrrelu[i])

#     if i>18 and i<=36:
#         flop56 += 2*_list56[i]*_list56[i-1]*3*3*16*16
#         memory56 += _list56[i]*_list56[i-1]*3*3
#         if i%2!=0:
#             print(i, _list56[i-1], _listrrelu[i])
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*16*16
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         else:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*16*16
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
    
#     if i>36 and i<=54:      
#         flop56 += 2*_list56[i]*_list56[i-1]*3*3*8*8
#         memory56 += _list56[i]*_list56[i-1]*3*3
#         if i%2!=0:
#             print(i, _list56[i-1], _listrrelu[i])
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*8*8
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         else:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*8*8
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3

#     if i==55:
#         flop56 += 2*_list56[i]*_list56[i-1]
#         memory56 += _list56[i]*_list56[i-1]
#         # print(i, _list56[i-1], _listrrelu[i])
#         flop += 2*_listrrelu[i]*_listrrelu[i-1]
#         memory += _listrrelu[i]*_listrrelu[i-1]

# print(flop56, flop, memory56, memory)
# exit()


# # # ### Wideresnet 40-4 #####################################################################

# _listwrn = [16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 10]

# # _listrrelu = [16, 61, 16, 38, 20, 49, 15, 42, 27, 38, 25, 38, 62, 127, 54, 82, 37, 63, 50, 60, 44, 58, 36, 40, 128, 256, 153, 236, 171, 191, 172, 156, 129, 120, 57, 42, 256, 10]
# # _listrrelu =  [15, 57, 32, 58, 31, 54, 28, 48, 22, 40, 22, 35, 64, 128, 42, 101, 81, 108, 56, 108, 22, 57, 64, 85, 128, 256, 215, 256, 243, 256, 248, 256, 251, 254, 243, 248, 256, 10]

# # CIFAR10
# # _listrrelu = [16, 61, 16, 38, 20, 49, 15, 42, 27, 38, 25, 38, 62, 127, 54, 82, 37, 63, 50, 60, 44, 58, 36, 40, 128, 256, 153, 236, 171, 191, 172, 156, 129, 120, 57, 42, 256, 10]
# # CIFAR100
# _listrrelu = [15, 57, 32, 58, 31, 54, 28, 48, 22, 40, 22, 35, 64, 128, 42, 101, 81, 108, 56, 108, 22, 57, 64, 85, 128, 256, 215, 256, 243, 256, 248, 256, 251, 254, 243, 248, 256, 100]
# # ## gamma=0.04

# # print(len(_listwrn), len(_listrrelu))
# # exit()

# ## Network slimming
# k = 3
# h_out_h_0 = 32
# h_out_w_0 = 32
# h_out_h_1 = 16
# h_out_w_1 = 16
# h_out_h_2 = 8
# h_out_w_2 = 8

# flopwrn = 0
# memorywrn = 0
# flop = 0
# memory = 0

# for i in range(len(_listrrelu)):
#     if i==0:
#         flopwrn += 2*_listwrn[i]*3*3*3*32*32
#         memorywrn += _listwrn[i]*3*3*3
#         flop += 2*_listrrelu[i]*3*3*3*32*32
#         memory += _listrrelu[i]*3*3*3

#     if i==0:
#         flopwrn += 2*_listwrn[i]*64*1*1*32*32
#         memorywrn += _listwrn[i]*64*1*1
#         flop += 2*_listrrelu[i]*64*1*1*32*32
#         memory += _listrrelu[i]*64*1*1
    

#     if i>0 and i<=12:
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*32*32
#         memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#         # flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*32*32
#         # memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i%2!=0:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*32*32
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         else:
#             flop += 2*_listwrn[i]*_listrrelu[i-1]*3*3*32*32
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
            
    
#     if i>12 and i<=24:
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*16*16
#         memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#         # flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*16*16
#         # memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i%2!=0:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*16*16
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         else:
#             flop += 2*_listwrn[i]*_listrrelu[i-1]*3*3*16*16
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i==13:
#             flopwrn += 2*_listwrn[i]*128*1*1*16*16
#             memorywrn += _listwrn[i]*128*1*1
#             flop += 2*_listrrelu[i]*128*1*1*16*16
#             memory += _listrrelu[i]*128*1*1
    
#     if i>24 and i<=36:      
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*8*8
#         memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#         # flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*8*8
#         # memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i%2!=0:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*8*8
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         else:
#             flop += 2*_listwrn[i]*_listrrelu[i-1]*3*3*8*8
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i==25:
#             flopwrn += 2*_listwrn[i]*256*1*1*8*8
#             memorywrn += _listwrn[i]*256*1*1
#             flop += 2*_listrrelu[i]*256*1*1*8*8
#             memory += _listrrelu[i]*256*1*1

#     if i==37:
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]
#         memorywrn += _listwrn[i]*_listwrn[i-1]
#         flop += 2*_listrrelu[i]*_listrrelu[i-1]
#         memory += _listrrelu[i]*_listrrelu[i-1]

# print(flopwrn, flop, memorywrn, memory)
# exit()

# ### Wideresnet 16-4 #####################################################################

# _listwrn = [16, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 10]
# _listrrelu = [8, 45, 10, 23, 61, 128, 91, 123, 122, 252, 222, 131, 256, 10]
# ## gamma=0.04


# ## Network slimming
# k = 3
# h_out_h_0 = 32
# h_out_w_0 = 32
# h_out_h_1 = 16
# h_out_w_1 = 16
# h_out_h_2 = 8
# h_out_w_2 = 8

# flopwrn = 0
# memorywrn = 0
# flop = 0
# memory = 0

# for i in range(len(_listrrelu)):
#     if i==0:
#         flopwrn += 2*_listwrn[i]*3*3*3*32*32
#         memorywrn += _listwrn[i]*3*3*3
#         flop += 2*_listrrelu[i]*3*3*3*32*32
#         memory += _listrrelu[i]*3*3*3
    
#     if i==0:
#         flopwrn += 2*_listwrn[i]*64*1*1*32*32
#         memorywrn += _listwrn[i]*64*1*1
#         flop += 2*_listrrelu[i]*64*1*1*32*32
#         memory += _listrrelu[i]*64*1*1

#     if i>0 and i<=4:
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*32*32
#         memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#         # flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*32*32
#         # memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i%2!=0:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*32*32
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         else:
#             flop += 2*_listwrn[i]*_listrrelu[i-1]*3*3*32*32
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3

    
#     if i>4 and i<=8:
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*16*16
#         memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#         # flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*16*16
#         # memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i%2!=0:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*16*16
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         else:
#             flop += 2*_listwrn[i]*_listrrelu[i-1]*3*3*16*16
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i==5:
#             flopwrn += 2*_listwrn[i]*128*1*1*32*32
#             memorywrn += _listwrn[i]*128*1*1
#             flop += 2*_listrrelu[i]*128*1*1*32*32
#             memory += _listrrelu[i]*128*1*1
    
#     if i>8 and i<=12:      
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*8*8
#         memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#         # flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*8*8
#         # memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i%2!=0:
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*8*8
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         else:
#             flop += 2*_listwrn[i]*_listrrelu[i-1]*3*3*8*8
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i==9:
#             flopwrn += 2*_listwrn[i]*256*1*1*32*32
#             memorywrn += _listwrn[i]*256*1*1
#             flop += 2*_listrrelu[i]*256*1*1*32*32
#             memory += _listrrelu[i]*256*1*1

#     if i==13:
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]
#         memorywrn += _listwrn[i]*_listwrn[i-1]
#         flop += 2*_listrrelu[i]*_listrrelu[i-1]
#         memory += _listrrelu[i]*_listrrelu[i-1]

# print(flopwrn, flop, memorywrn, memory)



# # # ### Wideresnet 50-2 #####################################################################

# _listwrn = [64, 128, 128, 256, 128, 128, 256, 128, 128, 256, 256, 256, 512, 256, 256, 512, 256, 256, 512, 256, 256, 512, 512, 512, 1024, 512, 512, 1024, 512, 512, 1024, 512, 512, 1024, 512, 512, 1024, 512, 512, 1024, 1024, 1024, 2048, 1024, 1024, 2048, 1024, 1024, 2048, 1000]

# # _listrrelu = [16, 61, 16, 38, 20, 49, 15, 42, 27, 38, 25, 38, 62, 127, 54, 82, 37, 63, 50, 60, 44, 58, 36, 40, 128, 256, 153, 236, 171, 191, 172, 156, 129, 120, 57, 42, 256, 10]
# # _listrrelu =  [64, 123, 122, 142, 103, 122, 210, 122, 123, 245, 254, 255, 233, 255, 249, 275, 255, 256, 426, 255, 256, 458, 512, 512, 279, 512, 512, 349, 512, 512, 524, 512, 512, 875, 512, 512, 990, 512, 512, 986, 1024, 1024, 538, 1024, 1024, 517, 1024, 1024, 467, 1000]
# # _listrrelu =  [64, 123, 122, 142, 103, 122, 210, 122, 123, 245, 254, 255, 233, 255, 249, 275, 255, 256, 426, 255, 256, 458, 512, 512, 279, 512, 512, 349, 512, 512, 524, 512, 512, 875, 512, 512, 990, 512, 512, 986, 1024, 1024, 538, 1024, 1024, 517, 1024, 1024, 467, 1000]

# _listrrelu =  [64, 123, 122, 142, 103, 122, 210, 122, 123, 245, 254, 255, 233, 255, 249, 275, 255, 256, 426, 255, 256, 458, 512, 512, 279, 512, 512, 349, 512, 512, 524, 512, 512, 875, 512, 512, 990, 512, 512, 986, 1024, 1024, 538, 1024, 1024, 517, 1024, 1024, 467, 1000]
# # ## gamma=0.04

# # print(len(_listwrn), len(_listrrelu))
# # exit()

# ## Network slimming


# flopwrn = 0
# memorywrn = 0
# flop = 0
# memory = 0

# # https://paperswithcode.com/lib/torchvision/wide-resnet#

# for i in range(len(_listrrelu)):
#     if i==0:
#         flopwrn += 2*_listwrn[i]*3*7*7*112*112
#         memorywrn += _listwrn[i]*3*7*7
#         flop += 2*_listrrelu[i]*3*7*7*112*112
#         memory += _listrrelu[i]*3*7*7



#     if i>0 and i<=9:
#         if i%3 == 1 or i%3 == 0:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-1]*1*1*56*56
#             memorywrn += _listwrn[i]*_listwrn[i-1]*1*1
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*1*1*56*56
#             memory += _listrrelu[i]*_listrrelu[i-1]*1*1
#         else:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*56*56
#             memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*56*56
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i == 4:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-3]*1*1*56*56
#             memorywrn += _listwrn[i]*_listwrn[i-3]*1*1
#             flop += 2*_listrrelu[i]*_listrrelu[i-3]*1*1*56*56
#             memory += _listrrelu[i]*_listrrelu[i-3]*1*1
        
            
    
#     if i>9 and i<=21:
#         if i%3 == 1 or i%3 == 0:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-1]*1*1*28*28
#             memorywrn += _listwrn[i]*_listwrn[i-1]*1*1
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*1*1*28*28
#             memory += _listrrelu[i]*_listrrelu[i-1]*1*1
#         else:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*28*28
#             memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*28*28
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3

#         if i == 13:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-3]*1*1*28*28
#             memorywrn += _listwrn[i]*_listwrn[i-3]*1*1
#             flop += 2*_listrrelu[i]*_listrrelu[i-3]*1*1*28*28
#             memory += _listrrelu[i]*_listrrelu[i-3]*1*1
        
    
#     if i>21 and i<=39: 
#         if i%3 == 1 or i%3 == 0:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-1]*1*1*14*14
#             memorywrn += _listwrn[i]*_listwrn[i-1]*1*1
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*1*1*14*14
#             memory += _listrrelu[i]*_listrrelu[i-1]*1*1
#         else:   
#             print(i)  
#             flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*14*14
#             memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*14*14
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i == 25:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-3]*1*1*14*14
#             memorywrn += _listwrn[i]*_listwrn[i-3]*1*1
#             flop += 2*_listrrelu[i]*_listrrelu[i-3]*1*1*14*14
#             memory += _listrrelu[i]*_listrrelu[i-3]*1*1
        
#     if i>39 and i<=48:      
#         if i%3 == 1 or i%3 == 0:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-1]*1*1*7*7
#             memorywrn += _listwrn[i]*_listwrn[i-1]*1*1
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*1*1*7*7
#             memory += _listrrelu[i]*_listrrelu[i-1]*1*1
#         else:  
#             print(i)   
#             flopwrn += 2*_listwrn[i]*_listwrn[i-1]*3*3*7*7
#             memorywrn += _listwrn[i]*_listwrn[i-1]*3*3
#             flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*7*7
#             memory += _listrrelu[i]*_listrrelu[i-1]*3*3
#         if i == 43:
#             print(i)
#             flopwrn += 2*_listwrn[i]*_listwrn[i-3]*1*1*7*7
#             memorywrn += _listwrn[i]*_listwrn[i-3]*1*1
#             flop += 2*_listrrelu[i]*_listrrelu[i-3]*1*1*7*7
#             memory += _listrrelu[i]*_listrrelu[i-3]*1*1

#     if i==49:
#         print(i)
#         flopwrn += 2*_listwrn[i]*_listwrn[i-1]
#         memorywrn += _listwrn[i]*_listwrn[i-1]
#         flop += 2*_listrrelu[i]*_listrrelu[i-1]
#         memory += _listrrelu[i]*_listrrelu[i-1]


# print(flopwrn, flop, memorywrn, memory)
# exit()


# ####  ResNet50 Imagenet #############################################################################
_list50 = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048, 1000]
_listrrelu = [64, 62, 62, 145, 59, 61, 215, 64, 63, 252, 126, 128, 308, 127, 127, 390, 128, 128, 493, 128, 128, 495, 256, 256, 341, 256, 256, 436, 256, 256, 561, 256, 256, 742, 256, 256, 834, 256, 256, 852, 512, 512, 1765, 512, 512, 2047, 512, 512, 680, 1000]
# print(len(_list50), len(_listrrelu))
# exit()
# gamma=0.1 CIFAR10

# ## Network slimming
# k = 3
# h_out_h_0 = 32
# h_out_w_0 = 32
# h_out_h_1 = 16
# h_out_w_1 = 16
# h_out_h_2 = 8
# h_out_w_2 = 8

flop50 = 0
memory50 = 0
flop = 0
memory = 0

for i in range(len(_listrrelu)):
    if i==0:
        flop50 += 2*_list50[i]*3*7*7*112*112
        memory50 += _list50[i]*3*7*7
        flop += 2*_listrrelu[i]*3*7*7*112*112
        memory += _listrrelu[i]*3*7*7



    if i>0 and i<=9:
        if i%3 == 1 or i%3 == 0:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-1]*1*1*56*56
            memory50 += _list50[i]*_list50[i-1]*1*1
            flop += 2*_listrrelu[i]*_listrrelu[i-1]*1*1*56*56
            memory += _listrrelu[i]*_listrrelu[i-1]*1*1
        else:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-1]*3*3*56*56
            memory50 += _list50[i]*_list50[i-1]*3*3
            flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*56*56
            memory += _listrrelu[i]*_listrrelu[i-1]*3*3
        if i == 4:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-3]*1*1*56*56
            memory50 += _list50[i]*_list50[i-3]*1*1
            flop += 2*_listrrelu[i]*_listrrelu[i-3]*1*1*56*56
            memory += _listrrelu[i]*_listrrelu[i-3]*1*1
        
            
    
    if i>9 and i<=21:
        if i%3 == 1 or i%3 == 0:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-1]*1*1*28*28
            memory50 += _list50[i]*_list50[i-1]*1*1
            flop += 2*_listrrelu[i]*_listrrelu[i-1]*1*1*28*28
            memory += _listrrelu[i]*_listrrelu[i-1]*1*1
        else:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-1]*3*3*28*28
            memory50 += _list50[i]*_list50[i-1]*3*3
            flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*28*28
            memory += _listrrelu[i]*_listrrelu[i-1]*3*3

        if i == 13:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-3]*1*1*28*28
            memory50 += _list50[i]*_list50[i-3]*1*1
            flop += 2*_listrrelu[i]*_listrrelu[i-3]*1*1*28*28
            memory += _listrrelu[i]*_listrrelu[i-3]*1*1
        
    
    if i>21 and i<=39: 
        if i%3 == 1 or i%3 == 0:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-1]*1*1*14*14
            memory50 += _list50[i]*_list50[i-1]*1*1
            flop += 2*_listrrelu[i]*_listrrelu[i-1]*1*1*14*14
            memory += _listrrelu[i]*_listrrelu[i-1]*1*1
        else:   
            print(i)  
            flop50 += 2*_list50[i]*_list50[i-1]*3*3*14*14
            memory50 += _list50[i]*_list50[i-1]*3*3
            flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*14*14
            memory += _listrrelu[i]*_listrrelu[i-1]*3*3
        if i == 25:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-3]*1*1*14*14
            memory50 += _list50[i]*_list50[i-3]*1*1
            flop += 2*_listrrelu[i]*_listrrelu[i-3]*1*1*14*14
            memory += _listrrelu[i]*_listrrelu[i-3]*1*1
        
    if i>39 and i<=48:      
        if i%3 == 1 or i%3 == 0:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-1]*1*1*7*7
            memory50 += _list50[i]*_list50[i-1]*1*1
            flop += 2*_listrrelu[i]*_listrrelu[i-1]*1*1*7*7
            memory += _listrrelu[i]*_listrrelu[i-1]*1*1
        else:  
            print(i)   
            flop50 += 2*_list50[i]*_list50[i-1]*3*3*7*7
            memory50 += _list50[i]*_list50[i-1]*3*3
            flop += 2*_listrrelu[i]*_listrrelu[i-1]*3*3*7*7
            memory += _listrrelu[i]*_listrrelu[i-1]*3*3
        if i == 43:
            print(i)
            flop50 += 2*_list50[i]*_list50[i-3]*1*1*7*7
            memory50 += _list50[i]*_list50[i-3]*1*1
            flop += 2*_listrrelu[i]*_listrrelu[i-3]*1*1*7*7
            memory += _listrrelu[i]*_listrrelu[i-3]*1*1

    if i==49:
        print(i)
        flop50 += 2*_list50[i]*_list50[i-1]
        memory50 += _list50[i]*_list50[i-1]
        flop += 2*_listrrelu[i]*_listrrelu[i-1]
        memory += _listrrelu[i]*_listrrelu[i-1]


print(flop50/2, flop/2, memory50, memory)
exit()