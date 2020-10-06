# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 23:33:28 2019

@author: Chiara Lanza
"""


import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import statistics
import functions_lab_2 as f
import matplotlib.patches as patches
import math
import os
#################################################################################
'''THE CODE REQUIRE FOLDS: 'per' 'ellipse', 'rect', 'rot', 'simm', 'bn'
    IN ORDER TO SAVE THE IMAGE  
'''


#These moles needs the function with the doctor choice
retry= ['low_risk_10.jpg','medium_risk_1.jpg', 'melanoma_5.jpg', 'melanoma_8.jpg', 'melanoma_9.jpg', 'melanoma_10.jpg', 'melanoma_11.jpg', 'melanoma_17.jpg', 'melanoma_21.jpg', 'melanoma_27.jpg']


comm=input("Which version of the tool do you want?\nFor the automatic tool with 3 principal centroids press '1'\nIf you want to decide the clustering version press '2'\n")
flag=True
if comm=='1' or comm=='2':
    flag=False
while flag==True:
    comm=input("Wrong input!\nWhich version of the tool do you want?1nFor the automatic tool with 3 principal centroids press '1'\nIf you want to decide the clustering version press '2'\n")
    if comm=='1' or comm=='2':
        flag=False
    
#this loop works on all the moles in the dataset (in the subfold lab2_moles)
for item in retry:#os.listdir("lab2_moles/"):
    #item="melanoma_4.jpg"
    original = "lab2_moles/"+item
    print(item)
    im_or=mpimg.imread(original)
    #the light version uses always 3 centroids and it works well and autonomous on most of the moles
    if comm=='1':
        im,percB=f.centroids3(im_or,original)
    #for 12 moles is necessary another version (so the other function) because 3 centroids are not enough
    #with this function the doctor can choose the best version between three different clustering
    if comm=='2':
        im,percB=f.Centroids(im_or,original)
    #the problematic moles are isolated in the vector retry, if you change the fold path in the loop with this vector 
    #the work is easier then retry the clustering on all the moles
    
    
    #this function makes the border of the image white because it's a lot of probably that the mole is in the center of the photo
    #and the border are noisy 
    cutIm=f.resize(percB, im)
    
    #a simply function for the area and vectorof coordinates of the moles
    posmole,posmolex,posmoley=f.posmolef(cutIm)
    
    #different functions that clean the image 
    posmole,im=f.cleanIm(cutIm,posmole,posmolex,posmoley)
    
    #we want the extremes of the mole
    vectx=np.zeros(posmole.shape[0], dtype=int)
    vecty=np.zeros(posmole.shape[0], dtype=int)
             
    for x in range(posmole.shape[0]):
        vectx[x]=int(posmole[x,0])
        vecty[x]=int(posmole[x,1])
    
    
    vectx.sort()
    vecty.sort()
    
    #function that clean the image for the last time and find the perimeter of the mole (the final area is compute too)
    im2,per,perList,posmole=f.perimetro(im,vectx[0],vecty[0],vectx[vectx.shape[0]-1],vecty[vecty.shape[0]-1],original)
    
    #another time we want the extrme of the mole and the medians of the coordinates 
    vectx=np.zeros(posmole.shape[0], dtype=int)
    vecty=np.zeros(posmole.shape[0], dtype=int)
             
    for x in range(posmole.shape[0]):
        vectx[x]=int(posmole[x,0])
        vecty[x]=int(posmole[x,1])
    
    
    vectx.sort()
    vecty.sort()
    
    
    mx=statistics.median(vectx)
    my=statistics.median(vecty)
    
    fig,ax = plt.subplots(1)
    ax.imshow(im2, cmap='viridis')
    plt.scatter(mx,my, color='r')
    rect=patches.Rectangle((vectx[0],vecty[0]),(vectx[vectx.shape[0]-1]-vectx[0]),(vecty[vecty.shape[0]-1]-vecty[0]), linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.title(original[11:-4])
    fig.savefig('folds/rect/'+original[11:-4]+'_bn')
    plt.show()
    plt.close(fig)
    
    
    fig2,ax = plt.subplots(1)
    ax.imshow(im_or, cmap='viridis')
    plt.scatter(mx,my, color='r')
    rect=patches.Rectangle((vectx[0],vecty[0]),(vectx[vectx.shape[0]-1]-vectx[0]+1),(vecty[vecty.shape[0]-1]-vecty[0]+1), linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.title(original)
    fig2.savefig('folds/rect/'+original[11:-4]+'_rect')
    plt.show()
    plt.close(fig2)
    
    #find the ration between the real perimeter compute on the image and the ideal one
    #of a perfect circle with the same area of the mole
    ideal=np.sqrt(posmole.shape[0]*4*np.pi)
    ratio=per/ideal
    print('perimetro: '+str(per))
    print('idealr: '+str(ideal))
    print('ratio: '+str(ratio))
    
    document = open('Ratio.txt', 'a')
    document.write(original[11:-4]+'\n'+str(ratio)+'\n\n')
    document.close() 
    
    #find the parameters of the image ellipse 
    xc,yc,alpha,beta,teta=f.imageEllipse(posmole,vectx[0],vecty[0],vectx[vectx.shape[0]-1],vecty[vecty.shape[0]-1])
    
    #plot the image ellipse on the original photo in order to check
    fig3,ax = plt.subplots(1)
    ax.imshow(im_or, cmap='viridis')
    ellipse=patches.Ellipse((xc,yc),beta,alpha,teta,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(ellipse)
    plt.title(original[11:-4]+' ellipse')
    fig3.savefig('folds/ellipse/'+original[11:-4]+'_im')
    plt.show()
    plt.close(fig3)
    
    #turn the image with the theta compute in the previous function
    imRot,ycR,xcR=f.turnIm(vectx[0],vecty[0],vectx[vectx.shape[0]-1],vecty[vecty.shape[0]-1],im2,teta,yc,xc)
    
    #plot the image rotate
    fig1=plt.figure()
    plt.imshow(imRot, cmap='viridis')
    plt.scatter(xcR,ycR, color='r')
    plt.title('Rotate')
    plt.show()
    fig1.savefig('folds/rot/'+original[11:-4]+'_im')
    plt.close(fig1)
    #find the symmetry index 
    f.symmetrixIndices(imRot,ycR,xcR,original)
