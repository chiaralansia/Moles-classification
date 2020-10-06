# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.stats as st
import math
plt.ion()

#the automatic function, which uses always 3 centroids and there is no need of intervent by the doctor
def centroids3(im_or,filein):
    #apply the kmeans algorithm (function) in order to have 3 (then 5 and 9) principal centroids
    kmeans= KMeans(n_clusters=3, random_state=0)
    [N1,N2,N3]=im_or.shape
    im_2D=im_or.reshape((N1*N2,N3))# N1*N2 rows and N3 columns
    
    kmeans.fit(im_2D)
    centroids=kmeans.cluster_centers_.astype('uint8')
    
    label3=kmeans.labels_
    
    newIm3=np.zeros((im_2D.shape[0],3), dtype=float)
    newIm3=newIm3.astype('uint8')
    c=np.ones((3,2), dtype=float)
    #the darkest centroid is the one with the smallest rgb triplet
    for i in range(3):
        c[i]=((int(centroids[i,0])+int(centroids[i,1])+int(centroids[i,2])),i)
        
    c=c[c[:,0].argsort()]
    
    newcentroids=np.ones((3,3), dtype=int)
    #we want just the darkest color and so we set to black it and white the other ones
    newcentroids[int(c[0,1])]=[0,0,0]
    newcentroids[int(c[1,1])]=[255,255,255]
    newcentroids[int(c[2,1])]=[255,255,255]
    #the number of black pixel is then used to cut the image, in order to delete black parts of the image which are not mole
    nb3=0 
    for i in range(0,label3.shape[0]-1):
        if label3[i]==0:
            nb3+=1
            newIm3[i]=newcentroids[0]
        if label3[i]==1:
            newIm3[i]=newcentroids[1]
        if label3[i]==2:
            newIm3[i]=newcentroids[2]
    im3=newIm3.reshape((N1,N2,N3)) 
    percB=nb3/newIm3.shape[0]
    
    fig1=plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im_or, cmap='viridis')
    plt.title(filein[11:-4])
    
    plt.subplot(1,2,2)
    plt.imshow(im3, cmap='viridis')
    plt.title('1')
    plt.show()
    fig1.savefig('folds/bn/'+filein[11:-4]+'_im')
    plt.close(fig1)
    return im3,percB

#the complex function which needs the doctor choose of the clustering
def Centroids(im_or,filein):
    im=im_or
    #apply the kmeans algo (function) in order to have 3 (then 5 and 9) principal centroids
    kmeans= KMeans(n_clusters=3, random_state=0)
    [N1,N2,N3]=im_or.shape
    im_2D=im_or.reshape((N1*N2,N3))# N1*N2 rows and N3 columns
    
    kmeans.fit(im_2D)
    centroids=kmeans.cluster_centers_.astype('uint8')
    
    label3=kmeans.labels_
    
    newIm3=np.zeros((im_2D.shape[0],3), dtype=float)
    newIm3=newIm3.astype('uint8')
    c=np.ones((3,2), dtype=float)
    for i in range(3):
        c[i]=((int(centroids[i,0])+int(centroids[i,1])+int(centroids[i,2])),i)
        
    c=c[c[:,0].argsort()]
    
    newcentroids=np.ones((3,3), dtype=int)
    #we want just the darkest color and so I set to black it and white the other ones
    newcentroids[int(c[0,1])]=[0,0,0]
    newcentroids[int(c[1,1])]=[255,255,255]
    newcentroids[int(c[2,1])]=[255,255,255]
    #the number of black pixel is then used to cut the image, in order to delete black parts of the image which are not mole
    nb3=0 
    for i in range(0,label3.shape[0]-1):
        if label3[i]==0:
            nb3+=1
            newIm3[i]=newcentroids[0]
        if label3[i]==1:
            newIm3[i]=newcentroids[1]
        if label3[i]==2:
            newIm3[i]=newcentroids[2]
    im3=newIm3.reshape((N1,N2,N3)) 
    percB=nb3/newIm3.shape[0]
    
    kmeans= KMeans(n_clusters=5, random_state=0)
    [N1,N2,N3]=im_or.shape
    im_2D=im_or.reshape((N1*N2,N3))# N1*N2 rows and N3 columns
    
    kmeans.fit(im_2D)
    centroids5=kmeans.cluster_centers_.astype('uint8')
    label5=kmeans.labels_
    
    
    c=np.ones((5,2), dtype=float)
    #sorting the centroids by the darkest color (the minimum sum of the rgb) we can try to isolate the mole
    for i in range(5):
        c[i]=((int(centroids5[i,0])+int(centroids5[i,1])+int(centroids5[i,2])),i)
        
    c=c[c[:,0].argsort()]
    
    newcentroids=np.ones((5,3), dtype=int)
    #we want just the darkest color and so we set to black it and white the other ones
    newcentroids[int(c[0,1])]=[0,0,0]
    newcentroids[int(c[1,1])]=[255,255,255]
    newcentroids[int(c[2,1])]=[255,255,255]
    newcentroids[int(c[3,1])]=[255,255,255]
    newcentroids[int(c[4,1])]=[255,255,255]
    
    nb5=0
    newIm5=np.zeros((im_2D.shape[0],3), dtype=float)
    newIm5=newIm5.astype('uint8')
    for i in range(0,label5.shape[0]-1):
        if label5[i]==0:
            nb5+=1
            newIm5[i]=newcentroids[0]
        if label5[i]==1:
            #nb+=1
            newIm5[i]=newcentroids[1]
        if label5[i]==2:
            newIm5[i]=newcentroids[2]                  
        if label5[i]==3:
            newIm5[i]=newcentroids[3]
        if label5[i]==4:
            newIm5[i]=newcentroids[4]
    percB5=nb5/newIm5.shape[0]
    im5=newIm5.reshape((N1,N2,N3))

    kmeans= KMeans(n_clusters=9, random_state=0)
    [N1,N2,N3]=im_or.shape
    im_2D=im_or.reshape((N1*N2,N3))# N1*N2 rows and N3 columns
    
    kmeans.fit(im_2D)
    centroids6=kmeans.cluster_centers_.astype('uint8')
    label6=kmeans.labels_
    
    
    c=np.ones((9,2), dtype=float)
    #sorting the centroids by the darkest color (the minimum sum of the rgb) we can try to isolate the mole
    for i in range(9):
        c[i]=((int(centroids6[i,0])+int(centroids6[i,1])+int(centroids6[i,2])),i)
        
    c=c[c[:,0].argsort()]
    
    newcentroids=np.ones((9,3), dtype=int)
    #we want just the darkest color and so we set to black it and white the other ones
    newcentroids[int(c[0,1])]=[0,0,0]
    newcentroids[int(c[1,1])]=[0,0,0]
    newcentroids[int(c[2,1])]=[0,0,0]
    newcentroids[int(c[3,1])]=[255,255,255]
    newcentroids[int(c[4,1])]=[255,255,255]
    newcentroids[int(c[5,1])]=[255,255,255]
    newcentroids[int(c[6,1])]=[255,255,255]
    newcentroids[int(c[7,1])]=[255,255,255]
    newcentroids[int(c[8,1])]=[255,255,255]
    
    nb6=0
    newIm6=np.zeros((im_2D.shape[0],3), dtype=float)
    newIm6=newIm6.astype('uint8')
    for i in range(0,label6.shape[0]-1):
        if label6[i]==0:
            nb6+=1
            newIm6[i]=newcentroids[0]
        if label6[i]==1:
            nb6+=1
            newIm6[i]=newcentroids[1]
        if label6[i]==2:
            nb6+=1
            newIm6[i]=newcentroids[2]                  
        if label6[i]==3:
            newIm6[i]=newcentroids[3]
        if label6[i]==4:
            newIm6[i]=newcentroids[4]
        if label6[i]==5:
            newIm6[i]=newcentroids[5]
        if label6[i]==6:
            newIm6[i]=newcentroids[6]
        if label6[i]==7:
            newIm6[i]=newcentroids[7]
        if label6[i]==8:
            newIm6[i]=newcentroids[8]
    percB6=nb6/newIm6.shape[0]
    im6=newIm6.reshape((N1,N2,N3))
    #plot the three clustering in order to make the doctor choose
    fig1=plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im_or, cmap='viridis')
    plt.title(filein[11:-4])
    
    plt.subplot(1,2,2)
    plt.imshow(im3, cmap='viridis')
    plt.title('1')
    plt.show()
    
    fig2=plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im_or, cmap='viridis')
    plt.title(filein[11:-4])
    
    plt.subplot(1,2,2)
    plt.imshow(im5, cmap='viridis')
    plt.title('2')
    plt.show()
    
    fig3=plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im_or, cmap='viridis')
    plt.title(filein[11:-4])
    
    plt.subplot(1,2,2)
    plt.imshow(im6, cmap='viridis')
    plt.title('3')
    plt.show()
    
    
    
    #the doctor can choose the best version, because even if most of them are almost perfect with 3 centroids, some of them need more
    
    comm=input("Which version is the most accurate?\npossible answers: '1' '2' or '3'\n"+filein+"\n")
    flag=True
    if comm=='1' or comm=='2' or comm=='3':
        flag=False
    while flag==True:
        comm=input("Wrong input!\nWhich version is the most accurate?\npossible answers: '1' '2' or '3'\n"+filein+"\n")
        if comm=='1' or comm=='2' or comm=='3':
            flag=False

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    #the output image and percentuage of the black pixels depend on the choice of the doctor
    if comm=='1':
        fig1.savefig('folds/bn/'+filein[11:-4]+'_c3')
        im=im3
        percB=percB
    if comm=='2':
        fig2.savefig('folds/bn/'+filein[11:-4]+'_c5')
        im=im5
        percB=percB5
    if comm=='3':
        im=im6
        percB=percB6
        fig3.savefig('folds/bn/'+filein[11:-4]+'_c9')
    

    return im, percB



def resize(percB, im):
    met=(1-percB)/3.75 #cut the image in  proportional product of the black %
    
    plt.imshow(im, cmap='viridis')
    plt.title('Before resize')
    plt.show()
    
    #for each x
    for i in range(0,int(im.shape[0]-1)):
        #'delete' (make white) the lowest part (y)
        for j in range(int(im.shape[1]*met)):
            im[i,j,:]=[255,255,255]
            #'delete' (make white) the upper part (y)
        for j in range(int(im.shape[1]*(1-met)), im.shape[0]):
            im[i,j,:]=[255,255,255]
    
    #for each y
    for i in range(0,im.shape[1]-1):
        #'delete' (make white) the right part (x)
        for j in range(int(im.shape[0]*met)):
            im[j,i,:]=[255,255,255]
        #'delete' (make white) the left part (x)
        for j in range(int(im.shape[0]*(1-met)), im.shape[0]):
            im[j,i,:]=[255,255,255]
    #this plot was udes for the report explaination setting to yellow the cutted part
    plt.imshow(im, cmap='viridis')
    plt.title('Resized image')
    plt.show(block=True)
    
    return im


def posmolef(im):
    posMolex=list()
    posMoley=list()
    
    for i in range(0,im.shape[0]-1):
        for j in range(0,im.shape[1]-1):
            #easly if the pixel is black, it's part of the mole
            if (im[i,j,0]==0 or im[i,j,0]==255) and im[i,j,1]==0 and im[i,j,2]==0:
                posMolex.append(j)
                posMoley.append(i)
    #in order to have a matrix of x and y with the perfect dimension 
    posmole=np.zeros((len(posMolex),2))            
    for x in range(len(posMolex)):
        posmole[x]=[int(posMolex[x]),int(posMoley[x])]
    return posmole,posMolex,posMoley

def cleanIm(cutIm,posmole,posMolex,posMoley):
    z = np.abs(st.zscore(posmole))
    for x in range(posmole.shape[1]):
        #delete the black pixel in a statostical way, 4 is choosen because in a gaussian distibution (which we hypotized we have)
        #the outlier are out of more or less 3 times the dev standard
        #but some images can be well isolate, if we use more then 4 some of the mole will be delete
        if z[x,0]+z[x,1]>4:
            cutIm[int(posmole[x,1]),int(posmole[x,0]),:]=[255,255,255]
            posMolex.remove(posmole[x,0])
            posMoley.remove(posmole[x,1])

    #if a black pixel is isolated, so there is no other black pixel around it, in the perimere count it cost 8 pixel, even if it's not part of the mole
    #it's better delete them
    for x in range(len(posmole)):
        i=int(posmole[x,1])
        j=int(posmole[x,0])
        if cutIm[i,j,1]==0 and cutIm[i+1,j,1]==255 and cutIm[i+1,j+1,1]==255 and cutIm[i+1,j-1,1]==255 and cutIm[i-1,j,1]==255 and cutIm[i-1,j-1,1]==255 and cutIm[i-1,j+1,1]==255 and cutIm[i,j+1,1]==255 and cutIm[i,j-1,1]==255:
            cutIm[i,j,:]=[255,255,255]
            posMolex.remove(posmole[x,0])
            posMoley.remove(posmole[x,1])
    #the mole area is now changed
    posmole=np.zeros((len(posMolex),2))            
    for x in range(len(posMolex)):
        posmole[x]=[int(posMolex[x]),int(posMoley[x])]
          
    #with the same statistical idea of the cleaning, but in a contrary way
    #if a pixel has an hig probability to be there, so it's under 3.3 dev standard
    #and one of his neighboor is white, make it black and add it to the area (the two lists of x and y)                       
    z = np.abs(st.zscore(posmole)) 
    for x in range(len(posmole)):   
        if z[x,0]+z[x,1]<3.3:
            if cutIm[int(posmole[x,1])+1,int(posmole[x,0])+1,1]!=0:
                cutIm[int(posmole[x,1])+1,int(posmole[x,0])+1,:]=[0,0,0]
                posMolex.append(int(posmole[x,0])+1)
                posMoley.append(int(posmole[x,1])+1)
            if cutIm[int(posmole[x,1])-1,int(posmole[x,0])+1,1]!=0:
                cutIm[int(posmole[x,1])-1,int(posmole[x,0])+1,:]=[0,0,0]
                posMolex.append(int(posmole[x,0])-1)
                posMoley.append(int(posmole[x,1])+1)
            if cutIm[int(posmole[x,1]),int(posmole[x,0])+1,1]!=0:
                cutIm[int(posmole[x,1]),int(posmole[x,0])+1,:]=[0,0,0]
                posMolex.append(int(posmole[x,0]))
                posMoley.append(int(posmole[x,1])+1)
            if cutIm[int(posmole[x,1])+1,int(posmole[x,0]),1]!=0:
                cutIm[int(posmole[x,1])+1,int(posmole[x,0]),:]=[0,0,0]
                posMolex.append(int(posmole[x,0])+1)
                posMoley.append(int(posmole[x,1]))
            if cutIm[int(posmole[x,1])+1,int(posmole[x,0])-1,1]!=0:
                cutIm[int(posmole[x,1])+1,int(posmole[x,0])-1,:]=[0,0,0]
                posMolex.append(int(posmole[x,0])+1)
                posMoley.append(int(posmole[x,1])-1)
            if cutIm[int(posmole[x,1]),int(posmole[x,0])-1,1]!=0:
                cutIm[int(posmole[x,1]),int(posmole[x,0])-1,:]=[0,0,0]
                posMolex.append(int(posmole[x,0]))
                posMoley.append(int(posmole[x,1])-1)
            if cutIm[int(posmole[x,1])-1,int(posmole[x,0])-1,1]!=0:
                cutIm[int(posmole[x,1])-1,int(posmole[x,0])-1,:]=[0,0,0]
                posMolex.append(int(posmole[x,0])-1)
                posMoley.append(int(posmole[x,1])-1)
            if cutIm[int(posmole[x,1])-1,int(posmole[x,0]),1]!=0:
                cutIm[int(posmole[x,1])-1,int(posmole[x,0]),:]=[0,0,0]
                posMolex.append(int(posmole[x,0])-1)
                posMoley.append(int(posmole[x,1]))
    #rebuilt the matrix of the mole positions
    #the points are in order so that we can rebuilt a vector with the positions of the black pixels
    posmole=np.zeros((len(posMolex),2))            
    for x in range(len(posMolex)):
        posmole[x]=[int(posMolex[x]),int(posMoley[x])]

    return posmole,cutIm 


                    
def perimetro(im_or,xi,yi,xf,yf,original):
    
    im=im_or.copy()
    im2=im_or.copy()
    per=list()
    #the range is between the limits of the rectange, but +-5 because they touch the mole in the estremes pixels
    for  i in range(yi-5,yf+5):
        for j in range(xi-5,xf+5):
            #simply check if the pixel is not white (blck or red)
            if im[i,j,1]==0:
                #if one the 8 pixel around the selected is white, it's part of the perimeter
                if ( im[i+1,j,1]==255 or im[i+1,j+1,1]==255 or im[i+1,j-1,1]==255 or im[i-1,j,1]==255 or im[i-1,j-1,1]==255 or im[i-1,j+1,1]==255 or im[i,j+1,1]==255 or im[i,j-1,1]==255):
                    #make them red in order to recognize them
                    im[i,j]=[255,0,0]
                    #save them in a list
                    per.append(str(i)+','+str(j))
    #just a list is created in the previus lines, we want a matrix because they are easier to work with
    perMat=np.zeros((len(per),2),dtype=int)
    for x in range(len(per)):
        perMat[x]=[int(per[x].split(',')[0]),int((per[x].split(',')[1]))]     
        
    fig2=plt.figure()
    plt.imshow(im, cmap='viridis')
    plt.title('Perimeter before cleaning')
    plt.show()
    fig2.savefig('folds/per/'+original[11:-4]+'_perBeforeCleaning')
    plt.close(fig2)
    #matrix cleaner of the mole
    #now that we have border black pixel, some of them are part of internal holes
    for i in range(perMat.shape[0]):
        flag=False
        flagOut=False
        add=[1,1,1,1]
        point=perMat[i]
        
        #built a rectangle stared from all the border pixel
        while flagOut==False:
            #save add values in order to loop on the first value and not the update ones 
            add0=add[0]
            add1=add[1]
            add2=add[2]
            add3=add[3]
            #if it's out of the extremes of the mole, it's not inside
            if (point[0]-add[0])<=yi or (point[0]+add[1])>=yf or (point[1]-add[2])<=xi or (point[1]+add[3])>=xf:               
                flagOut=True
            elif flag==False:
                flag=True
                for j in range((int(point[0])-add0),(int(point[0])+add1)):
                    #if there are white pixels, make the rectangle bigger in that direction
                    if im2[j,int(point[1])-add2,0]==255 and im2[j,int(point[1])-add2,1]==255:
                        add[2]+=1
                        flag=False
                    if im2[j,int(point[1])+add3,0]==255 and im2[j,int(point[1])+add3,1]==255:
                        add[3]+=1
                        flag=False
                for j in range((int(point[1])-add2),(int(point[1])+add3)):
                    if im2[int(point[0])-add0,j,0]==255 and im2[int(point[0])-add0,j,1]==255:
                        add[0]+=1
                        flag=False
                    if im2[int(point[0])+add1,j,0]==255 and im2[int(point[0])+add1,j,1]==255:
                        add[1]+=1
                        flag=False
                #if the flag was not update to false it means we have a rectangle with all the border in the mole
                #the inside can be setted to black
                if flag==True:
                    if (point[0]-add[0])>=yi and (point[0]+add[1])<=yf and (point[1]-add[2])>=xi and (point[1]+add[3])<=xf:
                        '''plt.figure()
                        plt.imshow(im2, cmap='viridis')
                        plt.scatter((int(point[0])),(int(point[1])), color='r')
                        plt.show(block=True)'''
                        for x in range((int(point[0])-add0),(int(point[0])+add1)):
                            for y in range((int(point[1])-add2),(int(point[1])+add3)):
                                im2[x,y]=[0,0,0]
                    '''plt.figure()
                    plt.imshow(im2, cmap='viridis')
                    plt.show(block=True)'''
                    #now we can pass on another point of the 'perimeter'
                    flagOut=True
        #the same idea is applied in the opposite way in order to delete the 'island' not part of the mole
        #outside the mole figure
        add=[1,1,1,1]
        flag=False
        flagOut=False
        while flagOut==False and add[2]<35 and add[3]<35 and add[0]<35 and add[1]<35:
            add0=add[0]
            add1=add[1]
            add2=add[2]
            add3=add[3]
            if flag==False:
                flag=True
                for j in range((int(point[0])-add0),(int(point[0])+add1)):
                    '''plt.figure()
                    plt.imshow(im, cmap='viridis')
                    plt.scatter(j,int(point[1])-add[2], color='r')
                    plt.show(block=True)'''
                    #if there is a black pixel we are still in the mole or in the island that has to be delete
                    if ((im2[j,int(point[1])-add2,0]==0 or im2[j,int(point[1])-add2,0]==255) and im2[j,int(point[1])-add2,1]==0) :
                        add[2]+=1
                        flag=False
                    if ((im2[j,int(point[1])+add3,0]==0 or im2[j,int(point[1])+add3,0]==255) and im2[j,int(point[1])+add3,1]==0) :
                        add[3]+=1
                        flag=False
                for j in range((int(point[1])-add2),(int(point[1])+add3)):
                    if ((im2[int(point[0])-add0,j,0]==0 or im2[int(point[0])-add0,j,0]==255) and im2[int(point[0])-add0,j,1]==0) :
                        add[0]+=1
                        flag=False
                    if ((im2[int(point[0])+add1,j,0]==0 or im2[int(point[0])+add1,j,0]==255) and im2[int(point[0])+add1,j,1]==0) :
                        add[1]+=1
                        flag=False
                #if the flag was not update to false it means we have a rectangle with all the border outside the mole
                #the inside can be setted to white
                if flag==True and add[2]<35 and add[3]<35 and add[0]<35 and add[1]<35:
                    for x in range((int(point[0])-add[0]),(int(point[0])+add[1])):
                        for y in range((int(point[1])-add[2]),(int(point[1])+add[3])):
                            im2[x,y]=[255,255,255]
                    flagOut=True
    per=list()
    for  i in range(yi-5,yf+5):
        for j in range(xi-5,xf+5):
            #simply check if the pixel is not white (black or red)
            if im2[i,j,1]==0:
                #if one the 8 pixel around the selected is white, it is part of the perimeter
                if ( im2[i+1,j,1]==255 or im2[i+1,j+1,1]==255 or im2[i+1,j-1,1]==255 or im2[i-1,j,1]==255 or im2[i-1,j-1,1]==255 or im2[i-1,j+1,1]==255 or im2[i,j+1,1]==255 or im2[i,j-1,1]==255):
                    #if they are not all black, is perimeter
                    im2[i,j]=[255,0,0]
                    per.append(str(i)+','+str(j))
    #matrix is easier to manage
    outputPer=np.zeros((im.shape[0],im.shape[1]),dtype=int)
    perMat=np.zeros((len(per),2),dtype=int)
    for x in range(len(per)):
        perMat[x]=[int(per[x].split(',')[0]),int((per[x].split(',')[1]))] 
        outputPer[perMat[x][0],perMat[x][1]]=1
    posmole=list()
    for  i in range(yi-5,yf+5):
        for j in range(xi-5,xf+5):
            if (im2[i,j,0]==0 or im2[i,j,0]==255) and im2[i,j,1]==0 :
                posmole.append(str(j)+','+str(i))
    
    plt.matshow(outputPer, fignum='3')
    plt.title('Perimeter')
    plt.show() 
    #plot after the perimeter detection and the final cleaning 
    fig1=plt.figure()
    plt.imshow(im2, cmap='viridis')
    plt.title('Image with perimeter')
    plt.show()
    fig1.savefig('folds/per/'+original[11:-4]+'_per')
    plt.close(fig1)
    posmoleMat=np.zeros((len(posmole),2), dtype=int)
    for i in range(len(posmole)):
        posmoleMat[i]=[int(posmole[i].split(',')[0]),int((posmole[i].split(',')[1]))]  
    return im2,len(per),perMat,posmoleMat


def imageEllipse(posmole, xi,yi,xf,yf):
    #studing the geometric moments
    #m00 is the area of themole
    mu00=posmole.shape[0]
    m10=0
    for x in range(posmole.shape[0]):
        m10+=posmole[x,0]
    m01=0
    for x in range(posmole.shape[0]):
        m01+=posmole[x,1]
    #the coordinates of the centroid
    xc=m10/posmole.shape[0]
    yc=m01/posmole.shape[0]
    
    #central moments invariant to the translation
    mu20=0
    for x in range(posmole.shape[0]):
        mu20+=pow((posmole[x,0]-xc),2)
    mu02=0
    for x in range(posmole.shape[0]):
        mu02+=pow((posmole[x,1]-yc),2)
    mu11=0
    for x in range(posmole.shape[0]):
        mu11+=(posmole[x,1]-yc)*(posmole[x,0]-xc)
    #parameter for the image ellipse
    beta=pow(np.divide(2*(mu20+mu02+np.sqrt(pow((mu20-mu02),2)+4*mu11)),mu00),1/2)
    alpha=pow(np.divide(2*(mu20+mu02-np.sqrt(pow((mu20-mu02),2)+4*mu11)),mu00),1/2)
    teta=(0.5)*np.arctan(np.divide((2*mu11),(+mu20-mu02)))
    
    print(teta)
    #alpha e beta are muplipy for 2 because they are used for the plot, where are needed the axes not semiaxes
    #theta is transfored in degree for the same reason
    return xc,yc,(2*alpha),(2*beta),math.degrees(teta)
    
    

def turnIm(xi,yi,xf,yf,im,teta, xc,yc):
    
    #we nees these vaues in order to not go outside the image and have out of range exceptions
    rows=im.shape[0]
    cols=im.shape[1]
    #coordinates of the center of the image.
    mid_coords = np.floor(0.5*np.array(im.shape))
    
    def rotate_pixel(pixel_coords, cos_angle, sin_angle):
        #Translate the coordinates so that the center of rotation coincides with the origin, by subtracting coordinates of the center of the image
        #Then the rotation is applied with the rotation matrix
        xf = (pixel_coords[0]-mid_coords[0])*cos_angle-(pixel_coords[1]-mid_coords[1])*sin_angle+mid_coords[0]
        yf = (pixel_coords[0]-mid_coords[0])*sin_angle+(pixel_coords[1]-mid_coords[1])*cos_angle+mid_coords[1]
    
        #Check if the new coordinates are out of the dimension of the image
        if 0<=int(np.round(xf))<rows and 0<=int(np.round(yf))<cols:
           return (int(np.round(xf)),int(np.round(yf)))
        else:
           return False
    
    rotated_img = np.zeros((rows,cols,3),dtype=int)
    rotated_img[:,:,:]=255
    cos_angle = np.cos(math.radians(teta))
    sin_angle = np.sin(math.radians(teta))
    for i in range(rows):
       for k in range(cols):
           coords = rotate_pixel((i,k), cos_angle, sin_angle)
           if(coords):
               rotated_img[coords] = im[i][k]
        
    #rotate the centroid coordinates
    xcR,ycR=rotate_pixel((xc,yc), cos_angle, sin_angle)
    
    #The pixel boundaries are at quantized locations, sin and cos return real numbers, so we have to round
    #them, and the result is to miss some pixels making holes in the turned image.
    #The solution used is to set to black pixels which are white but surrounded by other black pixels.
    for i in range(rows-1):
       for j in range(cols-1):
           if rotated_img[i,k,0]==255:
               if (rotated_img[i+1,j,1]==0  and rotated_img[i-1,j,1]==0 and rotated_img[i,j+1,1]==0 and rotated_img[i,j-1,1]==0):
                    rotated_img[i,j]=[0,0,0]
        
    return rotated_img,xcR,ycR
    
    
def symmetrixIndices(imRot,xc,yc,original):
    areas=np.zeros(4, dtype=int)
    symms=np.zeros(4,dtype=int)
    #in order to plot what we are doing for the symmetry
    immv1=imRot.copy()
    immv2=imRot.copy()
    immv3=imRot.copy()
    #Q1
    for i in range(0,xc+1):
        for k in range(0,yc+1):
            if imRot[i,k,0]==0:
                areas[0]+=1
                #Q1 vs Q2
                if imRot[xc+(xc-i),k,0]==imRot[i,k,0]:
                    immv1[xc+(xc-i),k]=[0,255,255]
                    symms[0]+=1
                #Q1 vs Q4
                if imRot[i,yc-(k-yc),0]==imRot[i,k,0]:
                    immv1[i,yc-(k-yc)]=[0,255,255]
                    symms[1]+=1
        #Q4
        for k in range(yc,imRot.shape[1]):
            if imRot[i,k,0]==0:
                areas[3]+=1
                #Q4 VS Q3
                if imRot[xc+(xc-i),k,0]==imRot[i,k,0]:
                    immv2[xc+(xc-i),k]=[0,255,50]
                    symms[2]+=1
    #Q2
    for i in range(xc,imRot.shape[0]):
        for k in range(0,yc+1):
            if imRot[i,k,0]==0:
                areas[1]+=1
                #Q2 VS Q3
                if imRot[i,yc-(k-yc),0]==imRot[i,k,0]:
                    immv3[i,yc-(k-yc)]=[0,255,255]
                    symms[3]+=1
    #we need the area of q3
    for i in range(xc,imRot.shape[0]):
        for k in range(yc,imRot.shape[1]):
            if imRot[i,k,0]==0:
                areas[2]+=1
       
    #check the biggest area, the ratio will be done between this one and the matched area,
    #so it will be bigger then 1 every time
    if areas[0]>areas[1] and symms[0]>0:
        simm1=str(areas[0]/symms[0])
        print('1 VS 2: 1 '+str(areas[0])+' '+str(areas[0]/symms[0]))
    elif symms[0]>0:
        simm1=str(areas[1]/symms[0])
        print('1 VS 2: 2 '+str(areas[1])+' '+str(areas[1]/symms[0]))
    else:
        print('No matches between 1 and 2')
        simm1=2
    if areas[0]>areas[3] and symms[1]>0:
        simm3=str(areas[0]/symms[1])
        print('1 VS 4: 1 '+str(areas[0])+' '+str(areas[0]/symms[1]))
    elif symms[1]>0:
        simm3=str(areas[3]/symms[1])
        print('1 VS 4: 4 '+str(areas[3])+' '+str(areas[3]/symms[1]))
    else:
        print('No matches between 1 and 4')
        simm3=2
    if areas[1]>areas[2] and symms[3]>0:
        simm4=str(areas[1]/symms[3])
        print('2 vs 3:2 '+str(areas[1])+' '+str(areas[1]/symms[3]))
    elif symms[3]>0:
        simm4=str(areas[2]/symms[3])
        print('2 vs 3:3 '+str(areas[2])+' '+str(areas[2]/symms[3]))
    else:
        print('No matches between 2 and 3')
        simm4=2
    if areas[3]>areas[2] and symms[2]>0:
        simm5=str(areas[3]/symms[2])
        print('4 vs 3:4 '+str(areas[3])+' '+str(areas[3]/symms[2]))
    elif symms[2]>0:
        simm5=str(areas[2]/symms[2])
        print('4 vs 3:3 '+str(areas[2])+' '+str(areas[2]/symms[2]))
    else:
        print('No matches between 4 and 3')
        simm5=2
    
    #the symmetry index is the sum of the idexes minus 4 because they will be of cours bigger then 1
    symmetry=float(simm1)+float(simm3)+float(simm4)+float(simm5)-4
    print(symmetry)
    #save the symmetry index in a file
    document = open('Symmetry.txt', 'a')
    document.write(original[11:-4]+'\n'+str(symmetry)+'\n\n')
    document.close()
    #plot and save all the symmetry evaluations
    fig1=plt.figure()
    plt.imshow(immv1, cmap='viridis')
    plt.scatter(yc,xc, color='r')
    plt.title('Symmetry q1')
    plt.show()
    fig1.savefig('folds/simm/'+original[11:-4]+'_symm1')
    plt.close(fig1)
    
    fig2=plt.figure()
    plt.imshow(immv2, cmap='viridis')
    plt.scatter(yc,xc, color='r')
    plt.title('Symmetry q2')
    plt.show()
    fig2.savefig('folds/simm/'+original[11:-4]+'_symm2')
    plt.close(fig2)
    
    fig3=plt.figure()
    plt.imshow(immv3, cmap='viridis')
    plt.scatter(yc,xc, color='r')
    plt.title('Symmetry q3')
    plt.show()
    fig3.savefig('folds/simm/'+original[11:-4]+'_symm3')
    plt.close(fig3)
    
    