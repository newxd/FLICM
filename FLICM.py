import numpy as np
import random
import cv2

def FLICM(img,c,N,alpha):


    [row, column] = img.shape
    n = img.size
    U = []
    for i in range(c):
        temp = []
        for j in range(n):
            temp.append(random.random())
        U.append(temp)
    U = np.array(U, dtype=float)
    #G=np.zeros([c,n])
    V = getV(img, U)
    G=getG(img,U,V,N)
    print 'initial G successful'
    Un=getU(img,V,G)
    t = 1
    while (np.sum(np.sum(np.abs(U - Un))) > alpha):
        print t, np.sum(np.sum(np.abs(U - Un)))
        t += 1
        V = getV(img, Un)
        G = getG(img, Un, V, N)
        U = Un.copy()
        Un = getU(img, V, G)


    color = np.linspace(0, 255, c)
    for i in range(row):
        for j in range(column):
            l = np.argsort(Un[:, i * column + j])
            img[i][j] = color[l[-1]]

    return img




def getV(img,U):
    n=img.size
    [row,colum]=img.shape
    V=[]
    for i in range(len(U)):
        a=np.sum(U[i])
        b=0.0
        for j in range(row):
            for k in range(colum):

                b+=((U[i][j*colum+k])**2*img[j][k])
        V.append(b/a)
    return V


#RFLICM
def getG(img,U,V,N):
    [row,column]=img.shape
    G=np.zeros([len(V),img.size])
    CU = np.zeros([N, N])
    t=(N-1)/2
    for i in range(t*2,row-t*2):
        for j in range(t*2,column-t*2):
            #temp=img[i-t:i+t+1][j-t:j+t+1]

            for m in range(-t,t+1):
                for n in range(-t,t+1):
                    CU[t+m][t+n]=getC(img,i+m,j+n,t)
            CU_mean=np.mean(CU)
            CU_mid=CU[t][t]
            for k in range(len(V)):
                #G[k][i*column+j]=
                Gnum=0.0
                for m in range(-t, t + 1):
                    for n in range(-t, t + 1):
                        cons=((1-U[k][(i+m)*column+j+n])**2*((img[i+m][j+n]-V[k])**2))
                        if CU_mid==0 or CU[t+m][t+n]==0:
                            CU_min=0.0
                        else:
                            CU_min=np.min([(CU[t+m][t+n]/CU_mid)**2,(CU_mid/CU[t+m][t+n])**2])
                        if CU[t+m][t+n]<CU_mean:
                            Gnum+=((1.0/(2-CU_min))*cons)
                        else:
                            Gnum+=((1.0/(2+CU_min))*cons)
                G[k][i * column + j] =Gnum
    return G

#FLICM
''''def getG(img,U,V,N):
    [row, column] = img.shape
    G = np.zeros([len(V), img.size])
    t = (N - 1) / 2
    for i in range(t , row - t ):
        for j in range(t , column - t ):
            # temp=img[i-t:i+t+1][j-t:j+t+1]
            #CU = np.zeros([N, N])
            for k in range(len(V)):
                # G[k][i*column+j]=
                Gnum = 0.0
                for m in range(-t, t + 1):
                    for n in range(-t, t + 1):
                        cons = ((1 - U[k][(i + m) * column + j + n]) ** 2 * ((img[i + m][j + n] - V[k]) ** 2))
                        #CU_min = np.min([(CU[t + m][t + n] / CU_mid) ** 2, (CU_mid / CU[t + m][t + n]) ** 2])
                        Dij=np.sqrt(m**2+n**2)
                        Gnum+=((1.0/(Dij+1))*cons)
                G[k][i * column + j] = Gnum
    return G'''''


def getC(img,i,j,t):
    temp=img[i-t:i+t+1,j-t:j+t+1]
    if np.mean(temp)==0:
        return 0
    else:
        return float(np.var(temp))/(np.mean(temp)**2)

def getU(img,V,G):
    c=len(V)
    [row,column]=img.shape
    n=img.size
    Un=np.zeros([c,n])
    for i in range(row):
        for j in range(column):
            for k in range(c):
                x = 0.0
                for l in range(c):
                    x += (((img[i][j] - V[k]) ** 2+G[k][i*column+j]) / ((img[i][j] - V[l]) ** 2+G[l][i*column+j]))
                Un[k][i * column + j] = 1.0 / x
    return Un

if __name__ == '__main__':
    img=cv2.imread('dataset/Otawwa_logratio.bmp',0)
    img=np.array(img,dtype=float)
    out=FLICM(img,2,3,0.05)
    cv2.imshow('out',out)

    cv2.imwrite('Otawwa_FLICM.bmp',out)
    cv2.waitKey(0)

