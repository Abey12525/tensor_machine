from PIL import Image
import numpy as np

im=Image.open('16.png','r')
pix=im.load()
imm=list(im.getdata())
print(imm)
s=im.size
m=np.array(im)
print(m[0,3])
m[0,3]=(m[0,3])*4
print(m[0,3])
"""for i in range(s[0]):
    for j in range(s[1]):
        print(m[i,j])
        
   """     
   
   
test_image=np.array([1,.0293,.15235,.83463,.3513,.94532,.2451,.57284,.234573,.245462,.2678,.8664,.0421,.3923,.3456,.7833,.7742,.4567,.4256,.2364,.2369,.7796,.5963,.4589,.358])
   
print(test_image.size)      
