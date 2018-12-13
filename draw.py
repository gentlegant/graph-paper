import  matplotlib.pyplot as plt
import numpy as np
import random
nums=[]
start=0
over=100
len1=2000
nums3=[]
for i in range(10):
    tmp=start+np.random.rand(len1)*(over-start)
    len1=int(len1/1.5)
    start=over
    over+=100
    nums3.append(random.randint(0,100))
    nums+=list(tmp)
    nums.append(1000)
numscp=nums.copy()
random.shuffle(numscp)


plt.subplot(311)
plt.bar(range(len(numscp)),numscp)

plt.subplot(312)
plt.bar(range(len(nums)),nums)
plt.subplot(313)
plt.bar(range(len(nums3)),nums3)
plt.show()
 
