import os
import numpy as np
def path(a):
    # sum = a[0
    wid = len(a)
    length = wid
    b = np.transpose(a)
    c = np.zeros_like(a)
    c[0][0] = a[0][0]
    for i in range(wid):
        for j in range(length):
            if i==0 and j!=0:
                c[i][j] = a[i][j]+c[i][j-1]
            elif j==0 and i!=0:
                c[i][j] = a[i][j]+c[i-1][j]
            else:
                c[i][j] = min(c[i-1][j]+a[i][j], c[i][j-1]+a[i][j])
    print(c)
    return c[-1][-1]

grid = [[1,3,1],[1,5,1],[4,2,1]]

print(path(grid))