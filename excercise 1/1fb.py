
for j in range(N):
    for i in range(P):
        for k in range(M):
            y[i, j] += z[i, k] * x[k, j]
