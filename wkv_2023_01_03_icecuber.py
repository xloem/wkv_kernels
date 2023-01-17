#This is from <@468093332535640064> and it agrees with the CUDA wkv on dummy
#data, however they are sometimes different in real model. It will be nice if
#someone can check for bugs.
#you can change `s //= 4` to `s //= 2` to fix the bug. This fix adds a sort of
#nop sometimes (first iteration does nothing), but it works, and I think it
#should be fine.
def RUN_CUDA(B, T, C, w, u, k, v):

    def associative_scan(f, A):
        A = tuple(I.detach().clone() for I in A)

        def get(start, end, step):
            return tuple(I[start:end:step] for I in A)

        def assign(a, b):
            assert(len(a) == len(b))
            for i in range(len(a)):
                a[i][:] = b[i]

        n = len(A[0])
        s = 2
        while s <= n:
            m = n - n % s
            a = get(s//2-1, m, s)
            b = get(s - 1, m, s)
            assign(b, f(a, b))
            s *= 2

        s //= 4
        while s >= 2:
            m = n - (n % s < s//2)*(s//2)
            a = get(s - 1, m, s)
            b = get(s//2*3-1, m, s)
            assign(b, f(a, b))
            s //= 2
        return A

    w = -torch.exp(w)
    k = k.swapaxes(1, 0)
    v = v.swapaxes(1, 0)

    def f(A, B):
        Ap, Aq, Ao, Aa = A
        Bp, Bq, Bo, Ba = B

        Ae = Ao + Ba * w
        Co = torch.maximum(Ae, Bo)
        x = torch.exp(Ae - Co)
        y = torch.exp(Bo - Co)

        Cp = x * Ap + y * Bp
        Cq = x * Aq + y * Bq
        Ca = Aa + Ba

        return Cp, Cq, Co, Ca

    a0 = torch.ones((T, 1, 1), dtype=w.dtype, device=w.device)
    q0 = torch.ones((T, B, C), dtype=w.dtype, device=w.device)
    p, q, o, _ = associative_scan(f, (v, q0, k, a0))

    p = torch.roll(p, 1, dims=0)
    q = torch.roll(q, 1, dims=0)
    o = torch.roll(o, 1, dims=0)

    no = torch.maximum(o, k + u)
    A = torch.exp(o - no)
    B = torch.exp(k + u - no)
    y = (A * p + B * v) / (A * q + B)
    y = torch.cat((v[:1, :, :], y[1:, :, :]))
    return y.swapaxes(1, 0)
