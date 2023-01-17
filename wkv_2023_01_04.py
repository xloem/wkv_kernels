# I tried to apply JIT on this but that only makes it even slower 😂 
class WKV_Kernel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, B, T, C, w, u, k, v):
        w = w.float()
        u = u.float()
        k = k.float()
        v = v.float()

        w = -torch.exp(w)
        k = k.swapaxes(1, 0)
        v = v.swapaxes(1, 0)
        sl = []
        s = 2
        while s <= T:
            sl += [(s, (s >> 1) - 1, s - 1, T - T % s)]
            s = s << 1
        s = s >> 1
        while s >= 2:
            sl += [(s, s - 1, (s >> 1) * 3 - 1, T - (T % s < (s >> 1)) * (s >> 1))]
            s = s >> 1

        oo = k.detach().clone()
        pp = v.detach().clone()
        qq = torch.ones((T, B, C), dtype=w.dtype, device=w.device)
        dd = torch.ones((T, 1, 1), dtype=w.dtype, device=w.device)
        for ss, sa, sb, sz in sl:
            p = pp[sb:sz:ss]
            q = qq[sb:sz:ss]
            d = dd[sb:sz:ss]
            o = oo[sb:sz:ss]
            e = oo[sa:sz:ss] + d * w
            x = torch.maximum(e, o)
            a = torch.exp(e - x)
            b = torch.exp(o - x)
            p[:] = a * pp[sa:sz:ss] + b * p
            q[:] = a * qq[sa:sz:ss] + b * q
            d[:] = dd[sa:sz:ss] + d
            o[:] = x

        p = torch.roll(pp, 1, dims=0)
        q = torch.roll(qq, 1, dims=0)
        o = torch.roll(oo, 1, dims=0)

        x = torch.maximum(o, k + u)
        a = torch.exp(o - x)
        b = torch.exp(k + u - x)
        y = (a * p + b * v) / (a * q + b)
        y = torch.cat((v[:1, :, :], y[1:, :, :]))
        y = y.swapaxes(1, 0)
        return y.type(w.dtype)
