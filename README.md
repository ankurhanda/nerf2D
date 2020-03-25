## nerf2D 

nerf2D is a 2D toy illustration of the [NeRF](http://www.matthewtancik.com/nerf). The code shows how adding the gamma encoding (Eq. 4) in the paper improves results significantly. 

The task is to reconstruct an image from its 2D coordinates. The dataset consists of tuple ((x, y), (r, g, b)) where the input is (x, y) and output is (r, g, b). We train a 2 layer MLP with relu activations. The input is normalised (as also mentioned in the paper) to range [-1, 1] and we also output in range [-1, 1]. However, simply training with with raw (x, y) results in blurry reconstructions while adding gamma encoding shows dramatic improvements in the results. The gamma encoding is able to preserve the sharp edges in the image.

![\gamma(p) = \[\sin(\pi x), \cos(\pi x), \sin(\pi y), \cos(\pi y), \sin(2\pi x), \cos(2\pi x), \sin(2\pi y), \cos(2\pi y), \cdots \cdots, \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x), \sin(2^{L-1}\pi y), \cos(2^{L-1}\pi y)\]](https://render.githubusercontent.com/render/math?math=%5Cgamma(p)%20%3D%20%5B%5Csin(%5Cpi%20x)%2C%20%5Ccos(%5Cpi%20x)%2C%20%5Csin(%5Cpi%20y)%2C%20%5Ccos(%5Cpi%20y)%2C%20%5Csin(2%5Cpi%20x)%2C%20%5Ccos(2%5Cpi%20x)%2C%20%5Csin(2%5Cpi%20y)%2C%20%5Ccos(2%5Cpi%20y)%2C%20%5Ccdots%20%5Ccdots%2C%20%5Csin(2%5E%7BL-1%7D%5Cpi%20x)%2C%20%5Ccos(2%5E%7BL-1%7D%5Cpi%20x)%2C%20%5Csin(2%5E%7BL-1%7D%5Cpi%20y)%2C%20%5Ccos(2%5E%7BL-1%7D%5Cpi%20y)%5D)

![equation](https://latex.codecogs.com/png.download?%5Cgamma%28p%29%20%3D%20%5B%5Csin%28%5Cpi%20x%29%2C%20%5Ccos%28%5Cpi%20x%29%2C%20%5Csin%28%5Cpi%20y%29%2C%20%5Ccos%28%5Cpi%20y%29%2C%20%5Csin%282%5Cpi%20x%29%2C%20%5Ccos%282%5Cpi%20x%29%2C%5Csin%282%5Cpi%20y%29%2C%20%5Ccos%282%5Cpi%20y%29%2C.....%2C%20%5Csin%282%5E%7BL-1%7D%5Cpi%20x%29%2C%20%5Ccos%282%5E%7BL-1%7D%5Cpi%20x%29%2C%20%5Csin%282%5E%7BL-1%7D%5Cpi%20y%29%2C%20%5Ccos%282%5E%7BL-1%7D%5Cpi%20y%29%20%5D)

The sin plots for various values of L look like the following 

![Sin-Plots](sin.png)

The corresponding cos plots are

![Cos-Plots](cos.png)

