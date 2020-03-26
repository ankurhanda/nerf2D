import glob
import os

dataset = 'house'

raw_xy  = sorted(glob.glob("./dataset/" + dataset + "/*raw_xy*.jpg"))
sin_cos = sorted(glob.glob("./dataset/" + dataset + "/*sin_cos*.jpg"))

count = 0
for x, y in zip(raw_xy, sin_cos):
    print(count, x, y)
    outfileName = 'evolution_{:04d}.jpg'.format(int(count))
    os.system("montage ./dataset/{}.jpg {} {} -geometry +0+0 {}".format(dataset, y, x, outfileName))

    count += 1 