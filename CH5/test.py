import matplotlib as mpl

# mpl.use('Agg')
import matplotlib.pyplot as plt
#  不要改变上面的顺序

import datetime


def Draw():
    appl = "step3/AAPL.csv"  # 苹果
    google = "step3/GOOG.csv"  # 谷歌
    ms = "step3/MSFT.csv"  # 微软
    f = open(appl, 'r')
    line = f.readline()
    x = []
    y = []
    for line in f.readlines():
        i1 = line.index(",", 0, len(line))
        dt = datetime.datetime.strptime(line[0:i1], "%Y-%m-%d").date()
        x.append(dt)
        line = line.split(',')
        y.append(float(line[1]))
    plt.plot(x, y, color='red', linewidth=1.0, label="apple")

    f = open(google, 'r')
    line = f.readline()
    x = []
    y = []
    for line in f.readlines():
        i1 = line.index(",", 0, len(line))
        dt = datetime.datetime.strptime(line[0:i1], "%Y-%m-%d").date()
        x.append(dt)
        line = line.split(',')
        y.append(float(line[1]))
    plt.plot(x, y, color='green', linewidth=1.0, label="google")

    f = open(ms, 'r')
    f.readline()
    x = []
    y = []
    for line in f.readlines():
        i1 = line.index(",", 0, len(line))
        dt = datetime.datetime.strptime(line[0:i1], "%Y-%m-%d").date()
        x.append(dt)
        line = line.split(',')
        y.append(float(line[1]))
    plt.plot(x, y, color='blue', linewidth=1.0, label="ms")


    plt.xticks(rotation=45)
    plt.legend(["Apple", "Google", "Microsoft"])
    plt.ylabel("Open")
    plt.savefig("step3/output/data.png")


x = []
y = []
readlines = [
    '2014-01-01,37.349998,37.889999,34.630001,37.840000,33.554665,930226200',

    '2015-01-01,37.349998,37.889999,34.630001,37.840000,33.554665,930226200',
    '2016-01-01,37.349998,37.889999,34.630001,37.840000,33.554665,930226200',
    '2017-01-01,37.349998,37.889999,34.630001,37.840000,33.554665,930226200',

    '2019-03-01,37.349998,37.889999,34.630001,37.840000,33.554665,930226200',
    '2020-01-01,37.349998,37.889999,34.630001,37.840000,33.554665,930226200',
    '2018-01-01,37.349998,37.889999,34.630001,37.840000,33.554665,930226200',
]

for line in readlines:
    i1 = line.index(",", 0, len(line))
    dt = datetime.datetime.strptime(line[0:i1], "%Y-%m-%d").date()
    x.append(dt)
    line = line.split(',')
    y.append(float(line[1]))
plt.plot(x, y, color='red', linewidth=1.0, label="apple")
plt.show()