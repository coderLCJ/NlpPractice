def Parse():
    path = "oxforddata.txt" #文件路径

    #将文件内容解析成一个多级字典，并返回
    #   请在此添加实现代码   #
	# ********** Begin *********#
    file = open(path, 'r')
    for i in range(7):
        file.readline()
    dat = {}
    name = ['tmax', 'tmin', 'af', 'rain', 'sun']
    for line in file.readlines():
        line = line.split()
        if int(line[0]) not in dat:
            dat[int(line[0])] = {}
        if int(line[1]) not in dat[int(line[0])]:
            dat[int(line[0])][int(line[1])] = {}
        for j in range(2, 7):
            if line[j] == '---':
                dat[int(line[0])][int(line[1])][name[j - 2]] = None
            else:
                line[j] = line[j].replace('*', '')
                dat[int(line[0])][int(line[1])][name[j - 2]] = float(line[j])

    return dat
	# ********** End **********#
Parse()