# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         convert
# Description:  
# Author:       Laity
# Date:         2021/9/20
# ---------------------------------------------
file = open('FB15k/test.txt', 'r')
out_file = open('D:/neo4jTest/import/2.csv', 'w')
i = 0
for line in file.readlines():
    part = line.split()
    new_line = part[0] + ',' + part[1] + ',' + part[2] + '\n'
    i += 1
    # print(new_line)
    out_file.write(new_line)
    if i == 1000:
        break