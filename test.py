# task1
# You should return your result.
import pandas as pd


def aboveAverage(s):
    # YOUR CODE HERE
    ave = 0
    for i in s:
        ave += i
    ave /= len(s)
    return pd.Series([i for i in s if i > ave])


s = pd.Series([1, 5, 3, 6, 8])
t = pd.Series([5, 6, 8])

assert (t.equals(aboveAverage(s)))
s = pd.Series([11, 12, 13, 14])
t = pd.Series([13, 14])
assert (t.equals(aboveAverage(s)))

# task2
# You should return your result.
import pandas as pd

def twoSum(s, target):
    # YOUR CODE HERE
    dic = {}
    for i, j in enumerate(s):
        dic[j] = i
    for i in range(len(s)):
        if target - s[i] in dic and target - s[i] != s[i]:
            return i, dic[target - s[i]]


s = pd.Series([1, 5, 3, 6, 8])
t = twoSum(s, 7)
assert (s[t[0]] + s[t[1]] == 7)
s = pd.Series([3, 2, 4])
t = twoSum(s, 6)
assert (s[t[0]] + s[t[1]] == 6)

# task3
# You should return your result.
import pandas as pd
import numpy as np

class Solution(object):
    def combinationSum(self, candidates, target):
        dic = {}
        for index, val in enumerate(candidates):
            dic[val] = index
        ret = []
        for v in candidates:
            temp = [v]
            s = v
            while s < target:
                if target - s in dic:
                    temp.append(target-s)
                    break
                s += v
                temp.append(v)
            if sum(temp) == target:
                ret.append(temp)
        return ret


s = pd.Series([2,3,6,7])
sol = Solution()
t = sol.combinationSum(s,7)
print(t)
assert(len(t) == 2)
for i in range(len(t)):
    assert(np.sum(t[i]) == 7)
s = pd.Series([2,3,5])
t = sol.combinationSum(s,8)
assert(len(t) == 3)
for i in range(len(t)):
    assert(np.sum(t[i]) == 8)


# task4
# You should return your result.
import pandas as pd
def spiralDataFrame(df):
    arr = df.values
    row = len(arr) - 1
    col = len(arr[0]) - 1
    size = len(arr) * len(arr[0])
    ret = []
    i = 0
    j = 0
    row_0 = 0
    col_0 = 0
    while len(ret) < size:
        while j <= col:
            ret.append(arr[i][j])
            j += 1
        j -= 1
        i += 1
        col -= 1
        row_0 += 1
        while i <= row and len(ret) < size:
            ret.append(arr[i][j])
            i += 1
        i -= 1
        j -= 1
        row -= 1
        while j >= col_0 and len(ret) < size:
            ret.append(arr[i][j])
            j -= 1
        j += 1
        col_0 += 1
        i -= 1
        while i >= row_0 and len(ret) < size:
            ret.append(arr[i][j])
            i -= 1
        i += 1
        j += 1
    return ret


df1 = pd.DataFrame({'c1': [1, 4, 7],
                    'c2': [2, 5, 8],
                    'c3': [3, 6, 9]})
t = spiralDataFrame(df1)
assert (t == [1,2,3,6,9,8,7,4,5])
df1 = pd.DataFrame({'c1': [1, 5, 9],
                    'c2': [2, 6, 10],
                    'c3': [3, 7, 11],
                    'c4': [4, 8, 12]})
t = spiralDataFrame(df1)
assert (t == [1,2,3,4,8,12,11,10,9,5,6,7])

# task5
# You should return your result.
import pandas as pd
def tranposeDataFrame(df):
    arr = df.values
    ret = []
    for i in range(len(arr[0])):
        temp = []
        for j in range(len(arr)):
            temp.append(arr[j][i])
        ret.append(temp)
    return ret


df1 = pd.DataFrame({'c1': [1, 4, 7],
                    'c2': [2, 5, 8],
                    'c3': [3, 6, 9]})
t = tranposeDataFrame(df1)
assert (t == [[1, 4, 7], [2, 5 ,8], [3, 6 ,9]])
df1 = pd.DataFrame({'c1': [1, 2, 3],
                    'c2': [4, 5, 6]})
t = tranposeDataFrame(df1)
assert (t == [[1, 2, 3], [4, 5, 6]])

# task 6
# You should return your result.
import pandas as pd
def setDataFrameZeros(df):
    arr = df.values
    n = len(arr)
    m = len(arr[0])
    loc = []
    for i in range(n):
        for j in range(m):
            if arr[i][j] == 0:
                loc.append((i, j))

    for i, j in loc:
        for k in range(m):
            arr[i][k] = 0
        for k in range(n):
            arr[k][j] = 0
    return pd.DataFrame(np.array(arr), columns=['c'+str(i) for i in range(1, m+1)])


df1 = pd.DataFrame({'c1': [1, 4, 7],
                    'c2': [2, 0, 8],
                    'c3': [3, 6, 9]})

df2 = pd.DataFrame({'c1': [1, 0, 7],
                    'c2': [0, 0, 0],
                    'c3': [3, 0, 9]})

assert (df2.equals(setDataFrameZeros(df1)))

df1 = pd.DataFrame({'c1': [0, 3, 1],
                    'c2': [1, 4, 3],
                    'c3': [2, 5, 1],
                    'c4': [0, 2, 5]})

df2 = pd.DataFrame({'c1': [0, 0, 0],
                    'c2': [0, 4, 3],
                    'c3': [0, 5, 1],
                    'c4': [0, 0, 0]})

assert (df2.equals(setDataFrameZeros(df1)))