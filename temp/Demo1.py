# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         Demo1
# Description:  
# Author:       Laity
# Date:         2021/9/27
# ---------------------------------------------
# CH6.String
def words(sentence):
    ret = sentence.split()
    for index, word in enumerate(ret):
        new_word = ''
        for letter in word:
            if letter not in [',', '.', '!']:
                new_word += letter
        ret[index] = new_word
    return ret


assert words('Hello, world! Hello, Python') == ['Hello', 'world', 'Hello', 'Python']


# ---------------------------------------------
def standardTime(time24):
    hour, mini = time24.split(':')
    hour = int(hour)
    mini = int(mini)
    if int(hour) > 12:
        m = 'PM'
        hour -= 12
    else:
        if hour == 0:
            hour = 12
        m = 'AM'
    return '%d:%02d %s' % (hour, mini, m)


assert standardTime('14:45') == '2:45 PM'


# ---------------------------------------------
def militaryTime(time12):
    time, m = time12.split()
    hour, mini = time.split(':')
    hour = int(hour)
    mini = int(mini)
    if m == 'PM':
        hour += 12
    elif m == 'AM' and hour == 12:
        hour = 0
    return '%02d:%02d' % (hour, mini)


assert militaryTime('9:05 AM') == '09:05'


# ---------------------------------------------
def checkPasswd(passwd):
    if len(passwd) < 8:
        return False
    r = [False, False, False, True]
    for s in passwd:
        if s.isdigit():
            r[0] = True
        if s.islower():
            r[1] = True
        if s.isupper():
            r[2] = True
        if not s.isdigit() and not s.isalpha():
            r[3] = False
    return r[0] and r[1] and r[2] and r[3]


assert checkPasswd('aA12331222') is True
# ---------------------------------------------
def isPalindrome(word):
    return word.lower() == word.lower()[::-1]


assert isPalindrome('Anna') is True
# ---------------------------------------------
# CH7.List
def hello(name_list):
    return ['Hello, ' + ret + '!' for ret in name_list]


assert hello(['Mary', 'Tracy']) == ['Hello, Mary!', 'Hello, Tracy!']
assert hello([]) == []
# ---------------------------------------------
def absList(num_list):
    return [abs(num) for num in num_list]


assert absList([-1, 2, -4]) == [1, 2, 4]
# ---------------------------------------------
def onlyPositive(num_list):
    return [num for num in num_list if num > 0]


num_list = [-2, 3, 5, -4]
assert onlyPositive(num_list) == [3, 5]
# ---------------------------------------------
def addMatrix(A, B):
    res = A
    for i in range(len(A)):
        for j in range(len(A[0])):
            res[i][j] = A[i][j] + B[i][j]
    return res


A = [[1, 2], [3, 4]]
B = [[1, 1], [2, 2]]
assert addMatrix(A, B) == [[2, 3], [5, 6]]
# ---------------------------------------------
def multiplyMatrix(A,B):
    r_A, l_A = len(A), len(A[0])
    l_B = len(B[0])
    ret = []
    for i in range(r_A):
        num = []
        for k in range(l_B):
            s = 0
            for j in range(l_A):
                s += A[i][j] * B[j][k]
            num.append(s)
        ret.append(num)
    return ret


A = [[1, 2], [3, 4]]
B = [[2, 1], [1, 2]]
assert multiplyMatrix(A, B) == [[4, 5], [10, 11]]
# ---------------------------------------------
def uniqueList(char_list):
    ret_list = []
    for ch in char_list:
        if ch not in ret_list:
            ret_list.append(ch)
    return ret_list


char_list = ['m', 'o', 'r', 'n', 'i', 'n', 'g', 'c', 'a', 'l', 'l']
assert uniqueList(char_list) == ['m', 'o', 'r', 'n', 'i', 'g', 'c', 'a', 'l']
# ---------------------------------------------
# CH8 Set
def equalSets(A, B):
    return A == B


assert equalSets({1, 2,}, {2, 1}) is True
# ---------------------------------------------
def uniqueWords(sentence):
    return {word for word in sentence.split()}


assert uniqueWords('good good study day day up') == {'good', 'study', 'day', 'up'}
# ---------------------------------------------
def dictionary(strings):
    return {s for s_list in strings for s in s_list.split()}


assert dictionary(['a deal is a deal', 'deal with it']) == {'is', 'a', 'it', 'with', 'deal'}
# ---------------------------------------------
def heavyOverlap(A, B):
    half = len(A)/2
    c = 0
    for a in A:
        if a in B:
            c += 1
    return c >= half


assert heavyOverlap({1, 2, 3}, {2, 3, 4, 5, 6, 7, 8})
# ---------------------------------------------
def objectAttribute(obj):
    return {id(obj), type(obj), obj}


assert objectAttribute(5) == {id(5), type(5), 5}
# ---------------------------------------------
