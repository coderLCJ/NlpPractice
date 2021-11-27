# task 1
import re
def validIPAdd(s):
    # YOUR CODE HERE
    comp = re.compile(r'^((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}')
    return False if comp.match(s) is None else True


assert validIPAdd('125.122.122') == False
assert validIPAdd('125.122.122.12') == True
assert validIPAdd('256.122.122.12') == False
assert validIPAdd('256.122.122.12') == False

# task 2
import re
def strongPassWord(s):
    # YOUR CODE HERE
    comp = re.compile(r'(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])[A-Za-z0-9]{8,10}')
    return False if comp.match(s) is None else True


assert strongPassWord('abc') == False
assert strongPassWord('ab123ABCD') == True
assert strongPassWord('ABC124~~~') == False
assert strongPassWord('ABCABC123') == False

# task 3
import re
def validDate(s):
    # YOUR CODE HERE
    comp = re.compile(r'([0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]{1}|[0-9]{1}[1-9][0-9]{2}|[1-9][0-9]{3})-(((0[13578]|1[02])-(0[1-9]|[12][0-9]|3[01]))|((0[469]|11)-(0[1-9]|[12][0-9]|30))|(02-(0[1-9]|[1][0-9]|2[0-8])))')
    return False if comp.match(s) is None else True


assert validDate("2017-02-11") == True
assert validDate("2017-02") == False
assert validDate("2017-02-29") == False
assert validDate("2017-15-11") == False

