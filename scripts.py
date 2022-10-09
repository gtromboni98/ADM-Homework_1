#GABRIELE TROMBONI, MATRICOLA N.2088799, E-MAIL: tromboni.2088799@studenti.uniroma1.it

#HOMEWORK_1

--------------------------------------------------------

#PROBLEM_1

--------------------------------------------------------

#INTRODUCTIONS

#-Say "Hello, World!" With Python

if __name__ == '__main__':
    print("Hello, World!")

#-Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

#-Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

#-Loops

if __name__ == '__main__':
    n = int(input())
    i = 0
    while i < n:
        print(i*i)
        i += 1

#-Write a function

def is_leap(year):
    leap = False
    if (year % 4 == 0 and year % 100 != 0):
        leap = True 
    elif (year % 100 == 0 and year % 400 == 0):
        leap = True
    return leap

#-Print Function

if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i, end="")

#-Python If-Else

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n%2 != 0:
        print('Weird')
    elif (n%2 == 0 and n in range(2,6)):
        print('Not Weird')
    elif (n%2 == 0 and n in range(6,21)):
        print('Weird')
    elif n > 20:
        print('Not Weird')

--------------------------------------------------------

#BASIC DATA TYPES

#-List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    lista=[[a,b,c] for a in range(x+1) for b in range(y+1) for c in range(z+1) if (a+b+c) != n]
    print(lista)

#-Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    set_score = set(arr)
    lista = list(set_score)
    lista_score = sorted(lista, reverse=True)
    up_score = lista_score[1]
    print(up_score)

#-Nested Lists

if __name__ == '__main__':
    lista = []
    l_score = []
    l_output = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        lista.append([name,score])
        l_score.append(score)
    l_score = set(l_score)
    l_score1 = list(l_score)
    l_score1.sort(reverse=True)
    x = l_score1[-2]
    for i in range(len(lista)):
        if lista[i][1] == x:
            l_output.append((lista[i][0]))
    l_output.sort()
    for i in l_output:
        print(i)
    
#-Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    if query_name in student_marks:
        out = student_marks[query_name]
    avg = sum(out)/len(out)
    print("{:.2f}".format(avg))

#-Lists

if __name__ == '__main__':
    N = int(input())
    l = []
    for i in range(N):
        cmd, *num = input().split() 
        if cmd == 'insert':
            l.insert(int(num[0]),int(num[1]))
        elif cmd == 'print':                       
            print(l)
        elif cmd == 'remove':
            l.remove(int(num[0]))
        elif cmd == 'append':
            l.append(int(num[0]))
        elif cmd == 'sort':
            l.sort()
        elif cmd == 'pop':
            l.pop()
        elif cmd == 'reverse':
            l.reverse()

#-Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    out = hash(t)
    print(out)

------------------------------------------------------------

#STRINGS

#-sWAP cASE

def swap_case(s):
    return(s.swapcase())

#-String Split and Join

def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return(line)
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#-What's Your Name?

def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")

#-Mutations

def mutate_string(string, position, character):
    mut_str = string[:position] + character + string[(position+1):]
    return(mut_str)

#-Find a string

def count_substring(string, sub_string):
    c = 0
    l = len(string)-len(sub_string)
    for i in range(l+1):
        a=0
        for j in range(len(sub_string)):
            if string[i+j] == sub_string[j]:
                a += 1
        if a == len(sub_string):
            c += 1 
    return c      

#-String Validators

if __name__ == '__main__':
    s = input()
    print(any([x for x in s if x.isalnum()]))
    print(any([x for x in s if x.isalpha()]))
    print(any([x for x in s if x.isdigit()]))
    print(any([x for x in s if x.islower()]))
    print(any([x for x in s if x.isupper()]))

#-Text Alignment

#Replace all ______ with rjust, ljust, center
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#-Text Wrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

#-Designer Door Mat

N, M = list(map(int, input().split()))
m = int(N/2)
for i in range(m):
    print((('.|.')*i).rjust(int(M/2)-1,'-') + '.|.' + ('.|.'*i).ljust(int(M/2)-1,'-'))
print('welcome'.upper().center(M, '-'))
for i in range(m,0,-1):
    print((('.|.')*(i-1)).rjust(int(M/2)-1,'-') + '.|.' + ('.|.'*(i-1)).ljust(int(M/2)-1,'-'))

#-String Formatting

def print_formatted(number):
    width = len(bin(number)[2:])
    for i in range(1,number+1):
        print(f"{str(i).rjust(width, ' ')} {oct(i)[2:].rjust(width, ' ')} {hex(i)[2:].upper().rjust(width,' ')} {bin(i)[2:].rjust(width,' ')}")

#-Capitalize!

def solve(s):
    word = list(s)
    check = True
    for i in range(len(word)):    
        if word[i] == ' ':
            check = True
            continue
        if check is True:
            word[i] = word[i].upper()
        check = False
    return ''.join(word)

------------------------------------------------------------

#SETS

#-Introduction to Sets

def average(array):
    heights = set(array)
    avg = sum(heights)/len(heights)
    return(round(avg,3))

#-Symmetric Difference

m = int(input())
a = set(map(int, input().split()))
n = int(input())
b = set(map(int, input().split()))
sym_diff = list((a.difference(b)).union(b.difference(a)))
sym_diff.sort()
for i in sym_diff:
    print(i, sep = '\n')

#-Set .add()

c_stamps = int(input())
names = set()
for _ in range(c_stamps):
    names.add(input())
print(len(names))

#-Set .discard(), .remove() & .pop()

n = int(input())
el = set(map(int, input().split()))
N = int(input())

for _ in range(N):
    commands = list(input().split())
    if 'pop' in commands:
        el.pop()
    elif 'remove' in commands:
        el.remove(int(commands[1]))
    elif 'discard' in commands:
        el.discard(int(commands[1])) 
print(sum(el))

#-Set .union() Operation

n = int(input())
eng_st = set(map(int, input().split()))
b = int(input())
fr_st = set(map(int, input().split()))
out = eng_st.union(fr_st)
print(len(out))

#-Set .intersection() Operation

n = int(input())
eng_st = set(map(int, input().split()))
b = int(input())
fr_st = set(map(int, input().split()))
out = eng_st.intersection(fr_st)
print(len(out))

#-Set .difference() Operation

n_en = int(input())
eng_st = set(map(int, input().split()))
n_fr = int(input())
fr_st = set(map(int, input().split()))
out = eng_st.difference(fr_st)
print(len(out))

#-Set .symmetric_difference() Operation

n_eng = int(input())
eng_st = set(map(int, input().split()))
n_fr = int(input())
fr_st = set(map(int, input().split()))
out = eng_st.symmetric_difference(fr_st)
print(len(out))

#-Set Mutations

n = len(input())
A = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    commands, a = input().split()
    s_N = set(map(int, input().split()))
    if commands == 'intersection_update':
        A.intersection_update(s_N)
    elif commands == 'update':
        A.update(s_N)
    elif commands == 'symmetric_difference_update':
        A.symmetric_difference_update(s_N)
    elif commands == 'difference_update':
        A.difference_update(s_N)
print(sum(A))

#-The Captain's Room

K = int(input())
n_rooms = list(map(int, input().split()))
rooms = set(n_rooms)
for i in list(rooms):
    n_rooms.remove(i)
s = set(n_rooms)
out = rooms.difference(s)
for i in out:
    print(i)

#-Check Subset

T = int(input())
for _ in range(T):
    n_A = len(input())
    A = set(map(int, input().split()))
    n_B = len(input())
    B = set(map(int, input().split()))
    if A.intersection(B) == A:
        print('True')
    else:
        print('False')

#-Check Strict Superset

A = set(map(int, input().split()))
n = int(input())
sets = set()
for _ in range(n):
    s = set(map(int, input().split()))
    sets.update(s)
print(A.issuperset(sets))

------------------------------------------------------------

#COLLECTIONS

#-collections.Counter()

from collections import Counter
X = int(input())
s_size = list(map(int, input().split()))
N = int(input())
n_shoe = Counter(s_size)
tot = 0
for _ in range(N):
    size, price = map(int, input().split())
    if size in n_shoe.keys() and n_shoe[size] > 0:
        tot = tot + price 
        n_shoe[size] -= 1
print(tot) 

#-Collections.namedtuple()

from collections import namedtuple
N = int(input())
table_s = namedtuple('table_s', input().rsplit())
print(sum([int(table_s(*input().rsplit()).MARKS) for _ in range(N)])/N)

#-Collections.OrderedDict()

from collections import OrderedDict
N = int(input())
d = OrderedDict()
for _ in range(N):
    *item_name, price = list(map(str, input().split()))
    item_name, price = " ".join(item_name), int(price)
    if item_name in d:
        d[item_name] += price
    else:
        d[item_name] = price
for i, j in d.items():
    print(i, j)

#-Collections.deque()

from collections import deque
d = deque()
for _ in range(int(input())):
    cmds = list(input().split())
    if 'append' in cmds:
        d.append(int(cmds[1]))
    elif 'appendleft' in cmds:
        d.appendleft(int(cmds[1]))
    elif 'pop' in cmds:
        d.pop()
    elif 'popleft' in cmds:
        d.popleft() 
for i in d:
    print(i, end = ' ')

#-DefaultDict Tutorial

from collections import defaultdict
n, m = map(int, input().split())
d = defaultdict(list)
for i in range(1, n+1):
    key = input()
    d[key].append(i)
for j in range(1, m+1):
    key = input()
    if key not in d:
        print(-1)
    else:
        print(" ".join([str(item) for item in d[key]]))            

--------------------------------------------------------------

#DATE AND TIME

#-Calendar Module

import calendar as cal

m, d, y = list(map(int, input().split()))
print(cal.day_name[cal.weekday(y,m,d)].upper())

--------------------------------------------------------------

#ERRORS AND EXCEPTIONS

#-Exceptions

for _ in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(a//b)
    except (ValueError, ZeroDivisionError) as e:
        print("Error Code:", e)

--------------------------------------------------------------

#BUILT-INS

#-Zipped!

students, subjects = map(int,input().split())
sub_score = [list(map(float,input().split())) for _ in range(subjects)]
for i in zip(*sub_score):
    avg = round(sum(i)/len(i),1)
    print(avg)

#-Athlete Sort

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
athl_sort = sorted(arr, key=lambda x: x[k])
for i in athl_sort:
    print(*[j for j in i], end=' ')
    print()

#-ginortS

S = input()
uppercase = list()
lowercase = list()
evens = list()
odds = list()
for i in range(len(S)):
    if S[i].isupper():
        uppercase.append(S[i])
    elif S[i].islower():
        lowercase.append(S[i])
    elif S[i].isdigit() and (int(S[i])%2)==0:
        evens.append(S[i])
    else:
        odds.append(S[i])
uppercase = sorted(uppercase)
lowercase = sorted(lowercase)
evens = sorted(evens)
odds = sorted(odds)
print(''.join(lowercase)+''.join(uppercase)+''.join(odds)+''.join(evens))

--------------------------------------------------------------

#PHYTON FUNCTIONALS

#-Map and Lambda Function

cube = lambda x: x ** 3 
        
def fibonacci(n):
    result = []
    a, b = 0, 1
    for _ in range(n):
        result.append(a)
        a, b = b, a+b
    return result

--------------------------------------------------------------

#REGEX AND PARSING 

#-Detect Floating Point Number

import re
T = int(input())
for _ in range(T):
    N = input()
    try:
        float(N)
        out = re.match('^[+-]{0,1}[\d]{0,}\.\d+$', N) 
        print(bool(out))
    except:
        print("False")

#-Re.split()

regex_pattern = r"[,.]+"

#-Group(), Groups() & Groupdict()

import re
S = str(input())
out = re.search(r"([A-Za-z0-9])\1+", S)
if out:
    print(out.group(1))
else:
    print('-1')

#-Re.start() & Re.end()

import re
S = str(input())
k = str(input())
pattern = re.finditer(r'(?=(' + k + '))', S)
match = False
for i in pattern:
    match = True
    print((i.start(1), i.end(1) - 1))
if match == False:
    print((-1,-1))
        
#-Re.findall() & Re.finditer()

import re
S = input()
pattern = re.findall(r'(?<=[qwrtypsdfghjklzxcvbnm])([aeiou]{2,})(?=[qwrtypsdfghjklzxcvbnm])', S.strip(), re.IGNORECASE) 
if pattern:
    for i in pattern:
        print(i)
else:
    print(-1) 

#-Regex Substitution

import re 
N = int(input())
def sub(items):
    if items.group(1) == '&&':
        return 'and'
    else:
        return 'or'
for _ in range(N):
    print(re.sub(r"(?<=\s)(\|\||&&)(?=\s)", sub, input()))

#-Validating Roman Numerals

regex_pattern = r"(M{0,3})(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$"

#-Validating phone numbers

import re
N = int(input())
for i in range(N):
    strings = input()
    if (len(strings)==10 and strings.isdigit()):
        valid_mn = re.findall(r"^[789]", strings)
        if len(valid_mn)==1:
            print("YES")
        else:
            print("NO")
    else:
        print("NO")

#-Validating and Parsing Email Addresses

import re
import email.utils
n = int(input())
for i in range(n):
    name, email = input().split()
    if bool(re.match(r"<[a-z][a-zA-Z0-9\-\.\_]+@[a-zA-Z]+\.[a-zA-Z]{1,3}>", email)):
        print(name, email)

#-HTML Parser - Part 1

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for attributes, attribute_value in attrs:
            print(f"-> {attributes} > {attribute_value}")
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for attributes, attribute_value in attrs:
            print(f"-> {attributes} > {attribute_value}")
    def handle_endtag(self, tag):
        print(f"End   : {tag}")

N = int(input())
l = ""
for _ in range(N):
    s = input()
    l += s
parser = MyHTMLParser()
parser.feed(l)

#-HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data != '\n':
            if "\n" in data:
                print(">>> Multi-line Comment")
                print(data)
            else:
                print(">>> Single-line Comment")
                print(data)
    def handle_data(self, data):
        if data != '\n':
            print(f">>> Data")
            print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

--------------------------------------------------------

#XML

#-XML 1 - Find the Score

def get_attr_number(node):
    i = 0
    for item in node:
        i = i + get_attr_number(item)
    return i + len(node.attrib)

--------------------------------------------------------

#NUMPY

#-Arrays

def arrays(arr):
   return (numpy.array(arr[::-1],float))

#-Shape and Reshape

import numpy 
arr = numpy.array(input().split(), int)
out = arr.reshape(3,3)
print(out)

#-Transpose and Flatten

import numpy as np
n, m = list(map(int,input().split()))
l = []
for i in range(n):
    l.append(input().split())
arr = np.array(l,int)
print(np.transpose(arr))
print(arr.flatten())

#-Concatenate

import numpy as np

n,m,p = map(int, input().split())   
arr_1 = np.array([list(map(int, input().split())) for i in range(n)])
arr_2 = np.array([list(map(int, input().split())) for i in range(m)])
print(np.concatenate((arr_1, arr_2)))

#-Zeros and Ones

import numpy as np
sizes = list(map(int,input().split()))
print(np.zeros(sizes, int))
print(np.ones(sizes, int))

#-Eye and Identity

import numpy as np 
np.set_printoptions(legacy='1.13')
N,M = list(map(int,input().split()))
print(np.eye(N,M))

#-Array Mathematics

import numpy as np 
N,M = map(int, input().split())
arr_1 = [np.array(input().split(),int) for _ in range(N)]
arr_2 = [np.array(input().split(),int) for _ in range(N)]
fs = [np.add, np.subtract, np.multiply, np.floor_divide, np.mod, np.power]
[print(f(arr_1,arr_2)) for f in fs]

#-Floor, Ceil and Rint

import numpy as np 
np.set_printoptions(legacy='1.13')
arr = np.array(input().split(),float)
[print(f(arr)) for f in [np.floor,np.ceil,np.rint]]

#-Sum and Prod

import numpy as np
n,m = map(int,input().split())
arr = np.array([list(map(int,input().split())) for _ in range(n)])
print(np.product(np.sum(arr,0)))

#-Min and Max

import numpy as np
n,m = map(int, input().split())
arr = np.array([list(map(int, input().split())) for _ in range(n)])
print(np.max(np.min(arr,1)))

#-Mean, Var, and Std

import numpy as np
n,m = map(int, input().split())
arr = np.array([list(map(int, input().split())) for _ in range(n)])
print(np.mean(arr,1))
print(np.var(arr,0))
print(np.around(np.std(arr),11))

#-Dot and Cross

import numpy as np 
n = int(input())
a = np.array([input().split() for _ in range(n)], int)
b = np.array([input().split() for _ in range(n)], int)
print(np.dot(a,b))

#-Inner and Outer

import numpy as np
a, b = list([map(int,input().split()) for _ in range (2)])
a, b = list(a),list(b)
print(np.inner(a,b))
print(np.outer(a,b))

#-Polynomials

import numpy as np
values = list(map(float, input().split()))
x = float(input())
print(np.polyval(values, x))

#-Linear Algebra

import numpy as np
n = int(input())
arr = [tuple(map(float,input().split())) for _ in range(n)]
det = np.linalg.det(arr)
print(det.round(2))

----------------------------------------------------------------

#PROBLEM_2

----------------------------------------------------------------

#-Birthday Cake Candles

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    return(candles.count(max(candles)))
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

------------------------------------------------------

#-Number Line Jumps

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    diff_v = v1 - v2
    diff_x = x2 - x1
    if v2 >= v1:
        return ("NO")
    elif ((x2-x1)%(diff_v)) != 0:
        return ("NO")
    else:
        return ("YES")
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

----------------------------------------------------------

#-Viral Advertising

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    cum = [2]
    for i in range(1,n):
        cum.append(cum[i-1] + (cum[i-1]//2))
    return(sum(cum))
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

----------------------------------------------------------

#-Insertion Sort - Part 1

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    for i in range(n-1,0,-1):
        if arr[i] < arr[i-1]:
            temp = arr[i]
            arr[i] = arr[i-1]
            print(*arr)
            arr[i-1]= temp
    print(*arr)

    # Write your code here

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

------------------------------------------------------------

#-Insertion Sort - Part 2

#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    for i in range(1,n):
        temp = arr[i]
        j = i
        while (j > 0 and arr[j-1] > temp):
            arr[j] = arr[j-1]
            j = j-1
        arr[j] = temp
        print(*arr)
            

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

----------------------THIS IS MY HOMEWORK--------------------------
