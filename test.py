
def digitsum(n):
    sum = 0
    while n>0:
        sum += n%10
        n = int(n/10)
    return sum

a=24*426+1
min = 10
minn = a
while a<=99999:
    if digitsum(a)==7:
        print a
    a+=852

print (21301*21301*21301-1)%2556

