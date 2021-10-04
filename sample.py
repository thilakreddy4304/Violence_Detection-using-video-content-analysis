

def solve(number):

    rev = 0
    while(number):
        rem = number % 10
        rev = rev * 10 + rem
        number //= 10

    return rev == number


number = int(input())
print(solve(number))
