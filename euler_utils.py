import itertools
import operator
import collections
import ipdb

import numpy as np


def triangle_numbers():
	n = 1
	while(True):
		yield n*(n+1)/2
		n+=1
def fibonnaci():
	curr = 1
	prev = 0
	yield prev
	yield curr
	while True:
		temp = curr
		curr = curr + prev
		prev = temp
		yield curr

def is_palindrome(number):
	ipdb.set_trace()
	number = str(number)
	for i in range(len(number)):
		if number[i] != number[-i-1]:
			return False

	return True


def prime_factorize(number, complete_prime_list = None):
	if number == 0:
		return []
	factors = []

	# this reduces the search space
	if complete_prime_list is not None:
		if number in complete_prime_list:
			return [number]

		for prime in complete_prime_list:
			if prime > number:
				break

			while number%prime == 0:
				factors.append(prime)
				number //= prime

		return factors

	while number %2 == 0:
		factors.append(2)
		number = number//2

	divisor = 3
	while divisor <= number:
		while number %divisor == 0:
			factors.append(divisor)
			number //= divisor
		divisor += 2

	return factors

# admittedly this helped # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
# though I did re implement it on my own and now mine is faster (not sure why)
def get_primes_below(number):
	if number <= 1:
		return []
	is_prime = np.ones(number/2, dtype = bool)
	for n in xrange(3, int(np.sqrt(number))+1, 2):
		if is_prime[n/2]:
			is_prime[(n**2)/2:number:n] = False

	primes = (np.where(is_prime)[0]*2)+1
	primes[0] = 2
	return primes

def is_prime(number):
	for i in xrange(2, int(np.sqrt(number)) + 1):
		if number % i == 0:
			return False

	return True

def get_divizors(n, proper = False):
	divizors = set()
	for x in xrange(1, int(n**.5 + 1)):
		if n%x == 0:
			divizors.add(x)
			divizors.add(n//x)
	if proper:
		divizors.remove(n)
	return divizors

def get_divizors_given_primes(n, complete_prime_list = None, proper = False):
	if n == 0:
		return []
	if complete_prime_list is not None:
		primes = prime_factorize(n, complete_prime_list)
	else:
		primes = prime_factorize(n)
	divizors = set([1, n])
	for i in xrange(1, len(primes)):
		for combo in itertools.combinations(primes, i):
			divizors.add(reduce(operator.mul, combo))

	if proper:
		divizors.remove(n)
	return divizors

# stolen from tyler
def get_num_divizors(n, complete_prime_list = None):
	if n == 0:
		return []
	if complete_prime_list is not None:
		primes = prime_factorize(n, complete_prime_list)
	else:
		primes = prime_factorize(n)
	
	factor_count = 1
	for prime in primes:
		power = 1
		while n%prime == 0:
			power += 1
			n = n/prime
		factor_count *= power
	return factor_count

def is_pandigital(numbers, require_all_digits = True):
    """determines if a string of numbers is pandigital"""
    if require_all_digits:
    	must_contain = set('123456789')
    else:	
    	must_contain = set([str(digit) for digit in xrange(1, len(numbers) + 1)])

    result = set(numbers)
    return len(numbers) == len(result) and result == must_contain
    # count = collections.Counter()
    # count.update([x for num in numbers for x in str(num)])
    # return all(x == 1 for x in count.values()) and set(count.keys()) == must_contain

# Dicksons method
# http://en.wikipedia.org/wiki/Formulas_for_generating_Pythagorean_triples
def pythagorean_triplets(return_r = False):
    r = 2
    while True:
        product = r**2/2
        factors = sorted(list(get_divizors(product)))
        for factor in factors[:len(factors)//2]:
            s = factor
            t = t = product/factor
            x = r + s
            y = r + t
            z = r + s + t
            if return_r:
                yield (x, y, z, r)
            else:
                yield (x, y, z)

        r += 2

def pandigitals(length, include_0 = False):
	digits = [str(x) for x in xrange(length, 0, -1)]
	if include_0:
		digits.append('0')
	gen = itertools.permutations(digits, len(digits))
	for elem in gen:
		yield int(reduce(operator.add, elem))

def pentagonals():
    n = 1
    while(True):
        yield n*(3*n-1)/2
        n += 1

def is_pentagonal(p):
    if p < 1:
        return False

    a = 3.
    b = -1.
    c = -p*2.
    n = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    if n == int(n):
        return True
    else:
        return False

def is_hexagonal(h):
    if h < 1:
        return False    
    a = 2.
    b = -1.
    c = -h
    n = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    if n == int(n):
        return True
    else:
        return False

def triangulars():
	n = 1
	while(True):
		yield n*(n+1)/2
		n += 1

def is_triangular(t):
    if t < 1:
        return False

    a = 2.
    b = -1.
    c = -t*2.
    n = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    if n == int(n):
        return True
    else:
        return False
