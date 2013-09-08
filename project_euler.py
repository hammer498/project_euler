import numpy as np
import matplotlib.pyplot as plt
import math
import operator
import itertools
import csv
import collections
import decimal
import sympy
import decimal



import euler_utils as utils

def problem_1():
	return sum(xrange(3,1000,3)) + sum(xrange(5,1000,5)) - sum(xrange(15,1000,15))

def problem_2():
	result_sum = 0
	for num in utils.fibonnaci():
		if num > 4000000:
			return result_sum
		if num %2 == 0:
			result_sum += num

def problem_3():
	return utils.prime_factorize(600851475143)[-1]

def problem_4():
	vect_palindrome = np.vectorize(utils.is_palindrome)
	domain = np.arange(1,1000)
	all = np.outer(domain, domain)
	usable = vect_palindrome(all)
	return np.max(all[usable])

def problem_5(num = 20):
	"""# 2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
"""

	primes = utils.get_primes_below(num)
	factors = [pow(prime, int(math.log(num)/math.log(prime))) for prime in primes]

	value = reduce(operator.mul, factors, 1)
	return value 


def problem_6():
	return sum(np.arange(1, 101))**2 - sum(np.arange(1, 101)**2)


def problem_7():
	"""By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13. 
What is the 10 001st prime number?"""
	primes = utils.get_primes_below(1000000)
	return primes[10000]

def problem_8():
	"""Find the greatest product of five consecutive digits in the 1000-digit number."""

	num = '7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450'
	window_length = 5

	def get_prod(num, indx1, indx2):
		prod = 1
		for i in range(indx1, indx2):
			prod *= int(num[i])
		return prod

	max_val = 0
	for i in range(4, len(num)):
		val = get_prod(num, i-window_length+1, i+1)
		if val > max_val:
			max_val = val

	return max_val

# TODO, understand the better way to do this (non cheater)
def problem_9():
	"""A Pythagorean triplet is a set of three natural numbers, a < b < c, for which,
a^2 + b^2 = c^2
For example, 3^2 + 4^2 = 9 + 16 = 25 = 5^2.
There exists exactly one Pythagorean triplet for which a + b + c = 1000.
Find the product abc."""
	
	def sol_1(real):
		b = -.5*real - .5*np.sqrt(real**2 + 2000*real - 1000000) + 500
 		a = 1000.*(real + np.sqrt(real**2 + 2000*real - 1000000))/(real + np.sqrt(real**2 + 2000*real - 1000000) + 1000)
 		c = real
 		return [a, b, c]

 	def sol_2(real):
		b = -.5*real + .5*np.sqrt(real**2 + 2000*real - 1000000) + 500
 		a = 1000.*(real - np.sqrt(real**2 + 2000*real - 1000000))/(real - np.sqrt(real**2 + 2000*real - 1000000) + 1000)
 		c = real
 		return [a, b, c]

 	def condition(numbers):
 		worked = True
 		for n in numbers:
 			if n%1 != 0 or n < 1:
 				worked = False
 		return worked

 	answers = []
 	for real in xrange(1000):
 		option = sol_1(real)
 		if condition(option):
 			print option
 			return reduce(operator.mul, option)


 	



def problem_10():
	"""The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
Find the sum of all the primes below two million."""
	return sum(utils.get_primes_below(2000000))

def problem_11():
	"""find largest product of 4 numbers in given grid going up, down, lr, diag"""

	grid = np.array([[8, 2, 22, 97, 38, 15, 0, 40, 0, 75, 4, 5, 7, 78, 52, 12, 50, 77, 91, 8],
					 [49, 49, 99, 40, 17, 81, 18, 57, 60, 87, 17, 40, 98, 43, 69, 48, 4, 56, 62, 0],
					 [81, 49, 31, 73, 55, 79, 14, 29, 93, 71, 40, 67, 53, 88, 30, 3, 49, 13, 36, 65],
					 [52, 70, 95, 23, 4, 60, 11, 42, 69, 24, 68, 56, 1, 32, 56, 71, 37, 2, 36, 91],
					 [22, 31, 16, 71, 51, 67, 63, 89, 41, 92, 36, 54, 22, 40, 40, 28, 66, 33, 13, 80],
					 [24, 47, 32, 60, 99, 03, 45, 2, 44, 75, 33, 53, 78, 36, 84, 20, 35, 17, 12, 50],
					 [32, 98, 81, 28, 64, 23, 67, 10, 26, 38, 40, 67, 59, 54, 70, 66, 18, 38, 64, 70],
					 [67, 26, 20, 68, 2, 62, 12, 20, 95, 63, 94, 39, 63, 8, 40, 91, 66, 49, 94, 21],
					 [24, 55, 58, 5, 66, 73, 99, 26, 97, 17, 78, 78, 96, 83, 14, 88, 34, 89, 63, 72],
					 [21, 36, 23, 9, 75, 0, 76, 44, 20, 45, 35, 14, 0, 61, 33, 97, 34, 31, 33, 95],
					 [78, 17, 53, 28, 22, 75, 31, 67, 15, 94, 3, 80, 4, 62, 16, 14, 9, 53, 56, 92],
					 [16, 39, 5, 42, 96, 35, 31, 47, 55, 58, 88, 24, 0, 17, 54, 24, 36, 29, 85, 57],
					 [86, 56, 0, 48, 35, 71, 89, 7, 5, 44, 44, 37, 44, 60, 21, 58, 51, 54, 17, 58],
					 [19, 80, 81, 68, 5, 94, 47, 69, 28, 73, 92, 13, 86, 52, 17, 77, 4, 89, 55, 40],
					 [4, 52, 8, 83, 97, 35, 99, 16, 7, 97, 57, 32, 16, 26, 26, 79, 33, 27, 98, 66],
					 [88, 36, 68, 87, 57, 62, 20, 72, 3, 46, 33, 67, 46, 55, 12, 32, 63, 93, 53, 69],
					 [4, 42, 16, 73, 38, 25, 39, 11, 24, 94, 72, 18, 8, 46, 29, 32, 40, 62, 76, 36],
					 [20, 69, 36, 41, 72, 30, 23, 88, 34, 62, 99, 69, 82, 67, 59, 85, 74, 4, 36, 16],
					 [20, 73, 35, 29, 78, 31, 90, 1, 74, 31, 49, 71, 48, 86, 81, 16, 23, 57, 5, 54],
					 [1, 70, 54, 71, 83, 51, 54, 69, 16, 92, 33, 48, 61, 43, 52, 1, 89, 19, 67, 48]])

	def largest_multiplication(grid, i, j):
		# only do right, down, and down diag
		best = 0
		prod = 1
		if i + 3 <= 19:
			# down
			for elem in grid[i:i+4, j]:
				prod *= elem

			if prod > best:
				best = prod

		prod = 1
		if j + 3 <= 19:
			# right
			for elem in grid[i, j:j+4]:
				prod *= elem

			if prod > best:
				best = prod

		prod = 1
		if i + 3 <= 19 and j + 3 <= 19:
			# diag down right
			for offset in range(4):
				prod *= grid[i+offset, j+offset]

			if prod > best:
				best = prod

		prod = 1
		if i + 3 <= 19 and j - 3 >= 0:
			# diag down left
			for offset in range(4):
				prod *= grid[i+offset, j-offset]

			if prod > best:
				best = prod

		return best

	bests = np.zeros(grid.shape)
	for i in xrange(20):
		for j in xrange(20):
			cur = largest_multiplication(grid, i, j)
			bests[i,j] = cur

	return np.max(bests)

# TODO: this is brute force
def problem_12():
	def get_natural_num_sum(n):
		return (n**2 + n)/2

	n = 1
	while True:
		val = len(utils.get_divizors(get_natural_num_sum(n)))
		if val > 500:
			return get_natural_num_sum(n)
		n += 1



def problem_13():

	numbers = [37107287533902102798797998220837590246510135740250,
				46376937677490009712648124896970078050417018260538,
				74324986199524741059474233309513058123726617309629,
				91942213363574161572522430563301811072406154908250,
				23067588207539346171171980310421047513778063246676,
				89261670696623633820136378418383684178734361726757,
				28112879812849979408065481931592621691275889832738,
				44274228917432520321923589422876796487670272189318,
				47451445736001306439091167216856844588711603153276,
				70386486105843025439939619828917593665686757934951,
				62176457141856560629502157223196586755079324193331,
				64906352462741904929101432445813822663347944758178,
				92575867718337217661963751590579239728245598838407,
				58203565325359399008402633568948830189458628227828,
				80181199384826282014278194139940567587151170094390,
				35398664372827112653829987240784473053190104293586,
				86515506006295864861532075273371959191420517255829,
				71693888707715466499115593487603532921714970056938,
				54370070576826684624621495650076471787294438377604,
				53282654108756828443191190634694037855217779295145,
				36123272525000296071075082563815656710885258350721,
				45876576172410976447339110607218265236877223636045,
				17423706905851860660448207621209813287860733969412,
				81142660418086830619328460811191061556940512689692,
				51934325451728388641918047049293215058642563049483,
				62467221648435076201727918039944693004732956340691,
				15732444386908125794514089057706229429197107928209,
				55037687525678773091862540744969844508330393682126,
				18336384825330154686196124348767681297534375946515,
				80386287592878490201521685554828717201219257766954,
				78182833757993103614740356856449095527097864797581,
				16726320100436897842553539920931837441497806860984,
				48403098129077791799088218795327364475675590848030,
				87086987551392711854517078544161852424320693150332,
				59959406895756536782107074926966537676326235447210,
				69793950679652694742597709739166693763042633987085,
				41052684708299085211399427365734116182760315001271,
				65378607361501080857009149939512557028198746004375,
				35829035317434717326932123578154982629742552737307,
				94953759765105305946966067683156574377167401875275,
				88902802571733229619176668713819931811048770190271,
				25267680276078003013678680992525463401061632866526,
				36270218540497705585629946580636237993140746255962,
				24074486908231174977792365466257246923322810917141,
				91430288197103288597806669760892938638285025333403,
				34413065578016127815921815005561868836468420090470,
				23053081172816430487623791969842487255036638784583,
				11487696932154902810424020138335124462181441773470,
				63783299490636259666498587618221225225512486764533,
				67720186971698544312419572409913959008952310058822,
				95548255300263520781532296796249481641953868218774,
				76085327132285723110424803456124867697064507995236,
				37774242535411291684276865538926205024910326572967,
				23701913275725675285653248258265463092207058596522,
				29798860272258331913126375147341994889534765745501,
				18495701454879288984856827726077713721403798879715,
				38298203783031473527721580348144513491373226651381,
				34829543829199918180278916522431027392251122869539,
				40957953066405232632538044100059654939159879593635,
				29746152185502371307642255121183693803580388584903,
				41698116222072977186158236678424689157993532961922,
				62467957194401269043877107275048102390895523597457,
				23189706772547915061505504953922979530901129967519,
				86188088225875314529584099251203829009407770775672,
				11306739708304724483816533873502340845647058077308,
				82959174767140363198008187129011875491310547126581,
				97623331044818386269515456334926366572897563400500,
				42846280183517070527831839425882145521227251250327,
				55121603546981200581762165212827652751691296897789,
				32238195734329339946437501907836945765883352399886,
				75506164965184775180738168837861091527357929701337,
				62177842752192623401942399639168044983993173312731,
				32924185707147349566916674687634660915035914677504,
				99518671430235219628894890102423325116913619626622,
				73267460800591547471830798392868535206946944540724,
				76841822524674417161514036427982273348055556214818,
				97142617910342598647204516893989422179826088076852,
				87783646182799346313767754307809363333018982642090,
				10848802521674670883215120185883543223812876952786,
				71329612474782464538636993009049310363619763878039,
				62184073572399794223406235393808339651327408011116,
				66627891981488087797941876876144230030984490851411,
				60661826293682836764744779239180335110989069790714,
				85786944089552990653640447425576083659976645795096,
				66024396409905389607120198219976047599490197230297,
				64913982680032973156037120041377903785566085089252,
				16730939319872750275468906903707539413042652315011,
				94809377245048795150954100921645863754710598436791,
				78639167021187492431995700641917969777599028300699,
				15368713711936614952811305876380278410754449733078,
				40789923115535562561142322423255033685442488917353,
				44889911501440648020369068063960672322193204149535,
				41503128880339536053299340368006977710650566631954,
				81234880673210146739058568557934581403627822703280,
				82616570773948327592232845941706525094512325230608,
				22918802058777319719839450180888072429661980811197,
				77158542502016545090413245809786882778948721859617,
				72107838435069186155435662884062257473692284509516,
				20849603980134001723930671666823555245252804609722,
				53503534226472524250874054075591789781264330331690]
	num_sum = reduce(operator.add, numbers)
	return int(str(num_sum)[:10])

def problem_14():
	def next(n):
		if n%2 == 0:
			return n/2
		else:
			return 3*n + 1

	length_dict = {1:1}
	def get_length(n):
		if n in length_dict:
			return length_dict[n]
		length = 1 + get_length(next(n))
		length_dict[n] = length
		return length

	longest_length = 0
	for i in xrange(1, 1000000):
		length = get_length(i)
		if length > longest_length:
			longest_length = length
			longest_num = i

	return longest_num

def problem_15():
	return math.factorial(40)/(math.factorial(20)*math.factorial(20))

def problem_16():
	num = str(2**1000)
	fun = lambda x, y: int(x)+int(y)
	return reduce(fun, num)


def find_max_triangle_path(triangle):
	def get_parents(indx, row):
		left = right = None
		if indx - 1 >= 0:
			left = indx - 1
		if indx + 1 <= row:
			right = indx

		return (left, right)
	
	for r_indx, row in enumerate(triangle[1:]):
		for e_indx, elem in enumerate(row):
			parents = get_parents(e_indx, r_indx + 1)
			parents = [triangle[r_indx][p] for p in parents if p is not None]
			row[e_indx] += max(parents)

	return max(triangle[-1])

def problem_17():
	below_20 = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 
 			    11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen'} 

 	tens = {1: 'ten', 2:'twenty', 3:'thirty', 4:'forty', 5:'fifty', 6:'sixty', 7:'seventy', 8:'eighty', 9:'ninety'}
	hundred = 'hundred'
	thousand = 'thousand'

	def get_2_digit(n, is_suffix = False):

		tens_digit = n/10
		ones_digit = n%10

		if is_suffix and n == 0:
			return ''

		if is_suffix:
			name = ' and '
		else:
			name = ''

		if n < 20:
			return name + below_20[n]

		if ones_digit == 0:
			return name + tens[tens_digit]
		else:
			return name + tens[tens_digit] + ' ' + below_20[n%10]

	def get_3_digit(n, is_suffix = False):
		if n < 100:
			return get_2_digit(n, is_suffix)

		hundreds = n/100
		tens = n%100
		
		if is_suffix:
			name = ' '
		else:
			name = ''

		return name + below_20[hundreds] + ' ' + hundred + get_2_digit(n%100, True)

	def get_6_digit(n):
		if n < 1000:
			return get_3_digit(n, False)

		thousands = n/1000
		hundreds = (n%1000)/100
		# if hundreds == 0:
		# 	return get_3_digit(thousands) + ' thousand ' + 'and ' + get_3_digit(n%1000, True)
		# else:
		return get_3_digit(thousands) + ' thousand' + get_3_digit(n%1000, True)

	letter_sum = 0
	for i in xrange(1, 1001):
		letter_sum += len(get_6_digit(i).replace(' ', ''))

	return letter_sum


def problem_18():

		triangle = [[75],
				[95, 64],
				[17, 47, 82],
				[18, 35, 87, 10],
				[20, 04, 82, 47, 65],
				[19, 01, 23, 75, 03, 34],
				[88, 02, 77, 73, 07, 63, 67],
				[99, 65, 04, 28, 06, 16, 70, 92],
				[41, 41, 26, 56, 83, 40, 80, 70, 33],
				[41, 48, 72, 33, 47, 32, 37, 16, 94, 29],
				[53, 71, 44, 65, 25, 43, 91, 52, 97, 51, 14],
				[70, 11, 33, 28, 77, 73, 17, 78, 39, 68, 17, 57],
				[91, 71, 52, 38, 17, 14, 91, 43, 58, 50, 27, 29, 48],
				[63, 66, 04, 68, 89, 53, 67, 30, 73, 16, 69, 87, 40, 31],
				[04, 62, 98, 27, 23, 9 , 70, 98, 73, 93, 38, 53, 60, 04, 23]]

		return find_max_triangle_path(triangle)

def problem_19():
# 	1 Jan 1900 was a Monday.
#   How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?

	def get_month_length_given_year(month_index, year):
		# Thirty days has September,
		# April, June and November.
		# All the rest have thirty-one,
		# Saving February alone,
		# Which has twenty-eight, rain or shine.
		# And on leap years, twenty-nine.

		def is_leap_year(year):
			# A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.
			# return year%4 == 0 and not (year% 100 == 0 and not year%400 == 0)
			return year%4 == 0 and not (year% 100 == 0 and not year%400 == 0)

		if month_index == 1:
			# handle February case
			if is_leap_year(year):
				return 29
			else:
				return 28

		month_lengths = [31, None, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
		return month_lengths[month_index]

	num_suns = 0
	day_indx = 1 #since this is all starting on a monday, and I need to be zero indexed on sundays
	for year in xrange(1901, 2001):
		for month in xrange(12):
			day_indx += get_month_length_given_year(month, year)
			if (day_indx + 1)%7 == 0: #+1 is to check first of next month instead of last of this one 
				num_suns += 1

	if (day_indx+1)%7 == 0:
		num_suns -= 1 #un_count the first day of the 21st century if need be

	return num_suns

	


def problem_20():
	return reduce(operator.add, [int(x) for x in str(math.factorial(100))])

def problem_21():
	start_num = 3
	factors = [utils.get_divizors(x, True) for x in xrange(start_num, 10000)]
	factor_sum = [reduce(operator.add, factor) for factor in factors]

	# padd to make indexing easier
	for i in range(start_num):
		factor_sum.insert(0, 0)
	
	amicables = set()
	for indx in xrange(start_num, 10000):
		if factor_sum[indx] > 9999 or factor_sum[indx] < 0:
			continue

		if indx == factor_sum[factor_sum[indx]] and indx != factor_sum[indx]:
			amicables.update([indx, factor_sum[factor_sum[indx]]])

	return reduce(operator.add, amicables)

def problem_22():
	def convert_letter_to_alphabet_index(ch):
		return ord(ch.upper()) - 64
	def get_score_from_name(name, indx):
		return indx * reduce(operator.add, [convert_letter_to_alphabet_index(ch) for ch in name])

	with open('euler_resources/names.txt', 'rb') as f:
		reader = csv.reader(f, delimiter = ',')
		names_list = [row for row in reader]
		names_list = names_list[0] #there is only 1 row

	names_list.sort()
	scores_list = [get_score_from_name(name, indx + 1) for indx, name in enumerate(names_list)] #they want index 1 indexed
	return reduce(operator.add, scores_list)

def problem_23():
	start_num = 3
	factor_sums = [reduce(operator.add, utils.get_divizors(x, True)) for x in xrange(start_num, 28123)]

	ab_set = set([i + start_num for i,f_sum in enumerate(factor_sums) if f_sum > i + 3])
	ab_sums = set(reduce(operator.add, x) for x in itertools.combinations(ab_set, 2))
	ab_sums = ab_sums.union(set([x*2 for x in ab_set]))
	answers = set(xrange(28124)).difference(ab_sums)
	return reduce(operator.add, answers)

def problem_24():
	perms = itertools.permutations(range(10),10)
	for indx, val in enumerate(perms):
		if indx == 1000000 - 1:
			return int(reduce(operator.add, [str(x) for x in val]))

def problem_25():
	for i, fib in enumerate(utils.fibonnaci()):
		if len(str(fib)) >= 1000:
			return i

def count_decimal_digits_in_divizor(n):
	decimal.getcontext().prec = 100
	digits = str(decimal.Decimal(1)/decimal.Decimal(n))[2:]
	count = collections.Counter(digits)

	if len(digits) < 100:
		return count, 1
	else :
		counts = np.array([count[x] for x in count])
		max_val = max(counts)
		return count, sum(counts >= max_val - 4)

def problem_26():
	def get_n_decimal_digits(n, num, den):
		decimal.getcontext().prec = n
		digits = str(decimal.Decimal(num)/decimal.Decimal(den))[2:]
		return digits

	max_length = 0
	decimal_count = 1000
	for i in xrange(1,1000):
		digits = get_n_decimal_digits(decimal_count, 1, i)
		if len(digits) < decimal_count:
			continue

		worked = False
		length_guess = 1
		while(not worked):
			for j in xrange(10, decimal_count-10 - length_guess):
				if digits[j] != digits[j+length_guess]:
					length_guess += 1
					break
			else:
				worked = True
		if length_guess > max_length:
			max_length = length_guess
			best_val = i

	return max_length, best_val

def problem_27():
	max_n_estimate = 200
	primes = set(utils.get_primes_below(max_n_estimate**2 + max_n_estimate*1000 + 1000))
	b_primes = utils.get_primes_below(1000)

	best_count = 0
	best_a = 0
	best_b = 0
	for a in xrange(-999,1000):
		for b in b_primes:
			n = 1
			while(True):
				if (n**2 + n*a + b) in primes:
					n+=1
				else:
					break
			if n > best_count:
				best_count = n
				best_a = a
				best_b = b
	return best_a, best_b, best_count


def problem_28():
	results = np.array([[1,1,1,1]])
	start = np.array([0,0,0,0])
	for i, _ in enumerate(xrange(3, 1002, 2)):
		increase = np.array([0,-2,-4,-6]) + 8*(i+1)
		results = np.vstack([results, results[-1,:] + increase])

	return np.sum(results[1:,:]) + 1


def problem_29(): # Doesn't work
	base_max = 100
	exp_max = 100
	uesful_exp_list = []
	for i in xrange(2,base_max+1):
		exponents = range(2, exp_max+1)

		remove_exp1 = 2
		while i**remove_exp1 <= base_max:
			remove_exp2 = remove_exp1*2
			while remove_exp2 <= exp_max:
				if remove_exp2 in exponents:
					exponents.remove(remove_exp2)
					print "removed exp ", remove_exp2, " from ", i
				remove_exp2 += remove_exp1

			remove_exp1 += 1

		uesful_exp_list.append(exponents)

	# return sum([len(x) for x in uesfull_exp_list])
	huge_set = set()
	for indx, row in enumerate(uesful_exp_list):
		for elem in row:
			if (indx+2)**elem in huge_set:
				print "base = ", indx+2, " exp = ", elem
			huge_set.add((indx+2)**elem)
	return huge_set

def problem_29_brute():
	massive_set = set()
	for b in xrange(2,101):
		for e in xrange(2,101):
			massive_set.add(b**e)
	# return len(massive_set)
	return massive_set

def problem_30(): #brute
	results = []
	for i in xrange(2, 200000): #picked arbitrarily, got lucky
		nums = [int(x)**5 for x in str(i)]
		if sum(nums) == i:
			results.append(i)
	return results

def problem_31():

	def make_change_options(currency_values, n, post_fix = []):
		"""Makes change given an array of sorted currency values and an int"""
		results = []
		if len(currency_values) == 1:
			return [n//currency_values[0]] + post_fix

		for i in xrange(0, n//currency_values[-1] + 1):
			post_fix_copy = post_fix[:]
			results.extend(make_change_options(currency_values[:-1], n - i * currency_values[-1], [i] + post_fix_copy))

		return results

	currency = [1, 2, 5, 10, 20, 50, 100, 200]
	amount = 200
	return len(np.array(make_change_options(currency, amount)).reshape(-1, len(currency)))

def problem_32():
	def is_pandigital(numbers):
		"""determines if a list of numbers is pandigital"""
		count = collections.Counter()
		count.update([x for num in numbers for x in str(num)])
		return all(x == 1 for x in count.values()) and len(count.keys()) == 9 and '0' not in count.keys()

	a = 1
	b = 1
	prod_set = set()
	while(True):
		# a incrementing logic
		b = a
		while(True):
			# b incrementing logic
			if is_pandigital([a, b, a*b]):
				prod_set.add(a*b)

			b += 1
			if len(str(a) + str(b) + str(a*b)) > 9:
				break

		a += 1
		if len(str(a)*2 + str(a*a)) > 9:
			break
	return prod_set

def problem_36():
	num_sum = 0
	for i in xrange(1000000):
		if utils.is_palindrome("{0}".format(i)) and utils.is_palindrome("{0:b}".format(i)):
			num_sum += i
	return num_sum

def problem_42():
    count = 0

    with open('euler_resources/words.txt', 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        words = [row for row in reader][0]

	triangle_nums = utils.triangle_numbers()
	triangle_set = set([triangle_nums.next() for i in xrange(2000)]) #arbitrarily big
	for word in words:
		score = sum([ord(x.upper()) - 64 for x in word])
		if score in triangle_set:
			count += 1
	return count



def problem_67():

	with open('euler_resources/triangle.txt', 'rb') as f:
	    reader = csv.reader(f, delimiter=' ')
	    triangle = [row for row in reader]

	triangle = [[int(elem) for elem in row] for row in triangle]
	return find_max_triangle_path(triangle)


def problem_435():
	pass

