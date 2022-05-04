import bisect
import collections
import collections.abc
import functools
import heapq
import operator
import os.path
import random
from itertools import chain, combinations
from statistics import mean
import numpy as np

def sequence(iterable):
    """ 
    Converts iterable to a sequence
    """
    return iterable if isinstance(iterable, collections.abc.Sequence) else tuple([iterable])

def remove_all(item, seq):
    """
    Returns a copy of seq with that item removed.
    """
    if isinstance(seq, str):
        return seq.replace(item,'')
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.removed(item)
        return rest
    else:
        return [x for x in seq if x!=item]


def unique(seq):
    """
    Removes duplicates from the sequence
    """
    return list(set(seq))

def count(seq):
    """
    Number of "True"s in the sequence
    """
    return sum(map(bool, seq))

def multimap(items):
    """
    (key, val) --> {key: [val, ...], ...}
    """
    result = collections.defaultdict(list)
    for (key, val) in items:
        result[key].append(val)
    return dict(result)

def multimap_items(mmap):
    """
    Yields all (key, val) pairs stored in the multimap.
    """
    for (key, vals) in mmap.items():
        for val in vals:
            yield key, val


def product(numbers):
    """
    Returns multiplication of numbers in self.numbers
    """
    result = 1
    for x in numbers:
        result *= x
    return result

def first(iterable, default=None):
    """
    Returns the first iterable or default
    """
    return next(iter(iterable), default)
    
def is_in(elt, seq):
    """
    Compares with 'is' not '=='
    """
    return any(x is elt for x in seq)

def mode(data):
    """
    Return the mode (most common element) of the data. If there is a tie, return a random one.
    """
    [(item, count)] = collections.Counter(data).most_common(1)
    return item

def power_set(iterable):
    """
    Return combinations of the parameter iterable
    power_set([1,2,3]) --> (1,),(2,),(3,),(1,2,),(2,3,),(1,3,),(1,2,3)
    """
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]


def extend(s, var, val):
    """
    Copy dictionary s, and extend it by setting var to val
    """
    return {**s, var: val}

def flatten(seqs):
    return sum(seqs, [])


identity = lambda x: x

def argmin_random_tie(seq, key=identity):
    """
    Returns a minimum element of seq
    """
    return min(shuffled(seq), key=key)

def argmax_random_tie(seq, key=identity):
    return max(shuffled(seq), key=key)

def shuffled(iterable):
    """
    Randomly shuffle the iterable
    """
    items = list(iterable)
    random.shuffle(items)
    return items


def histogram(values, mode=0, bin_function=None):
    """
    Returns a list of (value, count) pairs, summarizing input values
    mode=0 --> increasing, mode=1 --> decreasing order
    bin_function!=None --> maps it over values first
    """
    if bin_function:
        values = map(bin_function, values)
    
    bins = {}
    for val in values:
        bins[val] = bins.get(val,0) + 1

    if mode: 
        return sorted(list(bins.items()), key=lambda x: (x[1], x[0]), reverse=True)
    else:
        return sorted(bins.items())


def dot_product(x, y):
    """
    Element-wise product of vectors x and y
    """
    return sum(_x*_y for _x,_y in zip(x,y))


def element_wise_product(x, y):
    """
    Element-wise product of vectors x and y in vector
    """
    assert len(x) == len(y)
    return np.multiply(x,y)

def matrix_multiplication(x, *y):
    """
    Multiplies matrix x with matrices of y
    """
    result = x
    for _y in y:
        result = np.matmul(result, _y)
    return result

def vector_add(a, b):
    """
    Component-wise addition of two vectors a and b
    """
    return tuple(map(operator.add, a, b))

def scalar_vector_product(x, y):
    """
    Scalar multiplication of vector x and number y
    """
    return np.multiply(x, y)

def probability(p):
    """
    Return true with probability p
    """
    return p>random.uniform(0.0, 1.0)

def weighted_sample_with_replacement(n, seq, weights):
    """
    Pick n sample from seq at random with replacement and the probability of each element 
    in proportion to its corresponding weight.
    """

    sample = weighted_sample(seq, weights)
    return [sample() for _ in range(n)]

def weighted_sampler(seq, weights):
    """
    Returns a random-sample function that picks from seq weighted by weights
    """
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def weighted_choice(choices):
    """
    A weighted version of random.choice
    """

    total = sum(w for _,w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c,w in choices:
        if upto + w >= r:
            return c,w
        upto += w


def rounder(numbers, d=4):
    """
    Round a single number or sequence of numbers, to d decimal places
    """
    if isinstance(numbers, (int, float)):
        return round(numbers, d)
    else:
        constructor = type(numbers)
        return constructor(rounder(n, d) for n in numbers)

def num_or_str(x):
    """
    Convert from string to number
    """

    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()

def euclidean_distance(x, y):
    return np.sqrt(sum((_x,_y)**2 for _x,_y in zip(x,y)))

def manhattan_distance(x, y):
    return sum(abs(_x-_y) for _x, _y in zip(x,y))

def hamming_distance(x, y):
    return sum(_x != _y for _x,_y in zip(x,y))

def cross_entropy_loss(x,y):
    return (-1.0/len(x)) * sum(_x*np.log(_y) + (1 - _x)*np.log(1-_y) for _x,_y in zip(x,y))

def mean_squared_error_loss(x,y):
    return (1.0/len(x))*sum((_x-_y)**2 for _x,_y in zip(x,y))

def rms_error(x,y):
    return np.sqrt(ms_error(x,y))

def ms_error(x,y):
    return mean((_x, _y)**2 for _x,_y in zip(x,y))

def mean_error(x,y):
    return mean(abs(_x-_y) for _x,_y in zip(x,y))

def mean_boolean_error(x,y):
    return mean(_x != _y for _x,_y in zip(x,y))

def normalize(dist):
    """
    Normalizes between 1 and 0.
    """
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key]/total
            assert 0 <= dist[key] <= 1 
        return dist

    total = sum(dist)
    return [(n/total) for n in dist]

def random_weights(min_value, max_value, num_weights):
    return [random.uniform(min_value, max_value) for _ in range(num_weights)]

def sigmoid(x):
    """
    Sigmoid function
    """
    return 1/(1+np.exp(-x))

def sigmoid_derivative(value):
    return value*(1-value)

def elu(x, alpha=0.01):
    return x if x>0 else alpha*(np.exp(x)-1)

def elu_derivative(value, alpha=0.01):
    return 1 if value>0 else alpha*np.exp(value)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(value):
    return 1-(value**2)

def leaky_relu(x, alpha=0.01):
    return x if x>0 else alpha*_x

def leaky_relu_derivative(value, alpha=0.01):
    return 1 if value > 0 else alpha 

def relu(x):
    return max(0, x)

def relu_derivative(value):
    return 1 if value>0 else 0

def step(x):
    return 1 if x>=0 else 0

def gaussian(mean, st_dev, x):
    return 1/ (np.sqrt(2*np.pi)*st_dev) * np.e**(-0.5 * (float(x-mean)/st_dev)**2)

def linear_kernel(x, y=None):
    if y is None:
        y = x
    return np.dot(x, y.T)

def polynomial_kernel(x, y=None, degree=2.0):
    if y is None:
        y = x
    return (1.0 + np.dot(x, y.T)) ** degree

def rbf_kernel(x, y=None, gamma=None):
    """
    Radial basis function kernel
    """
    if y is None:
        y=x
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    return np.exp(-gamma*(-2.0 * np.dot(x, y.T) + np.sum(x*x, axis=1).reshape((-1,1)) + np.sum(y*y, axis=1).reshape((1,-1))))


orientations = EAST, NORTH, WEST, SOUTH = [(1,0), (0,1), (-1,0), (0,-1)]
turns = LEFT, RIGHT = [(+1), (-1)]

def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc)%len(headings)]

def turn_right(heading):
    return turn_heading(heading, RIGHT)

def turn_left(heading):
    return turn_left(heading, LEFT)

def distance(a, b):
    xA, yA = a
    xB, yB = b
    return np.hypot((xA-xB), (yA-yB))

def distance_squared(a, b):
    xA, yA = a
    xB, yB = b
    return (xA-xB)**2 - (yA-yB)**2




class injection:
    """
    Dependency injection of temprorary values for global functions, 
    ex: 'with injection(DataBase = MockDataBase): 
    """

    def __init__(self, **kwds):
        self.new = kwds

    def __enter__(self):
        self.old = {v: globals()[v] for v in self.new}
        globals().update(self.new)

    def __exit__(self, type, value, traceback):
        globals().update(self.old)


def memoize(fn, slot=None, maxsize=32):
    """
    fn: make it remember the computed value for any argument list
    slot : store the results in that slot of first argument, store=False : use lru_cache for values
    """

    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn

def name(obj):
    """
    Find a reasonable name for given object
    """
    return (getattr(obj, 'name', 0) or getattr(obj, '__name__', 0) or
            getattr(getattr(obj,'__class__',0), '__name__', 0) or str(obj))
    


def isnumber(x):
    return hasattr(x, '__int__')

def issequence(x):
    return isisntace(x, collection.abc.Sequnce)


def print_table(table, header=None, sep='   ', numfmt='{}'):
    """
    Prints table in a fashionable way.
    header : if specified will printeda as a first row.
    sep: separator between columns
    numfmt: format for all numbers ex: numfmt = {.2f}
    """
    
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]
    
    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row] for row in table]

    sizes = list(map(lambda seq: max(map(len, seq)), list(zip(*[map(str, row) for row in table]))))

    for row in rable:
        print(sep.join(getattr(str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))

def open_data(name, mode='r'):
    aima_root = os.path.dirname(__file__)
    aima_file = os.path.join(aima_root, *['aima-data', name])

    return open(aima_file, mode=mode)

def failure_test(algoritm, tests):
    """
    tests a list with each element in the form: (values, failure_output)
    """
    return mean(int(algorithm(x) != y) for x,y in tests)


#Expressions

class Expr:
    """
    """
    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args

    def __neg__(self):
        return Expr('-', self)

    def __pos__(self):
        return Expr('+', self)

    def __invert__(self):
        return Expr('~', self)

    def __add__(self, rhs):
        return Expr('+', self, rhs)

    def __sub__(self, rhs):
        return Expr('-', self, rhs)

    def __mul__(self, rhs):
        return Expr('*', self, rhs)

    def __pow__(self, rhs):
        return Expr('**', self, rhs)

    def __mod__(self, rhs):
        return Expr('%', self, rhs)

    def __and__(self, rhs):
        return Expr('&', self, rhs)

    def __xor__(self, rhs):
        return Expr('^', self, rhs)

    def __rshift__(self, rhs):
        return Expr('>>', self, rhs)

    def __lshift__(self, rhs):
        return Expr('<<', self, rhs)

    def __truediv__(self, rhs):
        return Expr('/', self, rhs)

    def __floordiv__(self, rhs):
        return Expr('//', self, rhs)

    def __matmul__(self, rhs):
        return Expr('@', self, rhs)

    def __or__(self, rhs):
        if isinstance(rhs, Expression):
            return Expr('|', self, rhs)
        else:
            return PartialExpr(rhs, self)


    def __radd__(self, lhs):
        return Expr('+', lhs, self)

    def __rsub__(self, lhs):
        return Expr('-', lhs, self)

    def __rmul__(self, lhs):
        return Expr('*', lhs, self)

    def __rpow__(self, lhs):
        return Expr('**', lhs, self)

    def __rmod__(self, lhs):
        return Expr('%', lhs, self)

    def __rand__(self, lhs):
        return Expr('&', lhs, self)

    def __rxor__(self, lhs):
        return Expr('^', lhs, self)

    def __rrshift__(self, lhs):
        return Expr('>>', lhs, self)

    def __rlshift__(self, lhs):
        return Expr('<<', lhs, self)

    def __rtruediv__(self, lhs):
        return Expr('/', lhs, self)

    def __rfloordiv__(self, lhs):
        return Expr('//', lhs, self)

    def __rmatmul__(self, lhs):
        return Expr('@', lhs, self)

    def __call__(self, *args):
        """
        Call: if 'f' is a Symbol, then f(0) == Expr('f', 0).
        """

        if self.args:
            raise ValueError('Can only do a call for a Symbol, not an Expr')
        else:
            return Expr(self.op, *args)

    
    def __eq__(self, other):
        return isinstance(other, Expr) and self.op == other.op and self.args == other.args

    def __lt__(self, other):
        return isinstance(other, Expr) and str(self) < str(other)

    def __hash__(self):
        return hash(self.op) ^ hash(self.args)

    def __repr__(self):
        op = self.op
        args = [str(arg) for arg in self.args]
        #f(x) or f(x,y)
        if op.isidentifier(): 
            return '{}({})'.format(op, ', '. join(args)) if args else op
        # -x or -(x+1)
        elif len(args) == 1: 
            return op + args[0]
        # (x-y)
        else:
            opp = (' ' + op + ' ')
            return '(' + opp.join(args) + ')'
        

Number = (int, float, complex)
Expression = (Expr, Number)

def Symbol(name):
    """
    A Symbol is an Expression without args
    """
    return Expr(name)

def symbols(names):
    return tuple(Symbol(name) for name in names.replace(',', ' ').split())

def subexpressions(x):
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpressions(arg)

def arity(expression):
    """
    Number of sub-expressions in the expression
    """
    if isinstance(expression, Expr):
        return len(expression.args)
    else:
        return 0



class PartialExpr:

    def __init__(self, op, lhs):
        self.op, self.lhs = op, lsh
    
    def __or__(self, rhs):
        return Expr(self.op, self.lhs, rhs)
    
    def __repr__(self):
        return "PartialExpr('{}', {})".format(self.op, self.lhs)


def expr(x):
    """
    Shortcut to create an Expression. If x is a str:
        - identifiers are automaticall defined as Symbols.
        - ==> is treated as an infix |'==>'|, as are <== and <=>
    If x is an Expression, it returns unchanged
    """
    return eval(expr_handle_infix_ops(x), defaultkeydict(Symbol)) if isinstance(x, str) else x

infix_ops = '==> <== <=>'.split()

def expr_handle_infix_ops(x):
    """
    expr_handle_infix_ops('P==>Q') : "P |'==>'| Q"
    """
    for op in infix_ops:
        x = x.replace(op, '|' + repr(op) + '|')
    return x


class defaultkeydict(collections.defaultdict):
    """
    Like defaultdict, default_factory is a function of the key 
    d = defaultkeydict(len)
    d['four']
    : 4
    """
    
    def __missing__(self, key):
        self[key] = result = self.default_factory(key)
        return result


class hashabledict(dict):
    def __hash__(self):
        return 1


class PriorityQueue:
    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order =='min':
            self.f = f
        elif order == 'max':
            self.f = lambda x: -f(x)
        else:
            raise ValueError("Order must be min or max")

    def append(self, item):
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        for item in items:
            self.append(item)

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception("Trying to pop from empty Priority Queue")

    def __len__(self):
        return len(self.heap)

    def __contains__(self, key):
        return any([item == key for _,item in self.heap])

    def __getitem__(self, key):
        for value,item in self.heap:
            if item == key:
                return value

        raise KeyError(str(key) + "is not in the priority queue")
    
    def __delitem__(self, key):
        try:
            del self.heap[[item == key for _,item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + "is not in the priority queue")
        heapq.heapify(self.heap)


class Bool(int):
    """
    True --> T
    False --> F
    """
    __str__ = __repr__ = lambda self: 'T' if self else 'F'

T = Bool(True)
F = Bool(False)





