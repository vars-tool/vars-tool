import numpy as np

import math

from typing import List
from scipy import stats

def sobol_sequence(
    sp:int, 
    params:int, 
    seed:int=None, 
    scramble:bool=True, 
    skip:int=1000, 
    leap:int=101
) -> np.ndarray:
    '''Sobol' sequences are low-discrepancy, quasi-random numbers. The
    code is taken from the `scipy dev-1.7` [1].
    
    Parameters
    ----------
    sp : int
        the number of sampling points
    params : int
        the number of parameters/factors/variables
    seed : int, optional
        randomization seed number, defaults to ``None``
    scramble : bool, optional
        scrambling the produced array, defaults to ``True``
    skip : int, optional
        the number of points to skip, defaults to ``1000``
    leap : int, optional
        the interval of picking values, defaults to ``101``
    
    Returns
    -------
    sobol_seq : array_like
        the sobol sequence
    
    Notes
    -----
    There are many versions of Sobol' sequences depending on their
    "direction numbers". This code uses direction numbers from [4]. Hence,
    the maximum number of dimension is 21201. The direction numbers have been
    precomputed with search criterion 6 and can be retrieved at
    https://web.maths.unsw.edu.au/~fkuo/sobol/
    
    References
    ----------
    .. [1] `scipy.stats._qmcs <https://github.com/scipy/scipy/stats/_qmc.py>`_
    .. [2] `scipy.stats.qmc.Sobol <https://scipy.github.io/devdocs/generated/scipy.stats.qmc.Sobol.html>`_
    .. [3] I. M. Sobol. The distribution of points in a cube 
           and the accurate evaluation of integrals. Zh. Vychisl.
           Mat. i Mat. Phys., 7:784-802, 1967.
    .. [4] S. Joe and F. Y. Kuo. Constructing sobol sequences with better
           two-dimensional projections. SIAM Journal on Scientific Computing,
           30(5):2635-2654, 2008.
    '''

    # check the seed number
    if seed:
        np.random.seed(int(seed))
    
    # check int signs
    sign_msg = ("the sign of '{}' must be positive (>0).")
    if sp < 0:
        raise ValueError(sign_msg.format('sp'))
    if params < 0:
        raise ValueError(sign_msg.format('params'))
    
    # check dtypes
    dtype_msg = ("dtype of '{}' array must be 'int', 'numpy.int32' or 'numpy.int64'.")
    if type(sp) not in [int, np.int32, np.int64]:
        raise ValueError(dtype_msg.format('sp'))
    if type(params) not in [int, np.int32, np.int64]:
        raise ValueError(dtype_msg.format('params'))
        
    if scramble is True:
        return _scrambled_sobol_generate(sp, params, skip, leap)
    else:
        return _sobol_generate(sp, params, skip, leap)

def _i4_bit_hi1(n):
    '''
    Description:
    ------------
    This function returns the position of the low 0 bit base 2 in an integer.
    
    
    Arguments:
    ----------
    :param n: the integer to be measured and should be >=0.
    :type n: {int, numpy.int32, numpy.int64}
    :param skip: the number of initial points to skip
    :type skip: {int, numpy.int32, numpy.int64}
    
    
    Returns:
    --------
    :return bit: the position of the low 1 bit.
    :rtype bit: int
    
    
    Credit:
    -------
    Original MATLAB version by John Burkardt
    Python version by Corrado Chisari    
    
    
    License:
    --------
    This code is distributed under the GNU LGPL license.
    
    
    Example:
    --------
          N    Binary     BIT
        ----    --------  ----
          0           0     0
          1           1     1
          2          10     2
          3          11     2 
          4         100     3
          5         101     3
          6         110     3
          7         111     3
          8        1000     4
          9        1001     4
         10        1010     4
         11        1011     4
         12        1100     4
         13        1101     4
         14        1110     4
         15        1111     4
         16       10000     5
         17       10001     5
        1023  1111111111    10
        1024 10000000000    11
        1025 10000000001    11
    '''

    i = math.floor (n)
    bit = 0
    while True:
        if (i <= 0):
            break
        bit += 1
        i = math.floor(i/2.)
        
    return bit


def _i4_bit_lo0(n):
    '''
    Description:
    ------------
    This function returns the position of the low 0 bit base 2 in an integer.
    
    
    Arguments:
    ----------
    :param n: the integer to be measured
    :type n: {int, numpy.int32, numpy.int64}
    :param skip: the number of initial points to skip
    :type skip: {int, numpy.int32, numpy.int64}
    
    
    Returns:
    --------
    :return bit: the position of the low 1 bit.
    :rtype bit: int
    
    
    Credit:
    -------
    Original MATLAB version by John Burkardt
    Python version by Corrado Chisari    
    
    
    License:
    --------
    This code is distributed under the GNU LGPL license.
    
    
    Example:
    --------

         N      Binary    BIT
       ----    --------  ----
          0           0     1
          1           1     2
          2          10     1
          3          11     3 
          4         100     1
          5         101     2
          6         110     1
          7         111     4
          8        1000     1
          9        1001     2
         10        1010     1
         11        1011     3
         12        1100     1
         13        1101     2
         14        1110     1
         15        1111     5
         16       10000     1
         17       10001     2
       1023  1111111111     1
       1024 10000000000     1
       1025 10000000001     1
    '''

    bit = 0
    i = math.floor(n)
    while True:
        bit = bit + 1
        i2 = math.floor(i / 2.)
        if (i == 2 * i2):
            break
        i = i2
        
    return bit


def _i4_sobol_generate(m:int, n:int, skip:int):
    '''
    Description:
    ------------
    The routine adapts the ideas of `Antonov and Salee` [1]_.
    Input/output, integer SEED, the ``seed`` for the sequence.
    This is essentially the index in the sequence of the quasirandom
    value to be generated. On output, ``seed`` has been set to the
    appropriate next value, usually simply ``seed+1``.
    If ``seed`` is less than 0 on input, it is treated as though it were 0.
    An input value of 0 requests the first (0th) element of the sequence.


    Arguments:
    ----------
    :param m: the spatial dimension
    :type m: {int, numpy.int32, numpy.int64}
    :param n: the number of points to generate
    :type n: {int, numpy.int32, numpy.int64}
    :param skip: the number of initial points to skip
    :type skip: {int, numpy.int32, numpy.int64}


    Returns:
    --------
    :return r: the points (R(m,n))
    :rtype r: np.ndarray


    Credit:
    -------
    Original MATLAB version by John Burkardt
    Python version by Corrado Chisari    


    License:
    --------
    This code is distributed under the GNU LGPL license.
    '''
    r = np.zeros((m,n))
    for j in range(1, n+1):
        seed = skip + j - 2
        [r[0:m,j-1], seed] = _i4_sobol(m, seed)
        
    return r


def _i4_sobol(dim_num:int, seed:int) -> List:
    '''
    Description:
    ------------
    The routine adapts the ideas of `Antonov and Salee` [1]_.
    Input/output, integer SEED, the ``seed`` for the sequence.
    This is essentially the index in the sequence of the quasirandom
    value to be generated. On output, ``seed`` has been set to the
    appropriate next value, usually simply ``seed+1``.
    If ``seed`` is less than 0 on input, it is treated as though it were 0.
    An input value of 0 requests the first (0th) element of the sequence.


    Arguments:
    ----------
    :param dim_num: the number of spatial dimensions.
    :type a,b: {int, numpy.int32, numpy.int64}, must be between 0 and 40
    :param seed: the seed number
    :type seed: {int, numpy.int32, numpy.int64}


    Returns:
    --------
    :return: a list of quasi and seed
    :rtype: list


    Credit:
    -------
    Original FORTRAN77 version by Bennett Fox
    Original MATLAB version by John Burkardt
    Python version by Corrado Chisari    


    License:
    --------
    This code is distributed under the GNU LGPL license.


    References:
    -----------
    .. [1] Antonov, Saleev,
           USSR Computational Mathematics and Mathematical Physics,
           Volume 19, 1980, pages 252 - 256.

    .. [2] Paul Bratley, Bennett Fox,
           Algorithm 659:
           Implementing Sobol's Quasirandom Sequence Generator,
           ACM Transactions on Mathematical Software,
           Volume 14, Number 1, pages 88-100, 1988.

    .. [3] Bennett Fox, Algorithm 647:
           Implementation and Relative Efficiency of Quasirandom
           Sequence Generators,
           ACM Transactions on Mathematical Software,
           Volume 12, Number 4, pages 362-376, 1986.

    .. [4] Ilya Sobol,
           USSR Computational Mathematics and Mathematical Physics,
           Volume 16, pages 236-242, 1977.

    .. [5] Ilya Sobol, Levitan, 
           The Production of Points Uniformly Distributed in a Multidimensional 
           Cube (in Russian),
           Preprint IPM Akad. Nauk SSSR, 
           Number 40, Moscow 1976.
    '''

    global atmost
    global dim_max
    global dim_num_save
    global initialized
    global lastq
    global log_max
    global maxcol
    global poly
    global recipd
    global seed_save
    global v

    if not 'initialized' in globals().keys():
        initialized = 0
        dim_num_save = -1

    if not initialized or dim_num != dim_num_save:
        initialized = 1
        dim_max = 40
        dim_num_save = -1
        log_max = 30
        seed_save = -1

    # initialize (part of) V.
        v = np.zeros((dim_max, log_max))
        v[0:40,0] = np.transpose([ \
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])

        v[2:40,1] = np.transpose([ \
            1, 3, 1, 3, 1, 3, 3, 1, \
            3, 1, 3, 1, 3, 1, 1, 3, 1, 3, \
            1, 3, 1, 3, 3, 1, 3, 1, 3, 1, \
            3, 1, 1, 3, 1, 3, 1, 3, 1, 3 ])

        v[3:40,2] = np.transpose([ \
            7, 5, 1, 3, 3, 7, 5, \
            5, 7, 7, 1, 3, 3, 7, 5, 1, 1, \
            5, 3, 3, 1, 7, 5, 1, 3, 3, 7, \
            5, 1, 1, 5, 7, 7, 5, 1, 3, 3 ])

        v[5:40,3] = np.transpose([ \
            1, 7, 9,13,11, \
            1, 3, 7, 9, 5,13,13,11, 3,15, \
            5, 3,15, 7, 9,13, 9, 1,11, 7, \
            5,15, 1,15,11, 5, 3, 1, 7, 9 ])

        v[7:40,4] = np.transpose([ \
            9, 3,27, \
            15,29,21,23,19,11,25, 7,13,17, \
            1,25,29, 3,31,11, 5,23,27,19, \
            21, 5, 1,17,13, 7,15, 9,31, 9 ])

        v[13:40,5] = np.transpose([ \
         37,33, 7, 5,11,39,63, \
         27,17,15,23,29, 3,21,13,31,25, \
         9,49,33,19,29,11,19,27,15,25 ])

        v[19:40,6] = np.transpose([ \
            13, \
            33,115, 41, 79, 17, 29,119, 75, 73,105, \
            7, 59, 65, 21, 3,113, 61, 89, 45,107 ])

        v[37:40,7] = np.transpose([ \
            7, 23, 39 ])


        poly = [ \
            1, 3, 7, 11, 13, 19, 25, 37, 59, 47, \
            61, 55, 41, 67, 97, 91, 109, 103, 115, 131, \
            193, 137, 145, 143, 241, 157, 185, 167, 229, 171, \
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299 ]

        atmost = 2**log_max - 1

    # find the number of bits in ATMOST.
        maxcol = _i4_bit_hi1(atmost)

    # initialize row 1 of V.
        v[0,0:maxcol] = 1

    # things to do only if the dimension changed.
    if dim_num != dim_num_save:
        if (dim_num<1 or dim_max<dim_num):
            raise RuntimeError('_i4_sobol is facing a fatal error; the \
            spatial dimension should be between 1 and 40')

        dim_num_save = dim_num
        
    # initialize the remaining rows of V.
        for i in range(2, dim_num+1):
            
    # the bits of the integer POLY(I) gives the form of polynomial I.
    # find the degree of polynomial I from binary encoding.
            j = poly[i-1]
            m = 0
            while ( 1 ):
                j = math.floor ( j / 2. )
                if ( j <= 0 ):
                    break
                m = m + 1

            # expand this bit pattern to separate components of the logical array INCLUD.
            j = poly[i-1]
            includ = np.zeros(m)
            for k in range(m, 0, -1):
                j2 = math.floor(j / 2.)
                includ[k-1] = (j != 2 * j2)
                j = j2
                
            # calculate the remaining elements of row I as explained
            # in Bratley and Fox, section 2.
            for j in range(m+1, maxcol+1):
                newv = v[i-1,j-m-1]
                l = 1
                for k in range(1, m+1):
                    l = 2 * l
                    if includ[k-1]:
                        newv = np.bitwise_xor( int(newv), int(l * v[i-1,j-k-1]) )
                v[i-1,j-1] = newv
                
        # multiply columns of V by appropriate power of 2.
        l = 1
        for j in range(maxcol-1, 0, -1):
            l = 2 * l
            v[0:dim_num,j-1] = v[0:dim_num,j-1] * l

        # RECIPD is 1/(common denominator of the elements in V)
        recipd = 1.0 / (2 * l)
        lastq=np.zeros(dim_num)

    seed = int(math.floor(seed))

    if seed<0:
        seed = 0

    if seed == 0:
        l = 1
        lastq=np.zeros(dim_num)

    elif seed == seed_save+1:

        # find the position of the right-hand zero in SEED.
        l = _i4_bit_lo0(seed)

    elif seed<=seed_save:

        seed_save = 0
        l = 1
        lastq = np.zeros(dim_num)

        for seed_temp in range(int(seed_save), int(seed)):
            l = _i4_bit_lo0(seed_temp)
            for i in range(1, dim_num+1):
                lastq[i-1] = np.bitwise_xor(int(lastq[i-1]), int(v[i-1,l-1]))

        l = _i4_bit_lo0(seed)

    elif (seed_save+1 < seed):

        for seed_temp in range(int(seed_save + 1), int(seed)):
            l = _i4_bit_lo0(seed_temp)
            for i in range(1, dim_num+1):
                lastq[i-1] = np.bitwise_xor (int(lastq[i-1]), int(v[i-1,l-1]))
        
        l = _i4_bit_lo0(seed)

    # check that the user is not calling too many times!
    if maxcol<l:
        raise RuntimeError('fatal error: too many calls')

    # calculate the new components of QUASI.
    quasi = np.zeros(dim_num)
    for i in range(1, dim_num+1):
        quasi[i-1] = lastq[i-1] * recipd
        lastq[i-1] = np.bitwise_xor(int(lastq[i-1]), int(v[i-1,l-1]))

    seed_save = seed
    seed = seed + 1

    return [quasi, seed]


def _i4_uniform(a, b, seed):
    '''
    Description:
    ------------
    This function returns a scaled pseudo-random I4.
    The pseudorandom number will be scaled to be uniformly distributed
    between ``a`` and ``b``.


    Arguments:
    ----------
    :param a,b: the minimum and maximum acceptable values, respectively.
    :type a,b: {int, numpy.int32, numpy.int64}
    :param seed: the seed number
    :type seed: {int, numpy.int32, numpy.int64}


    Returns:
    --------
    :return c: the randomly chosen integer.
    :rtype c: {int, numpy.int32, numpy.int64}
    :return seed: the updated seed.
    :rtype seed: {int, numpy.int32, numpy.int64}


    Credit:
    -------
    Original MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari    


    License:
    --------
    This code is distributed under the GNU LGPL license.


    References:
    -----------
    .. [1] Paul Bratley, Bennett Fox, Linus Schrage,
           A Guide to Simulation,
           Springer Verlag, pages 201-202, 1983.

    .. [2] Pierre L'Ecuyer,
           Random Number Generation,
           in Handbook of Simulation,
           edited by Jerry Banks,
           Wiley Interscience, page 95, 1998.

    .. [3] Bennett Fox, Algorithm 647:
           Implementation and Relative Efficiency of Quasirandom
           Sequence Generators,
           ACM Transactions on Mathematical Software,
           Volume 12, Number 4, pages 362-376, 1986.

    .. [4] Peter Lewis, Allen Goodman, James Miller
           A Pseudo-Random Number Generator for the System/360,
           IBM Systems Journal,
           Volume 8, pages 136-143, 1969.
    '''

    if (seed == 0):
        raise RuntimeError('input seed number is "0"; please change the seed number.')

    seed = math.floor(seed)
    a = round(a)
    b = round(b)

    seed = mod(seed, 2147483647)

    if (seed<0) :
        seed = seed + 2147483647

    k = math.floor(seed/127773)

    seed = 16807*(seed - k*127773) - k*2836

    if (seed < 0):
        seed = seed + 2147483647

    r = seed * 4.656612875e-10
    # scale R to lie between ``a-0.5`` and ``b+0.5``.
    r = (1.0-r)*(min(a, b)-0.5) + r*(max(a, b)+0.5)

    # use rounding to convert R to an integer between A and B.
    value = round(r)
    value = max(value, min(a, b))
    value = min(value, max(a, b))

    c = value

    return [int(c), int(seed)]


def _prime_ge(n:int) -> int:
    '''
    Description:
    ------------
    This function returns the smallest prime greater than or equal to N.


    Arguments:
    -----------
    :param n: the input number
    :type n: {int, numpy.int32, numpy.int64}


    Returns:
    --------
    :return p: the immediate next prime number
    :rtype p: {int, numpy.int32, numpy.int64}


    Credit:
    -------
    Original MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari    


    License:
    --------
    This code is distributed under the GNU LGPL license.


    Example:
    --------
        N     PRIME_GE

        10    2
        1     2
        2     2
        3     3
        4     5
        5     5
        6     7
        7     7
        8     11
        9     11
        10    11

    '''
    p = max(math.ceil(n), 2)
    while not _isprime(p):
        p = p + 1

    return p


def _isprime(n:int) -> bool:
    '''
    Description:
    ------------
    This functions returns ``True`` if `n` is a prime number, 
    ``False`` otherwise.


    Arguments:
    ----------
    :param n: the integer to be checked
    :type n: {int, numpy.int32, numpy.int64}


    Returns:
    --------
    :rtype: {bool; ``True`` or ``False``}


    Credit:
    -------
    Corrado Chisari


    License:
    --------
    This code is distributed under the GNU LGPL license.
    '''

    if n!=int(n) or n<1:
        return False
    p=2
    while p<n:
        if n%p==0:
            return False
        p+=1
    
    return True
 

def _sobol_generate(sp, params, skip, leap):
    '''
    Description:
    ------------
    generate sobol sequence for ``sp`` sampling points and ``params``
    parameters/factors/variabls. This functions returns the sequence
    without scrambling.
    
    
    Arguments:
    ----------
    :param sp: the number of sampling points
    :type sp: {int, np.int32, np.int64}
    :param params: the number of parameters
    :type params: {int, np.int32, np.int64}
    :param skip: the number of sampling points to skip from the beginning
    :type skip: {int, np.int32, np.int64}
    :param leap: the leap of numbers to choose from
    :type leap: {int, np.int32, np.int64}
    
    
    Returns:
    --------
    :return samples: the generated sample
    :rtype samples: np.ndarray
    
    
    Credit:
    -------
    Daniele Bigoni (dabi@imm.dtu.dk)
    '''

    # Generate sobol sequence
    samples = _i4_sobol_generate(params, sp*(leap+1), skip).T;

    # Remove leap values
    samples = samples[0:samples.shape[0]:(leap+1),:]

    return samples


def _scrambled_sobol_generate(sp, params, skip, leap):
    '''
    Description:
    ------------
    generate sobol sequence for ``sp`` sampling points and ``params``
    parameters/factors/variabls. Owen [1]_ scrambling method is used
    to shuffle the sampled array.
    
    
    Arguments:
    ----------
    :param sp: the number of sampling points
    :type sp: {int, np.int32, np.int64}
    :param params: the number of parameters
    :type params: {int, np.int32, np.int64}
    :param skip: the number of sampling points to skip from the beginning
    :type skip: {int, np.int32, np.int64}
    :param leap: the leap of numbers to choose from
    :type leap: {int, np.int32, np.int64}
    
    
    Returns:
    --------
    :return samples: the generated sample
    :rtype samples: np.ndarray
    

    Reference:
    ----------
    .. [1] Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
           Journal of Complexity, 14(4):466-489, December 1998.
    
    
    Credit:
    -------
    Daniele Bigoni (dabi@imm.dtu.dk)
    '''

    # Generate sobol sequence
    samples = _sobol_generate(sp, params, skip, leap);

    # Scramble the sequence
    for col in range(0, params):
        samples[:, col] = _scramble(samples[:, col]);

    return samples


def _scramble(X):
    """
    Description:
    ------------
    Owen [1]_ scrambling method to shuffle the sampled array.
    
    
    Arguments:
    ----------
    :param X: the input vector of data
    :type X: np.ndarray
    
    
    Returns:
    --------
    :return X: the scrambled vector of data
    :rtype X: np.ndarray
    

    Reference:
    ----------
    .. [1] Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
           Journal of Complexity, 14(4):466-489, December 1998.
    
    
    Credit:
    -------
    Daniele Bigoni (dabi@imm.dtu.dk)
    """

    N = len(X) - (len(X) % 2)
    
    idx = X[0:N].argsort()
    iidx = idx.argsort()
    
    # Generate binomial values and switch position for the second half of the array
    bi = stats.binom(1,0.5).rvs(size=N//2).astype(bool)
    pos = stats.uniform.rvs(size=N//2).argsort()
    
    # Scramble the indexes
    tmp = idx[0:N//2][bi];
    idx[0:N//2][bi] = idx[N//2:N][pos[bi]];
    idx[N//2:N][pos[bi]] = tmp;
    
    # Apply the scrambling
    X[0:N] = X[0:N][idx[iidx]];
    
    # Apply scrambling to sub intervals
    if N > 2:
        X[0:N//2] = _scramble(X[0:N//2])
        X[N//2:N] = _scramble(X[N//2:N])
    
    return X

