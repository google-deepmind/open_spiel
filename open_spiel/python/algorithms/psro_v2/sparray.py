"""
Module for sparse arrays using dictionaries. Inspired in part 
by ndsparse (https://launchpad.net/ndsparse) by Pim Schellart

Jan Erik Solem, Feb 9 2010.
solem@maths.lth.se (bug reports and feedback welcome)
"""

import numpy


class sparray(object):
    """ Class for n-dimensional sparse array objects using
        Python's dictionary structure.
    """
    def __init__(self, shape, default=0, dtype=float):
        
        self.__default = default #default value of non-assigned elements
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.__data = {}


    def __setitem__(self, index, value):
        """ set value to position given in index, where index is a tuple. """
        self.__data[index] = value

    def __getitem__(self, index):
        """ get value at position given in index, where index is a tuple. """
        return self.__data.get(index,self.__default)

    def __delitem__(self, index):
        """ index is tuples of element to be deleted. """
        if index in self.__data:
            del(self.__data[index])
            

    def __add__(self, other):
        """ Add two arrays. """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] + other.__default
            out.__default = self.__default + other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val + other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __sub__(self, other):
        """ Subtract two arrays. """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] - other.__default
            out.__default = self.__default - other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val - other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __mul__(self, other):
        """ Multiply two arrays (element wise). """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] * other.__default
            out.__default = self.__default * other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val * other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __div__(self, other):
        """ Divide two arrays (element wise). 
            Type of division is determined by dtype. """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] / other.__default
            out.__default = self.__default / other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val / other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __truediv__(self, other):
        """ Divide two arrays (element wise). 
            Type of division is determined by dtype. """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] / other.__default
            out.__default = self.__default / other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val / other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __floordiv__(self, other):
        """ Floor divide ( // ) two arrays (element wise). """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] // other.__default
            out.__default = self.__default // other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val // other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __mod__(self, other):
        """ mod of two arrays (element wise). """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] % other.__default
            out.__default = self.__default % other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val % other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __pow__(self, other):
        """ power (**) of two arrays (element wise). """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] ** other.__default
            out.__default = self.__default ** other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val ** other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __iadd__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] + other.__default
            self.__default = self.__default + other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val + other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __isub__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] - other.__default
            self.__default = self.__default - other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val - other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __imul__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] * other.__default
            self.__default = self.__default * other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val * other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __idiv__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] / other.__default
            self.__default = self.__default / other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val / other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __itruediv__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] / other.__default
            self.__default = self.__default / other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val / other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __ifloordiv__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] // other.__default
            self.__default = self.__default // other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val // other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __imod__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] % other.__default
            self.__default = self.__default % other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val % other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __ipow__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] ** other.__default
            self.__default = self.__default ** other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val ** other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __str__(self):
        return str(self.dense())

    def dense(self):
        """ Convert to dense NumPy array. """
        out = self.__default * numpy.ones(self.shape)
        for ind in self.__data:
            out[ind] = self.__data[ind]
        return out

    def sum(self):
        """ Sum of elements."""
        s = self.__default * numpy.array(self.shape).prod()
        for ind in self.__data:
            s += (self.__data[ind] - self.__default)
        return s


if __name__ == "__main__":
    
    #test cases
    
    #create a sparse array
    A = sparray((3,3))
    print('shape =', A.shape, 'ndim =', A.ndim)
    A[(1,1)] = 10
    A[2,2] = 10
    
    #access an element
    print(A[2,2])
    
    print('remove an element...')
    print(A)
    del(A[2,2])
    print(A)
    
    print('array with different default value...')
    B = sparray((3,3),default=3)
    print(B)

    print('adding...')
    print(A+A)
    print(A+B)
    print(B+B)
    
    print('subtracting...')
    print(A-A)
    print(A-B)
    print(B-B)
    
    print('multiplication...')
    print(A*A)
    print(A*B)
    print(B*B)
    
    print('sum of elements...')
    print(A.sum())
    
    print('mix with NumPy arrays...')
    print(A.dense() * numpy.ones((3,3)))
    
    print('Frobenius norm...')
    print(sum( (A.dense().flatten()-B.dense().flatten())**2 ))
    print(((A-B)*(A-B)).sum())

    
