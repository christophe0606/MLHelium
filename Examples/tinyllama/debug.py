import struct
import numpy as np 



def f16_to_int(x):
    r = struct.pack('<e', x)
    i = struct.unpack('<h',x)[0]
    return(i & 0x0FFFF)

def f16_to_f8(x):
    r = struct.pack('<e', x)
    i = struct.unpack('<h',r)[0]
    i = (i >> 8) & 0x0FF
    return(i & 0x0FF)

def array_f16_to_f8(x):
    r = x.tobytes()[1::2]
    r = np.frombuffer(r,dtype='int8')
    return(r)


def f8_to_f16(x):
    x = x << 8
    print(f"{x:04x}")
    #r = struct.pack('<h', x)
    x = x.to_bytes(length=2,byteorder='little')
    f = struct.unpack('<e',x)[0]
    return(f)

def array_f8_to_f16(x):
    float16 = np.frombuffer(np.array(np.frombuffer(x, dtype='u1'), dtype='>u2').tobytes(), dtype='f2')
    return(float16)


def test(f):
    print(f)
    f16 = np.float16(f)
    f8 = f16_to_f8(f16)
    
    print(f"{f16_to_int(f16):04x}")
    print(f"{f8:02x}")
    print(f"{f8_to_f16(f8)}")
    print("")

def test_array(f,n=10):
    a = np.ones(n,dtype=np.float16)*np.float16(f)
    print(a)
    f8 = array_f16_to_f8(a)
    vhex = np.vectorize(lambda x: hex(x&0x0FF))
    print(vhex(f8))
    print(f8.dtype)
    f16 = array_f8_to_f16(f8)
    print(f16)
    
#test(4.23)
test_array(4.23)
#
#test(-4.23)
#test_array(-4.23)
#


def interleave(a):
   if len(a)<16:
       r = 16-len(a) 
       a = np.pad(a, (0,r), 'constant', constant_values=0)
   nb = len(a)
   b = np.zeros(len(a),dtype=np.int8)
   b[0::2] = a[:len(b[0::2])]
   b[1::2] = a[8:len(b[1::2])+8]
   return(b)

#N = 9
#a = np.array(range(N),dtype=np.int8) + np.ones(N,dtype=np.int8)

#print(interleave(a))

