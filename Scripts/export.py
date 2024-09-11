import numpy as np
import struct 

FLOAT32_TYPE = 0
FLOAT16_TYPE = 1
INT8_TYPE = 2
UINT32_TYPE = 3

def _dtype_to_int(d):
    if d == np.float32:
        return FLOAT32_TYPE
    if d == np.float16:
        return FLOAT16_TYPE
    if d == np.int8:
        return INT8_TYPE
    if d == np.uint32:
        return UINT32_TYPE
    return(None)

def _int_to_dtype(d):
    if d == FLOAT32_TYPE:
        return(np.float32)
    if d == FLOAT16_TYPE:
        return(np.float16)
    if d == INT8_TYPE:
        return(np.int8)
    if d == UINT32_TYPE:
        return(np.uint32)
    return(None)

def _align(file,pos,alignment):
    r = pos % alignment
    if r != 0:
       r = alignment - r 
       z = np.zeros(r,dtype=np.uint8)
       #z = np.ones(r,dtype=np.uint8)*255
       res = struct.pack(f'{len(z)}B', *z)
       file.write(res)
    return(r)

def _serialize_u32(file, tensor):
    """ writes one uint32 tensor to file that is open in wb mode """
    tensor = tensor.astype(np.uint32).flatten()
    b = struct.pack(f'{len(tensor)}L', *tensor)
    file.write(b)
    return(len(b))

def _serialize_f32(file, tensor):
    """ writes one f32 tensor to file that is open in wb mode """
    tensor = tensor.astype(np.float32).flatten()
    b = struct.pack(f'{len(tensor)}f', *tensor)
    file.write(b)
    return(len(b))

def _serialize_f16(file, tensor):
    """ writes one f16 tensor to file that is open in wb mode """
    tensor = tensor.astype(np.float16).flatten()
    b = struct.pack(f'{len(tensor)}e', *tensor)
    file.write(b)
    return(len(b))

def _serialize_int8(file, tensor):
    """ writes one f16 tensor to file that is open in wb mode """
    tensor = tensor.astype(np.int8).flatten()
    b = struct.pack(f'{len(tensor)}b', *tensor)
    file.write(b)
    return(len(b))

def convert_tensor(x):
    if x.dtype == np.double:
        x = x.astype(dtype=np.float32)
    return(x)

def serialize_tensors(file,tensors,alignment=8):
    """
    Serialize the tensors to a binary file with alignment.

    The format is as simple as possible and with enough information
    to be able to read the samples back with Python.

    The shapes and strides are not saved. The C code using this data
    has the knowledge of the shapes and strides.

    The format is:
    Nb of tensors : uint32 
    Array of uint32 lengths for each tensor. Length expressed in bytes.
    Array of uint32 datatype (to be able to read back with Python)
    Array of uint32 offset in the file where to find the tensor
    Aligned tensor follows
    """
    pos = 0

    # Convenience. User can work with default of double array
    # and they are converted to f32 here.
    tensors=[convert_tensor(x) for x in tensors]

    # Write number of tensors
    pos += _serialize_u32(file,np.array([len(tensors)]))

    # Write lengths of the tensors
    lengths=np.array([x.size for x in tensors],dtype=np.uint32)
    lengths_in_bytes=np.array([x.nbytes for x in tensors],dtype=np.uint32)

    pos += _serialize_u32(file,lengths_in_bytes)
    
    # Write datatypes of tensors
    dt=np.array([_dtype_to_int(x.dtype) for x in tensors],dtype=np.uint32)
    pos += _serialize_u32(file,dt)

    # Offsets of tensors in files
    offset_pos = pos
    offsets=np.zeros(len(tensors),dtype=np.uint32)
    # Write offsets of the tensors
    pos += _serialize_u32(file,offsets)
    pos += _align(file,pos,alignment)

    # Write tensors
    k = 0 
    for t in tensors:
        offsets[k] = pos 
        if t.dtype == np.float32:
           pos += _serialize_f32(file,t)
        elif t.dtype == np.float16:
           pos += _serialize_f16(file,t)
        elif t.dtype == np.int8:
           pos += _serialize_int8(file,t)
        elif t.dtype == np.uint32:
           pos += _serialize_u32(file,t)
        else:
           raise NameError(f"Unsupported datatype {str(t.dtype)}")
        
        k = k + 1
        if k != len(tensors):
           pos += _align(file,pos,alignment)

    file.seek(offset_pos)
    _serialize_u32(file,offsets)

def _read_uint32(file):
    b = file.read(4)
    nb_tensors = struct.unpack(f'L', b)
    return(nb_tensors[0])

def _read_uint32_array(file,nb):
    b = file.read(4*nb)
    a = struct.unpack(f'{nb}L', b)
    return(list(a))

def _read_f32_array(file,nb):
    b = file.read(4*nb)
    a = struct.unpack(f'{nb}f', b)
    return(list(a))

def _read_f16_array(file,nb):
    b = file.read(2*nb)
    a = struct.unpack(f'{nb}e', b)
    return(list(a))

def _read_int8_array(file,nb):
    b = file.read(nb)
    a = struct.unpack(f'{nb}b', b)
    return(list(a))

def read_tensors(file):
    res = []
    nb=_read_uint32(file)
    tensor_sizes = _read_uint32_array(file,nb)
    tensor_dt = _read_uint32_array(file,nb)

    tensor_offsets = _read_uint32_array(file,nb)

    for s,dt,o in zip(tensor_sizes,tensor_dt,tensor_offsets):
        # Position at tensor offset
        file.seek(o)

        # Read tensor
        if dt == FLOAT32_TYPE:
           a = _read_f32_array(file,s>>2)
        elif dt == FLOAT16_TYPE:
           a = _read_f16_array(file,s>>1)
        elif dt == INT8_TYPE:
           a = _read_int8_array(file,s)
        elif dt == UINT32_TYPE:
           a = _read_uint32_array(file,s>>2)
        res.append(a)

    return(res)
    
