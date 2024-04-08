import argparse
from export import serialize_tensors,read_tensors

parser = argparse.ArgumentParser(
                    prog='convert_to_c',
                    description='Convert network to a C array')

parser.add_argument('filename')
parser.add_argument('-n', '--name',default="network")
parser.add_argument('-i', '--input')

args = parser.parse_args()

with open(args.input,"rb") as f:
    res = f.read()
    
COLS = 10
with open(f"{args.filename}.c","w") as c:
    nb = 0 
    l = len(res)
    print(f"""#include "{args.filename}.h"
#include "arm_math_types.h"

#ifndef ALIGN_NETWORK 
#define ALIGN_NETWORK __ALIGNED(16)
#endif

""",file=c)
    print("ALIGN_NETWORK",file=c)
    print(f"const uint8_t {args.name}[NB_{args.name.upper()}]={{",file=c,end="")
    for b in res:
        print("0x%02x," % b,file=c,end="")
        nb = nb + 1
        if (nb == COLS):
            nb = 0 
            print("",file=c)
    print("};",file=c)

with open(f"{args.filename}.h","w") as c:
    nb = 0 
    l = len(res)
    print(f"""#ifndef {args.name.upper()}_H 
#define {args.name.upper()}_H 

#include "arm_math_types.h"

#ifdef   __cplusplus
extern "C"
{{
#endif

""",file=c)
    print(f"#define NB_{args.name.upper()} {l}",file=c)
    print(f"extern const uint8_t {args.name}[NB_{args.name.upper()}];",file=c)
    
    print("""
#ifdef   __cplusplus
}
#endif
""",file=c)

    print("#endif",file=c)