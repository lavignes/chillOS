#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Mapping, Optional

import llvmlite.ir as ir
import llvmlite.binding as bind

@dataclass
class Type:
    name: str
    inner: ir.Type
    children: Optional[List['Type']] = None

PRIMS: Mapping[str, Type] = {
    '()': Type('()', ir.LiteralStructType([]), []),
    'Bool': Type('Bool', ir.IntType(1)),
    'U8': Type('U8', ir.IntType(8)),
    'I8': Type('I8', ir.IntType(8)),
    'U16': Type('U16', ir.IntType(16)),
    'I16': Type('I16', ir.IntType(16)),
    'U32': Type('U32', ir.IntType(32)),
    'I32': Type('I32', ir.IntType(32)),
    'U64': Type('U64', ir.IntType(64)),
    'I64': Type('I64', ir.IntType(64)),
    'Int': Type('Int', ir.IntType(64)),
    'UInt': Type('UInt', ir.IntType(64)),
}

Unit = PRIMS['()']
Bool = PRIMS['Bool']
U8 = PRIMS['U8']
I8 = PRIMS['I8']
U16 = PRIMS['U16']
I16 = PRIMS['I16']
U32 = PRIMS['U32']
I32 = PRIMS['I32']
U64 = PRIMS['U64']
I64 = PRIMS['I64']
Int = PRIMS['Int']
UInt = PRIMS['UInt']
