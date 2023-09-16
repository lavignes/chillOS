#!/usr/bin/env python3

from abc import ABC
from dataclasses import dataclass
import functools
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Tuple, Union, Sequence, cast

import ply.lex as lex
import ply.yacc as yacc

eprint = functools.partial(print, file=sys.stderr)

@dataclass(frozen=True)
class Name:
    pkg: str
    ident: str

@dataclass(frozen=True)
class Type(ABC):
    pass

@dataclass(frozen=True)
class TypeInteger(Type):
    ffi: str

@dataclass(frozen=True)
class TypePointer(Type):
    ty: Union[Type, Name]
    mut: bool

@dataclass(frozen=True)
class TypeQPointer(Type):
    ty: TypePointer

@dataclass(frozen=True)
class TypeFallible(Type):
    ty: Union[Type, Name]

@dataclass(frozen=True)
class TypeArray(Type):
    ty: Union[Type, Name]
    size: 'Expr'

@dataclass(frozen=True)
class TypeSlice(Type):
    ty: TypePointer

@dataclass(frozen=True)
class TypeStruct(Type):
    fields: Sequence['Field']

@dataclass(frozen=True)
class TypeUnion(Type):
    fields: Sequence['Field']

@dataclass(frozen=True)
class TypeFn(Type):
    args: Sequence[Union[Type, Name]]
    rets: Union[Type, Name]

@dataclass(frozen=True)
class TypeBool(Type):
    ffi: str

@dataclass(frozen=True)
class TypeNil(Type):
    ffi: str

Nil = TypeNil(ffi='Nil')
Bool = TypeBool(ffi='Bool')
U8 = TypeInteger(ffi='U8')
I8 = TypeInteger(ffi='I8')
U16 = TypeInteger(ffi='U16')
I16 = TypeInteger(ffi='I16')
U32 = TypeInteger(ffi='U32')
I32 = TypeInteger(ffi='I32')
U64 = TypeInteger(ffi='U64')
I64 = TypeInteger(ffi='I64')
UInt = TypeInteger(ffi='UInt')
Int = TypeInteger(ffi='Int')

PRIMS: Mapping[Name, Type] = {
    Name('*', 'Nil'): Nil,
    Name('*', 'Bool'): Bool,
    Name('*', 'U8'): U8,
    Name('*', 'I8'): I8,
    Name('*', 'U16'): U16,
    Name('*', 'I16'): I16,
    Name('*', 'U32'): U32,
    Name('*', 'I32'): I32,
    Name('*', 'U64'): U64,
    Name('*', 'I64'): I64,
    Name('*', 'UInt'): UInt,
    Name('*', 'Int'): Int,
}

CUSTOM_TYPES: Dict[Name, Type] = {}

# very basic and buggy type checker and inference
# in the real compiler we need to make multiple passes
# over the AST to infer types correctly. this fails in lots
# of places besides basic integers and expressions
def expr_type(expr: 'Expr') -> Optional[Union[Type, Name]]:
    if expr.ty is not None:
        if isinstance(expr.ty, Name):
            return CUSTOM_TYPES.get(expr.ty, expr.ty)
        return expr.ty
    if isinstance(expr, ExprIndex):
        lhs = expr_type(expr.lhs)
        if isinstance(lhs, TypeArray):
            expr.ty = lhs.ty
            return lhs.ty
        if isinstance(lhs, TypeSlice):
            expr.ty = lhs.ty.ty
            return lhs.ty.ty
        raise SyntaxError('illegal index: {lhs}[{rhs}]')
    if isinstance(expr, ExprAccess):
        lhs = expr_type(expr.lhs)
        if isinstance(lhs, TypeStruct):
            for field in lhs.fields:
                if field.name == expr.field:
                    expr.ty = field.ty
                    return field.ty
        raise SyntaxError('illegal access: {lhs}.{rhs}')
    if isinstance(expr, ExprCast):
        lhs = expr_type(expr.expr)
        ty = expr.ty
        if lhs == ty:
            return lhs
        if isinstance(lhs, TypeInteger) and isinstance(ty, TypeInteger):
            return ty
        if isinstance(lhs, TypePointer) and isinstance(ty, TypePointer) and ty.ty == U8:
            return ty
        if isinstance(lhs, TypePointer) and lhs.ty == U8 and isinstance(ty, TypePointer):
            return ty
        raise SyntaxError(f'illegal cast: {lhs} as {ty}')
    if isinstance(expr, ExprBinOp):
        lhs = expr_type(expr.lhs)
        rhs = expr_type(expr.rhs)
        if not isinstance(lhs, Type) or not isinstance(rhs, Type):
            return None
        if expr.op == '+' or expr.op == '-':
            if isinstance(lhs, TypePointer) and isinstance(rhs, TypeInteger) and rhs.ffi == 'Int':
                expr.ty = lhs
                return lhs
        if expr.op == '<<' or expr.op == '>>':
            if isinstance(lhs, TypeInteger) and isinstance(rhs, TypeInteger) and rhs.ffi == 'UInt':
                expr.ty = lhs
                return lhs
            raise SyntaxError(f'illegal operation: {lhs} {expr.op} {rhs}')
        if expr.op in ['==', '!=', '>=', '<=', '<', '>'] and lhs == rhs:
            expr.ty = Bool
            return Bool
        if expr.op in ['&&', '||']:
            if isinstance(lhs, TypeBool) and lhs == rhs:
                expr.ty = Bool
                return Bool
            raise SyntaxError(f'illegal operation: {lhs} {expr.op} {rhs}')
        if lhs == rhs:
            expr.ty = lhs
            return lhs
        raise SyntaxError(f'illegal operation: {lhs} {expr.op} {rhs}')
    if isinstance(expr, ExprSlice):
        lhs = expr_type(expr.lhs)
        if not isinstance(lhs, TypePointer):
            return None
        expr.ty = TypeSlice(lhs)
        return TypeSlice(lhs)
    if isinstance(expr, ExprUnaryOp):
        rhs = expr_type(expr.rhs)
        if not isinstance(rhs, Type):
            return None
        if expr.op == '-':
            if isinstance(rhs, TypePointer):
                expr.ty = rhs
                return rhs
        if expr.op == '!':
            # use ^ <uint>::MAX to complement ints instead of !
            if isinstance(rhs, TypeBool):
                expr.ty = rhs
                return rhs
        if expr.op == '*':
            # TODO: the type of this result should indicate mut or not mut
            # in fact, I think all exprs might want to store whether they are
            # mut or perhaps have special cases for lvalues?
            # it is illegal to perform certain operations on this expr if it
            # isnt mut, and we're losing that information here.. though we still
            # have it in the AST, so we can emit the error there! Mutable operation
            # checking can be done post type-checking!
            if isinstance(rhs, TypePointer):
                expr.ty = rhs.ty
                return rhs
        if expr.op == '&':
            expr.ty = TypePointer(rhs, mut=False)
            return expr.ty
        if expr.op == '&mut':
            expr.ty = TypePointer(rhs, mut=True)
            return expr.ty
        raise SyntaxError(f'illegal operation: {expr.op} {rhs}')
    return None


COUNTER: int = 0

class Lexer:
    def __init__(self):
        self.inner = lex.lex(module=self)

    reserved = {
            'fn': 'FN',
            'return': 'RETURN',
            'continue': 'CONTINUE',
            'break': 'BREAK',
            'if': 'IF',
            'else': 'ELSE',
            'for': 'FOR',
            'pub': 'PUB',
            'let': 'LET',
            'as': 'AS',
            'use': 'USE',
            'mut': 'MUT',
            'extern': 'EXTERN',
            'type': 'TYPE',
            'pkg': 'PKG',
            'sizeof': 'SIZEOF',
            'union': 'UNION',
            'lengthof': 'LENGTHOF',
            'err': 'ERR',
            'ok': 'OK',
            'try': 'TRY',
            'nil': 'NIL',
    }

    tokens = list(reserved.values()) + ['ID', 'INTEGER', 'BOOL', 'BRACE_OPEN', 'BRACE_CLOSE',
            'BRACKET_OPEN', 'BRACKET_CLOSE', 'PLUS', 'MINUS', 'STAR', 'SOLIDUS', 'BANG', 'CARET',
            'AMPERSAND', 'PAREN_OPEN', 'PAREN_CLOSE', 'EQUAL', 'DOUBLE_EQUAL', 'LESS_THAN',
            'GREATER_THAN', 'LESS_EQUAL', 'GREATER_EQUAL', 'PIPE', 'COLON', 'SEMI', 'MODULUS',
            'DOT', 'DOUBLE_COLON', 'COMMA', 'STRING', 'SHIFT_LEFT', 'SHIFT_RIGHT', 'AND', 'OR',
            'NOT_EQUAL', 'PLUS_EQUAL', 'MINUS_EQUAL', 'STAR_EQUAL', 'SOLIDUS_EQUAL',
            'MODULUS_EQUAL', 'SHIFT_LEFT_EQUAL', 'SHIFT_RIGHT_EQUAL', 'AND_EQUAL', 'CARET_EQUAL',
            'PIPE_EQUAL', 'QUESTION', 'TILDE', 'HASH']

    t_BRACE_OPEN = r'{'
    t_BRACE_CLOSE = r'}'
    t_BRACKET_OPEN = r'\['
    t_BRACKET_CLOSE = r']'
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_STAR = r'\*'
    t_SOLIDUS = r'/'
    t_BANG = r'!'
    t_CARET = r'\^'
    t_AMPERSAND = r'&'
    t_PAREN_OPEN = r'\('
    t_PAREN_CLOSE = r'\)'
    t_EQUAL = r'='
    t_DOUBLE_EQUAL = r'=='
    t_LESS_THAN = r'<'
    t_GREATER_THAN = r'>'
    t_LESS_EQUAL = r'<='
    t_GREATER_EQUAL = r'>='
    t_PIPE = r'\|'
    t_COLON = r':'
    t_SEMI = r';'
    t_MODULUS = r'%'
    t_DOT = r'\.'
    t_DOUBLE_COLON = r'::'
    t_COMMA = r','
    t_SHIFT_LEFT = r'<<'
    t_SHIFT_RIGHT = r'>>'
    t_AND = r'&&'
    t_OR = r'\|\|'
    t_NOT_EQUAL = r'!='
    t_PLUS_EQUAL = r'\+='
    t_MINUS_EQUAL = r'-='
    t_STAR_EQUAL = r'\*='
    t_SOLIDUS_EQUAL = r'/='
    t_MODULUS_EQUAL = r'%='
    t_SHIFT_LEFT_EQUAL = r'<<='
    t_SHIFT_RIGHT_EQUAL = r'>>='
    t_AND_EQUAL = r'&='
    t_CARET_EQUAL = r'^='
    t_PIPE_EQUAL = r'\|='
    t_QUESTION = r'\?'
    t_TILDE = r'~'
    t_HASH = r'\#'

    t_ignore = ' \r\t'

    def t_comment(self, t):
        r'//.*\n'
        self.inner.lineno += 1

    def t_newline(self, t):
        r'\n+'
        self.inner.lineno += len(t.value)

    def t_error(self, t):
        eprint(f'ERROR: {self.inner.lineno}: unexpected character: `{t.value[0]}`')
        self.inner.skip(1)

    def t_BOOL(self, t):
        r'true|false'
        if cast(str, t.value) == 'true':
            t.value = True
        else:
            t.value = False
        return t

    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        if t.value == '_':
            global COUNTER
            COUNTER += 1
            t.value = f'__noname{COUNTER}'
        t.type = self.reserved.get(t.value, 'ID')
        return t

    def t_INTEGER(self, t):
        r"((0x[0-9a-fA-F][0-9a-fA-F_]*)|(0b[01][01_]*)|(\d[\d_]*)|('((\\x[0-9a-fA-F]{2})|[ -~])'))([iuIU](8|16|32|64)?)?"
        ty = None
        if cast(str, t.value).lower().endswith('u8'):
            t.value = t.value[:-2]
            ty = U8
        if cast(str, t.value).lower().endswith('i8'):
            t.value = t.value[:-2]
            ty = I8
        if cast(str, t.value).lower().endswith('u16'):
            t.value = t.value[:-3]
            ty = U16
        if cast(str, t.value).lower().endswith('i16'):
            t.value = t.value[:-3]
            ty = I16
        if cast(str, t.value).lower().endswith('u32'):
            t.value = t.value[:-3]
            ty = U32
        if cast(str, t.value).lower().endswith('i32'):
            t.value = t.value[:-3]
            ty = I32
        if cast(str, t.value).lower().endswith('u64'):
            t.value = t.value[:-3]
            ty = U64
        if cast(str, t.value).lower().endswith('i64'):
            t.value = t.value[:-3]
            ty = I64
        if cast(str, t.value).lower().endswith('u'):
            t.value = t.value[:-1]
            ty = UInt
        if cast(str, t.value).lower().endswith('i'):
            t.value = t.value[:-1]
            ty = Int

        if cast(str, t.value).startswith('0x'):
            t.value = int(cast(str, t.value[2:]).replace('_', ''), base=16)
        elif cast(str, t.value).startswith('0b'):
            t.value = int(cast(str, t.value[2:]).replace('_', ''), base=2)
        elif cast(str, t.value).startswith("'"):
            t.value = t.value[1:-1]
            if cast(str, t.value).startswith('\\x'):
                t.value = int(t.value[2:], base=16)
            else:
                t.value = ord(t.value)
        else:
            t.value = int(cast(str, t.value).replace('_', ''))

        t.value = (t.value, ty)
        return t

    def t_STRING(self, t):
        r'"((\\x[0-9a-fA-F]{2})|\\t|\\n|\\r|\\\\|\\\'|\\"|[ -~])*?"'
        t.value = t.value[1:-1]
        value = ''
        i = 0
        while i < len(t.value):
            if cast(str, t.value)[i:].startswith('\\x'):
                value += chr(int(t.value[i+2:i+4], base=16))
                i += 4
            elif cast(str, t.value)[i:].startswith('\\t'):
                value += '\t'
                i += 2
            elif cast(str, t.value)[i:].startswith('\\n'):
                value += '\n'
                i += 2
            elif cast(str, t.value)[i:].startswith('\\r'):
                value += '\r'
                i += 2
            elif cast(str, t.value)[i:].startswith('\\\\'):
                value += '\\'
                i += 2
            elif cast(str, t.value)[i:].startswith('\\\''):
                value += '\''
                i += 2
            elif cast(str, t.value)[i:].startswith('\\"'):
                value += '"'
                i += 2
            else:
                value += t.value[i]
                i += 1
        t.value = value
        return t

class Parser:
    tokens = Lexer.tokens

    def __init__(self):
        self.inner = yacc.yacc(module=self, write_tables=False, debug=False)
        self.scopes = [dict()]
        self.aliases = dict() # use foo as bar;
        self.return_types = []

    def parse(self, text: str) -> 'Pkg':
        lexer = Lexer()
        return self.inner.parse(text, lexer=lexer.inner)

    precedence = (
        ('left', 'OK', 'ERR'),
        ('left', 'TILDE'),
        ('left', 'OR'),
        ('left', 'AND'),
        ('left', 'PIPE'),
        ('left', 'CARET'),
        ('left', 'AMPERSAND'),
        ('left', 'DOUBLE_EQUAL', 'NOT_EQUAL'),
        ('left', 'LESS_THAN', 'LESS_EQUAL', 'GREATER_THAN', 'GREATER_EQUAL'),
        ('left', 'SHIFT_LEFT', 'SHIFT_RIGHT'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'STAR', 'SOLIDUS', 'MODULUS'),
        ('left', 'AS'),
        ('right', 'UMINUS', 'BANG', 'USTAR', 'UAMPERSAND', 'SIZEOF', 'LENGTHOF'),
        ('left', 'PAREN_OPEN', 'DOT', 'BRACKET_OPEN'),
    )

    def p_pkg(self, p):
        '''
        pkg : attrs PKG ID set_pkg_name SEMI pkg_items
        '''
        p[0] = Pkg(p[6], p[1])

    def p_set_pkg_name(self, p):
        '''
        set_pkg_name :
        '''
        self.pkg_name = p[-1]

    def p_error(self, p):
        raise SyntaxError(f'{p}')

    def p_empty(self, p):
        '''
        empty :
        '''

    def p_pkg_items_1(self, p):
        '''
        pkg_items : empty
        '''
        p[0] = []

    def p_pkg_items_2(self, p):
        '''
        pkg_items : pkg_items attr_pkg_item
        '''
        p[0] = p[1] + [p[2]]

    def p_attr_pkg_item(self, p):
        '''
        attr_pkg_item : attrs pkg_item
        '''
        p[2].attrs += p[1]
        p[0] = p[2]

    def p_attrs_1(self, p):
        '''
        attrs : empty
        '''
        p[0] = []

    def p_attrs_2(self, p):
        '''
        attrs : attrs attr
        '''
        p[0] = p[1] + [p[2]]

    def p_attr(self, p):
        '''
        attr : HASH BRACKET_OPEN ID string_list BRACKET_CLOSE
        '''
        p[0] = Attr([p[3]] + p[4])

    def p_string_list_1(self, p):
        '''
        string_list : empty
        '''
        p[0] = []

    def p_id_list_2(self, p):
        '''
        string_list : string_list STRING
        '''
        p[0] = p[1] + [p[2]]

    def p_pkg_item_1(self, p):
        '''
        pkg_item : PUB pkg_bind
                 | PUB pkg_fn
                 | PUB pkg_type
        '''
        p[2].pub = True
        p[0] = p[2]

    def p_pkg_item_2(self, p):
        '''
        pkg_item : pkg_bind
                 | pkg_fn
                 | pkg_type
        '''
        p[0] = p[1]

    def p_pkg_item_3(self, p):
        '''
        pkg_item : EXTERN
        '''

    def p_pkg_item_4(self, p):
        '''
        pkg_item : USE ID SEMI
        '''
        p[0] = PkgUse(line=p.lineno(1), name=p[2], attrs=[])

    def p_pkg_item_5(self, p):
        '''
        pkg_item : USE ID AS ID SEMI
        '''
        self.aliases[p[4]] = p[2]
        p[0] = PkgUse(line=p.lineno(1), name=p[2], attrs=[])

    def p_pkg_bind_1(self, p):
        '''
        pkg_bind : LET ID COLON type EQUAL expr SEMI
        '''
        name = Name(self.pkg_name, p[2])
        self.scopes[-1][name] = p[4]
        p[0] = PkgBind(line=p.lineno(1), name=name, ty=p[4], pub=False, val=p[6], attrs=[], mut=False)

    def p_pkg_bind_2(self, p):
        '''
        pkg_bind : LET ID COLON type EQUAL BRACE_OPEN init_list BRACE_CLOSE SEMI
        '''
        name = Name(self.pkg_name, p[2])
        self.scopes[-1][name] = p[4]
        val = ExprStruct(ty=p[4], vals=p[7])
        p[0] = PkgBind(line=p.lineno(1), name=name, ty=p[4], pub=False, val=val, attrs=[], mut=False)

    def p_pkg_bind_3(self, p):
        '''
        pkg_bind : LET ID COLON type EQUAL BRACKET_OPEN expr_list BRACKET_CLOSE SEMI
        '''
        name = Name(self.pkg_name, p[2])
        self.scopes[-1][name] = p[4]
        val = ExprArray(ty=p[4], vals=p[7])
        p[0] = PkgBind(line=p.lineno(1), name=name, ty=p[4], pub=False, val=val, attrs=[], mut=False)

    def p_pkg_bind_4(self, p):
        '''
        pkg_bind : LET MUT ID COLON type SEMI
        '''
        name = Name(self.pkg_name, p[3])
        self.scopes[-1][name] = p[5]
        p[0] = PkgBind(line=p.lineno(1), name=name, ty=p[5], pub=False, val=None, attrs=[], mut=True)

    def p_pkg_bind_5(self, p):
        '''
        pkg_bind : LET MUT ID COLON type EQUAL expr SEMI
        '''
        name = Name(self.pkg_name, p[3])
        self.scopes[-1][name] = p[5]
        p[0] = PkgBind(line=p.lineno(1), name=name, ty=p[5], pub=False, val=p[7], attrs=[], mut=True)

    def p_pkg_bind_6(self, p):
        '''
        pkg_bind : LET MUT ID COLON type EQUAL BRACE_OPEN init_list BRACE_CLOSE SEMI
        '''
        name = Name(self.pkg_name, p[3])
        self.scopes[-1][name] = p[5]
        val = ExprStruct(ty=p[5], vals=p[8])
        p[0] = PkgBind(line=p.lineno(1), name=name, ty=p[5], pub=False, val=val, attrs=[], mut=True)

    def p_pkg_bind_7(self, p):
        '''
        pkg_bind : LET MUT ID COLON type EQUAL BRACKET_OPEN expr_list BRACKET_CLOSE SEMI
        '''
        name = Name(self.pkg_name, p[3])
        self.scopes[-1][name] = p[5]
        val = ExprArray(ty=p[5], vals=p[8])
        p[0] = PkgBind(line=p.lineno(1), name=name, ty=p[5], pub=False, val=val, attrs=[], mut=True)

    def p_pkg_fn(self, p):
        '''
        pkg_fn : FN ID PAREN_OPEN begin_scope arg_list PAREN_CLOSE begin_return_type stmt_block end_scope end_return_type
        '''
        name = Name(self.pkg_name, p[2])
        self.scopes[-1][name] = name
        if isinstance(p[7], TypeNil):
            p[8] += [StmtRet(line=p.lineno(1), val=ExprNil(ty=Nil))] # add implied return nil
        elif isinstance(p[7], TypeFallible) and isinstance(p[7].ty, TypeNil):
            p[8] += [StmtRet(line=p.lineno(1), val=ExprFallible(ty=p[7], expr=ExprNil(ty=Nil), ok=True))]
        p[0] = PkgFn(line=p.lineno(1), name=name, ty=name, pub=False, attrs=[], args=p[5], rets=p[7], stmts=p[8])

    def p_begin_return_type_1(self, p):
        '''
        begin_return_type : empty
        '''
        self.return_types.append(Nil)
        p[0] = Nil

    def p_begin_return_type_2(self, p):
        '''
        begin_return_type : COLON type
        '''
        self.return_types.append(p[2])
        p[0] = p[2]

    def p_end_return_type(self, p):
        '''
        end_return_type : empty
        '''
        self.return_types.pop()

    def p_begin_scope(self, p):
        '''
        begin_scope : empty
        '''
        self.scopes.append(dict())

    def p_arg_list_1(self, p):
        '''
        arg_list : empty
        '''
        p[0] = []

    def p_arg_list_2(self, p):
        '''
        arg_list : ID COLON type
        '''
        self.scopes[-1][Name('*', p[1])] = p[3]
        p[0] = [FnArg(name=p[1], ty=p[3], mut=False)]

    def p_arg_list_3(self, p):
        '''
        arg_list : ID COLON type COMMA arg_list
        '''
        self.scopes[-1][Name('*', p[1])] = p[3]
        p[0] = [FnArg(name=p[1], ty=p[3], mut=False)] + p[5]

    def p_arg_list_4(self, p):
        '''
        arg_list : MUT ID COLON type
        '''
        self.scopes[-1][Name('*', p[2])] = p[4]
        p[0] = [FnArg(name=p[2], ty=p[4], mut=True)]

    def p_arg_list_5(self, p):
        '''
        arg_list : MUT ID COLON type COMMA arg_list
        '''
        self.scopes[-1][Name('*', p[2])] = p[4]
        p[0] = [FnArg(name=p[2], ty=p[4], mut=True)] + p[6]

    def p_pkg_type(self, p):
        '''
        pkg_type : TYPE ID COLON type SEMI
        '''
        name = Name(self.pkg_name, p[2])
        self.scopes[-1][name] = p[4]
        CUSTOM_TYPES[name] = p[4]
        p[0] = PkgType(line=p.lineno(1), name=name, ty=p[4], pub=False, attrs=[])

    def p_type_1(self, p):
        '''
        type : name
        '''
        for scope in reversed(self.scopes):
            if Name('*', p[1]) in scope:
                p[0] = scope[Name('*', p[1])]
                return
        p[0] = PRIMS.get(p[1], p[1])

    def p_type_2(self, p):
        '''
        type : AMPERSAND type
        '''
        p[0] = TypePointer(p[2], mut=False)

    def p_type_3(self, p):
        '''
        type : AMPERSAND MUT type
        '''
        p[0] = TypePointer(p[3], mut=True)

    def p_type_2_1(self, p):
        '''
        type : QUESTION type
        '''
        p[0] = TypeQPointer(TypePointer(p[2], mut=False))

    def p_type_3_1(self, p):
        '''
        type : QUESTION MUT type
        '''
        p[0] = TypeQPointer(TypePointer(p[3], mut=True))

    def p_type_4(self, p):
        '''
        type : BRACKET_OPEN expr BRACKET_CLOSE type
        '''
        p[0] = TypeArray(p[4], size=p[2])

    def p_type_5(self, p):
        '''
        type : BRACKET_OPEN BRACKET_CLOSE type
        '''
        p[0] = TypeSlice(TypePointer(p[3], mut=False))

    def p_type_6(self, p):
        '''
        type : BRACKET_OPEN BRACKET_CLOSE MUT type
        '''
        p[0] = TypeSlice(TypePointer(p[4], mut=True))

    def p_type_7(self, p):
        '''
        type : BRACE_OPEN field_list BRACE_CLOSE
        '''
        p[0] = TypeStruct(fields=p[2])

    def p_type_7_1(self, p):
        '''
        type : UNION BRACE_OPEN field_list BRACE_CLOSE
        '''
        p[0] = TypeUnion(fields=p[3])

    def p_type_8(self, p):
        '''
        type : PAREN_OPEN PAREN_CLOSE
        '''
        p[0] = Nil

    def p_type_9(self, p):
        '''
        type : FN PAREN_OPEN type_list PAREN_CLOSE
        '''
        p[0] = TypeFn(args=p[3], rets=Nil)

    def p_type_10(self, p):
        '''
        type : FN PAREN_OPEN type_list PAREN_CLOSE COLON type
        '''
        p[0] = TypeFn(args=p[3], rets=p[6])

    def p_type_11(self, p):
        '''
        type : BANG type
        '''
        p[0] = TypeFallible(ty=p[2])

    def p_type_list_1(self, p):
        '''
        type_list : empty
        '''
        p[0] = []

    def p_type_list_2(self, p):
        '''
        type_list : type
        '''
        p[0] = [p[1]]

    def p_type_list_3(self, p):
        '''
        type_list : type COMMA type_list
        '''
        p[0] = [p[1]] + p[3]

    def p_name_1(self, p):
        '''
        name : ID
        '''
        for scope in reversed(self.scopes):
            if Name('*', p[1]) in scope:
                p[0] = Name('*', p[1])
                return
        if Name('*', p[1]) in PRIMS:
            p[0] = Name('*', p[1])
            return
        p[0] = Name(self.pkg_name, p[1])

    def p_name_2(self, p):
        '''
        name : ID DOUBLE_COLON ID
        '''
        if p[1] in self.aliases:
            p[0] = Name(self.aliases[p[1]], p[3])
            return
        p[0] = Name(p[1], p[3])

    def p_field_list_1(self, p):
        '''
        field_list : empty
        '''
        p[0] = []

    def p_field_list_2(self, p):
        '''
        field_list : attrs ID COLON type
        '''
        p[0] = [Field(name=p[2], ty=p[4], attrs=p[1], pub=False)]

    def p_field_list_3(self, p):
        '''
        field_list : attrs ID COLON type COMMA field_list
        '''
        p[0] = [Field(name=p[2], ty=p[4], attrs=p[1], pub=False)] + p[6]

    def p_field_list_4(self, p):
        '''
        field_list : attrs PUB ID COLON type
        '''
        p[0] = [Field(name=p[2], ty=p[4], attrs=p[1], pub=True)]

    def p_field_list_5(self, p):
        '''
        field_list : attrs PUB ID COLON type COMMA field_list
        '''
        p[0] = [Field(name=p[2], ty=p[4], attrs=p[1], pub=True)] + p[6]

    def p_stmt_block(self, p):
        '''
        stmt_block : BRACE_OPEN begin_scope stmt_list BRACE_CLOSE end_scope
        '''
        p[0] = p[3]

    def p_end_scope(self, p):
        '''
        end_scope :
        '''
        self.scopes.pop()

    def p_stmt_list_1(self, p):
        '''
        stmt_list : empty
        '''
        p[0] = []

    def p_stmt_list_3(self, p):
        '''
        stmt_list : stmt_list stmt
        '''
        p[0] = p[1] + [p[2]]

    def p_stmt_1(self, p):
        '''
        stmt : LET ID COLON type EQUAL expr SEMI
        '''
        self.scopes[-1][Name('*', p[2])] = p[4]
        p[0] = StmtBind(line=p.lineno(1), name=p[2], ty=p[4], val=p[6], mut=False)

    def p_stmt_1_1(self, p):
        '''
        stmt : LET ID EQUAL expr SEMI
        '''
        ty = expr_type(p[4])
        self.scopes[-1][Name('*', p[2])] = ty
        p[0] = StmtBind(line=p.lineno(1), name=p[2], ty=cast(Type, ty), val=p[4], mut=False)

    def p_stmt_2(self, p):
        '''
        stmt : LET ID COLON type EQUAL BRACE_OPEN init_list BRACE_CLOSE SEMI
        '''
        self.scopes[-1][Name('*', p[2])] = p[4]
        val = ExprStruct(ty=p[4], vals=p[7])
        p[0] = StmtBind(line=p.lineno(1), name=p[2], ty=p[4], val=val, mut=False)

    def p_stmt_3(self, p):
        '''
        stmt : LET ID COLON type EQUAL BRACKET_OPEN expr_list BRACKET_CLOSE SEMI
        '''
        self.scopes[-1][Name('*', p[2])] = p[4]
        val = ExprArray(ty=p[4], vals=p[7])
        p[0] = StmtBind(line=p.lineno(1), name=p[2], ty=p[4], val=val, mut=False)

    def p_stmt_4(self, p):
        '''
        stmt : LET MUT ID COLON type SEMI
        '''
        self.scopes[-1][Name('*', p[3])] = p[5]
        p[0] = StmtBind(line=p.lineno(1), name=p[3], ty=p[5], val=None, mut=True)

    def p_stmt_5(self, p):
        '''
        stmt : LET MUT ID COLON type EQUAL expr SEMI
        '''
        self.scopes[-1][Name('*', p[3])] = p[5]
        p[0] = StmtBind(line=p.lineno(1), name=p[3], ty=p[5], val=p[7], mut=True)

    def p_stmt_5_1(self, p):
        '''
        stmt : LET MUT ID EQUAL expr SEMI
        '''
        ty = expr_type(p[5])
        self.scopes[-1][Name('*', p[3])] = ty
        p[0] = StmtBind(line=p.lineno(1), name=p[3], ty=cast(Type, ty), val=p[5], mut=True)

    def p_stmt_6(self, p):
        '''
        stmt : LET MUT ID COLON type EQUAL BRACE_OPEN init_list BRACE_CLOSE SEMI
        '''
        self.scopes[-1][Name('*', p[3])] = p[5]
        val = ExprStruct(ty=p[5], vals=p[8])
        p[0] = StmtBind(line=p.lineno(1), name=p[3], ty=p[5], val=val, mut=True)

    def p_stmt_7(self, p):
        '''
        stmt : LET MUT ID COLON type EQUAL BRACKET_OPEN init_list BRACKET_CLOSE SEMI
        '''
        self.scopes[-1][Name('*', p[3])] = p[5]
        val = ExprArray(ty=p[5], vals=p[8])
        p[0] = StmtBind(line=p.lineno(1), name=p[3], ty=p[5], val=val, mut=True)

    def p_stmt_8(self, p):
        '''
        stmt : RETURN SEMI
        '''
        ret = self.return_types[-1]
        if isinstance(ret, TypeFallible) and isinstance(ret.ty, TypeNil):
            p[0] = StmtRet(line=p.lineno(1), val=ExprFallible(ty=ret, expr=ExprNil(ty=Nil), ok=True))
            return
        p[0] = StmtRet(line=p.lineno(1), val=ExprNil(ty=Nil))

    def p_stmt_9(self, p):
        '''
        stmt : RETURN expr SEMI
        '''
        p[0] = StmtRet(line=p.lineno(1), val=p[2])

    def p_stmt_10(self, p):
        '''
        stmt : CONTINUE SEMI
        '''
        p[0] = StmtCont(line=p.lineno(1), label=None)

    def p_stmt_11(self, p):
        '''
        stmt : CONTINUE ID SEMI
        '''
        p[0] = StmtCont(line=p.lineno(1), label=p[2])

    def p_stmt_12(self, p):
        '''
        stmt : BREAK SEMI
        '''
        p[0] = StmtBreak(line=p.lineno(1), label=None)

    def p_stmt_13(self, p):
        '''
        stmt : BREAK ID SEMI
        '''
        p[0] = StmtBreak(line=p.lineno(1), label=p[2])

    def p_stmt_14(self, p):
        '''
        stmt : if_stmt
             | if_let_stmt
             | for_stmt
             | for_let_stmt
             | expr_stmt SEMI
             | try_stmt
        '''
        p[0] = p[1]

    def p_stmt_15(self, p):
        '''
        stmt : BRACE_OPEN begin_scope stmt_list BRACE_CLOSE end_scope
        '''
        p[0] = StmtBlock(line=p.lineno(1), stmts=p[3], label=None)

    def p_stmt_16(self, p):
        '''
        stmt : BRACE_OPEN DOUBLE_COLON ID begin_scope stmt_list BRACE_CLOSE end_scope
        '''
        self.scopes[-1][Name('*', p[3])] = p[3]
        p[0] = StmtBlock(line=p.lineno(1), stmts=p[5], label=p[3])

    def p_expr_stmt_1(self, p):
        '''
        expr_stmt : expr EQUAL expr
                  | expr PLUS_EQUAL expr
                  | expr MINUS_EQUAL expr
                  | expr STAR_EQUAL expr
                  | expr SOLIDUS_EQUAL expr
                  | expr MODULUS_EQUAL expr
                  | expr SHIFT_LEFT_EQUAL expr
                  | expr SHIFT_RIGHT_EQUAL expr
                  | expr AND_EQUAL expr
                  | expr CARET_EQUAL expr
                  | expr PIPE_EQUAL expr
        '''
        p[0] = StmtAssign(line=p.lineno(2), lhs=p[1], op=p[2], rhs=p[3])

    def p_expr_stmt_2(self, p):
        '''
        expr_stmt : expr PAREN_OPEN expr_list PAREN_CLOSE
        '''
        expr = ExprCall(ty=None, lhs=p[1], args=p[3])
        p[0] = StmtCall(line=p.lineno(2), expr=expr)

    def p_for_stmt_1(self, p):
        '''
        for_stmt : FOR stmt_block
        '''
        p[0] = StmtFor(line=p.lineno(1), expr=None, stmts=p[2], label=None)

    def p_for_stmt_2(self, p):
        '''
        for_stmt : FOR DOUBLE_COLON ID stmt_block
        '''
        self.scopes[-1][Name('*', p[3])] = p[4] # ID should be in inner scope
        p[0] = StmtFor(line=p.lineno(1), expr=None, stmts=p[4], label=p[3])

    def p_for_stmt_3(self, p):
        '''
        for_stmt : FOR expr stmt_block
        '''
        p[0] = StmtFor(line=p.lineno(1), expr=p[2], stmts=p[3], label=None)

    def p_for_stmt_4(self, p):
        '''
        for_stmt : FOR DOUBLE_COLON ID expr stmt_block
        '''
        self.scopes[-1][Name('*', p[3])] = p[3] # should be in inner scope
        p[0] = StmtFor(line=p.lineno(1), expr=p[4], stmts=p[5], label=p[3])

    def p_for_let_stmt_1(self, p):
        '''
        for_let_stmt : FOR LET if_let_bind stmt_block end_scope
        '''
        p[0] = StmtForLet(line=p.lineno(1), bind=p[3], stmts=p[4], label=None)

    def p_for_let_stmt_2(self, p):
        '''
        for_stmt : FOR DOUBLE_COLON ID LET if_let_bind stmt_block end_scope
        '''
        self.scopes[-1][Name('*', p[3])] = p[3] # should be in inner scope
        p[0] = StmtForLet(line=p.lineno(1), bind=p[5], stmts=p[6], label=p[3])

    def p_if_stmt_1(self, p):
        '''
        if_stmt : IF expr stmt_block
        '''
        p[0] = StmtIf(line=p.lineno(1), expr=p[2], stmts=p[3], else_stmts=[], label=None)

    def p_if_stmt_2(self, p):
        '''
        if_stmt : IF expr stmt_block ELSE stmt_block
        '''
        p[0] = StmtIf(line=p.lineno(1), expr=p[2], stmts=p[3], else_stmts=p[5], label=None)

    def p_if_stmt_3(self, p):
        '''
        if_stmt : IF expr stmt_block ELSE if_stmt
        '''
        p[0] = StmtIf(line=p.lineno(1), expr=p[2], stmts=p[3], else_stmts=[p[5]], label=None)

    def p_if_stmt_4(self, p):
        '''
        if_stmt : IF DOUBLE_COLON ID expr stmt_block
        '''
        self.scopes[-1][Name('*', p[3])] = p[3] # should be in inner scope
        p[0] = StmtIf(line=p.lineno(1), expr=p[4], stmts=p[5], else_stmts=[], label=p[3])

    def p_if_stmt_5(self, p):
        '''
        if_stmt : IF DOUBLE_COLON ID expr stmt_block ELSE stmt_block
        '''
        self.scopes[-1][Name('*', p[3])] = p[3] # should be in inner scope
        p[0] = StmtIf(line=p.lineno(1), expr=p[4], stmts=p[5], else_stmts=p[7], label=p[3])

    def p_if_stmt_6(self, p):
        '''
        if_stmt : IF DOUBLE_COLON ID expr stmt_block ELSE if_stmt
        '''
        self.scopes[-1][Name('*', p[3])] = p[3] # should be in inner scope
        p[0] = StmtIf(line=p.lineno(1), expr=p[4], stmts=p[5], else_stmts=[p[7]], label=p[3])

    def p_if_let_stmt_1(self, p):
        '''
        if_let_stmt : IF LET if_let_bind stmt_block end_scope
        '''
        p[0] = StmtIfLet(line=p.lineno(1), taken=p[3], stmts=p[4], not_taken=None, else_stmts=[], label=None)

    def p_if_let_stmt_2(self, p):
        '''
        if_let_stmt : IF LET if_let_bind stmt_block end_scope ELSE stmt_block
        '''
        p[0] = StmtIfLet(line=p.lineno(1), taken=p[3], stmts=p[4], not_taken=None, else_stmts=p[7], label=None)

    def p_if_let_stmt_2_1(self, p):
        '''
        if_let_stmt : IF LET if_let_bind stmt_block end_scope ELSE else_let_bind stmt_block end_scope
        '''
        p[0] = StmtIfLet(line=p.lineno(1), taken=p[3], stmts=p[4], not_taken=p[7], else_stmts=p[8], label=None)

    def p_if_let_stmt_5(self, p):
        '''
        if_let_stmt : IF DOUBLE_COLON ID LET if_let_bind stmt_block end_scope
        '''
        self.scopes[-1][Name('*', p[3])] = p[3] # should be in inner scope
        p[0] = StmtIfLet(line=p.lineno(1), taken=p[5], stmts=p[6], not_taken=None, else_stmts=[], label=p[3])

    def p_try_let_stmt_1(self, p):
        '''
        try_stmt : TRY LET ID COLON type EQUAL expr try_else
        '''
        self.scopes[-1][Name('*', p[3])] = p[5]
        bind = StmtBind(line=p.lineno(1), name=p[3], ty=p[5], val=p[7], mut=False)
        p[0] = StmtTryLet(line=p.lineno(1), bind=bind, else_bind=p[8][0], else_stmts=p[8][1], func_return_type=self.return_types[-1])

    def p_try_let_stmt_2(self, p):
        '''
        try_stmt : TRY LET MUT ID COLON type EQUAL expr try_else
        '''
        self.scopes[-1][Name('*', p[4])] = p[6]
        bind = StmtBind(line=p.lineno(1), name=p[4], ty=p[6], val=p[8], mut=True)
        p[0] = StmtTryLet(line=p.lineno(1), bind=bind, else_bind=p[9][0], else_stmts=p[9][1], func_return_type=self.return_types[-1])

    def p_try_else_1(self, p):
        '''
        try_else : SEMI
        '''
        p[0] = (None, [])

    def p_try_else_2(self, p):
        '''
        try_else : ELSE stmt_block
        '''
        p[0] = (None, p[2])

    def p_try_else_3(self, p):
        '''
        try_else : ELSE else_let_bind stmt_block end_scope
        '''
        p[0] = (p[2], p[3])

    def p_if_let_bind_1(self, p):
        '''
        if_let_bind : begin_scope ID COLON type EQUAL expr
        '''
        self.scopes[-1][Name('*', p[2])] = p[4]
        p[0] = StmtBind(line=p.lineno(1), name=p[2], ty=p[4], val=p[6], mut=False)

    def p_if_let_bind_2(self, p):
        '''
        if_let_bind : begin_scope MUT ID COLON type EQUAL expr
        '''
        self.scopes[-1][Name('*', p[3])] = p[5]
        p[0] = StmtBind(line=p.lineno(1), name=p[3], ty=p[5], val=p[7], mut=True)

    def p_else_let_bind_1(self, p):
        '''
        else_let_bind : begin_scope ID
        '''
        self.scopes[-1][Name('*', p[2])] = Int
        p[0] = StmtBind(line=p.lineno(1), name=p[2], ty=Int, val=None, mut=False)

    def p_else_let_bind_2(self, p):
        '''
        else_let_bind : begin_scope MUT ID
        '''
        self.scopes[-1][Name('*', p[4])] = Int
        p[0] = StmtBind(line=p.lineno(1), name=p[3], ty=Int, val=None, mut=True)

    def p_expr_1(self, p):
        '''
        expr : binary_expr
             | unary_expr
             | index_expr
             | view_expr
             | call_expr
             | cast_expr
             | field_expr
             | sizeof_expr
             | lengthof_expr
             | primary_expr
             | fallible_expr
        '''
        p[0] = p[1]

    def p_expr_2(self, p):
        '''
        expr : PAREN_OPEN expr PAREN_CLOSE
        '''
        p[0] = p[2]

    def p_index_expr(self, p):
        '''
        index_expr : expr BRACKET_OPEN expr BRACKET_CLOSE
        '''
        p[0] = ExprIndex(ty=None, lhs=p[1], rhs=p[3])

    def p_view_expr(self, p):
        '''
        view_expr : expr TILDE expr
        '''
        p[0] = ExprSlice(ty=None, lhs=p[1], rhs=p[3])

    def p_field_expr(self, p):
        '''
        field_expr : expr DOT ID
        '''
        p[0] = ExprAccess(ty=None, lhs=p[1], field=p[3])

    def p_call_expr(self, p):
        '''
        call_expr : expr PAREN_OPEN expr_list PAREN_CLOSE
        '''
        p[0] = ExprCall(ty=None, lhs=p[1], args=p[3])

    def p_expr_list_1(self, p):
        '''
        expr_list : empty
        '''
        p[0] = []

    def p_expr_list_2(self, p):
        '''
        expr_list : expr
        '''
        p[0] = [p[1]]

    def p_expr_list_3(self, p):
        '''
        expr_list : expr COMMA expr_list
        '''
        p[0] = [p[1]] + p[3]

    def p_binary_expr(self, p):
        '''
        binary_expr : expr PLUS expr
                    | expr MINUS expr
                    | expr STAR expr
                    | expr SOLIDUS expr
                    | expr AND expr
                    | expr OR expr
                    | expr MODULUS expr
                    | expr AMPERSAND expr
                    | expr PIPE expr
                    | expr CARET expr
                    | expr LESS_THAN expr
                    | expr GREATER_THAN expr
                    | expr LESS_EQUAL expr
                    | expr GREATER_EQUAL expr
                    | expr DOUBLE_EQUAL expr
                    | expr NOT_EQUAL expr
                    | expr SHIFT_LEFT expr
                    | expr SHIFT_RIGHT expr
        '''
        p[0] = ExprBinOp(ty=None, lhs=p[1], op=p[2], rhs=p[3])

    def p_cast_expr_1(self, p):
        '''
        cast_expr : expr AS type
        '''
        p[0] = ExprCast(ty=p[3], expr=p[1])

    def p_fallible_expr_1(self, p):
        '''
        fallible_expr : expr OK
        '''
        p[0] = ExprFallible(ty=self.return_types[-1], expr=p[1], ok=True)

    def p_fallible_expr_1_1(self, p):
        '''
        fallible_expr : expr OK type
        '''
        p[0] = ExprFallible(ty=p[3], expr=p[1], ok=True)

    def p_fallible_expr_2(self, p):
        '''
        fallible_expr : expr ERR
        '''
        p[0] = ExprFallible(ty=self.return_types[-1], expr=p[1], ok=False)

    def p_fallible_expr_2_1(self, p):
        '''
        fallible_expr : expr ERR type
        '''
        p[0] = ExprFallible(ty=p[3], expr=p[1], ok=False)

    def p_sizeof_expr(self, p):
        '''
        sizeof_expr : SIZEOF type
        '''
        p[0] = ExprSizeof(ty=UInt, rhs=p[2])

    def p_lengthof_expr(self, p):
        '''
        lengthof_expr : LENGTHOF expr
        '''
        p[0] = ExprLengthof(ty=UInt, rhs=p[2])

    def p_unary_expr_1(self, p):
        '''
        unary_expr : MINUS expr %prec UMINUS
                   | AMPERSAND expr %prec UAMPERSAND
                   | STAR expr %prec USTAR
                   | BANG expr %prec BANG
        '''
        p[0] = ExprUnaryOp(ty=None, op=p[1], rhs=p[2])

    def p_unary_expr_2(self, p):
        '''
        unary_expr : AMPERSAND MUT expr %prec UAMPERSAND
        '''
        p[0] = ExprUnaryOp(ty=None, op='&mut', rhs=p[3])

    def p_primary_expr_1(self, p):
        '''
        primary_expr : name
        '''
        ty = None
        for scope in reversed(self.scopes):
            if p[1] in scope:
                ty = scope[p[1]]
                break
        if ty is None and p[1] in PRIMS:
            ty = PRIMS[p[1]]
        p[0] = ExprName(ty=ty, name=p[1])

    def p_primary_expr_2(self, p):
        '''
        primary_expr : INTEGER
        '''
        # TODO: Need ambiguous integer type that we can resolve to that is not
        # illegal during type checking
        p[0] = ExprInteger(ty=p[1][1], val=p[1][0])

    def p_primary_expr_3(self, p):
        '''
        primary_expr : BOOL
        '''
        p[0] = ExprBool(ty=Bool, val=p[1])

    def p_primary_expr_5(self, p):
        '''
        primary_expr : NIL
        '''
        p[0] = ExprNil(ty=Nil)

    def p_primary_expr_6(self, p):
        '''
        primary_expr : STRING
        '''
        vals = []
        for b in cast(str, p[1]).encode():
            vals += [ExprInteger(ty=U8, val=b)]
        # a string is just a view over a constant array
        size = ExprInteger(ty=UInt, val=len(vals))
        lhs = ExprIndex(ty=U8, lhs=ExprArray(ty=TypeArray(ty=U8, size=size), vals=vals), rhs=ExprInteger(ty=UInt, val=0))
        ty = TypePointer(ty=U8, mut=False)
        p[0] = ExprSlice(ty=TypeSlice(ty), lhs=ExprUnaryOp(ty=ty, op='&', rhs=lhs), rhs=size)

    def p_primary_expr_7(self, p):
        '''
        primary_expr : FN PAREN_OPEN begin_scope arg_list PAREN_CLOSE begin_return_type stmt_block end_scope end_return_type
        '''
        # TODO: lambdas need a way to restrict their scope only to globals.
        # will probably create some scope flags or something that indicate that we need
        # to skip to the global scope if we are searching from a lambda scope
        if isinstance(p[6], TypeNil):
            p[7] += [StmtRet(line=p.lineno(1), val=ExprNil(ty=Nil))] # add implied return nil
        elif isinstance(p[6], TypeFallible) and isinstance(p[6].ty, TypeNil):
            p[7] += [StmtRet(line=p.lineno(1), val=ExprFallible(ty=p[6], expr=ExprNil(ty=Nil), ok=True))]
        p[0] = ExprFn(ty=None, args=p[4], rets=p[6], stmts=p[7])

    def p_init_list_1(self, p):
        '''
        init_list : empty
        '''
        p[0] = []

    def p_init_list_2(self, p):
        '''
        init_list : ID COLON expr
        '''
        p[0] = [(p[1], p[3])]

    def p_init_list_3(self, p):
        '''
        init_list : ID COLON expr COMMA init_list
        '''
        p[0] = [(p[1], p[3])] + p[5]

@dataclass
class Attr:
    items: Sequence[str]

@dataclass
class Pkg:
    items: Sequence['PkgItem']
    attrs: Sequence[Attr]

@dataclass
class PkgItem(ABC):
    line: int
    attrs: Sequence[Attr]

@dataclass
class PkgUse(PkgItem):
    name: str

@dataclass
class PkgBind(PkgItem):
    name: Name
    ty: Union[Type, Name]
    pub: bool
    val: Optional['Expr']
    mut: bool

@dataclass
class FnArg:
    name: str
    ty: Union[Type, Name]
    mut: bool

@dataclass
class Field:
    name: str
    ty: Union[Type, Name]
    attrs: Sequence[Attr]
    pub: bool

@dataclass
class PkgFn(PkgItem):
    name: Name
    ty: Union[Type, Name]
    pub: bool
    args: Sequence[FnArg]
    rets: Union[Type, Name]
    stmts: Sequence['Stmt']

@dataclass
class PkgType(PkgItem):
    name: Name
    ty: Union[Type, Name]
    pub: bool

@dataclass
class Stmt(ABC):
    line: int

@dataclass
class StmtBind(Stmt):
    name: str
    ty: Union[Type, Name]
    val: Optional['Expr']
    mut: bool

@dataclass
class StmtAssign(Stmt):
    lhs: 'Expr'
    op: str
    rhs: 'Expr'

@dataclass
class StmtRet(Stmt):
    val: 'Expr'

@dataclass
class StmtCont(Stmt):
    label: Optional[str]

@dataclass
class StmtBreak(Stmt):
    label: Optional[str]

@dataclass
class StmtBlock(Stmt):
    stmts: Sequence[Stmt]
    label: Optional[str]

@dataclass
class StmtIf(Stmt):
    expr: 'Expr'
    stmts: Sequence[Stmt]
    else_stmts: Sequence[Stmt]
    label: Optional[str]

@dataclass
class StmtIfLet(Stmt):
    taken: StmtBind
    stmts: Sequence[Stmt]
    not_taken: Optional[StmtBind]
    else_stmts: Sequence[Stmt]
    label: Optional[str]

@dataclass
class StmtTryLet(Stmt):
    bind: StmtBind
    else_bind: Optional[StmtBind]
    else_stmts: Sequence[Stmt]
    func_return_type: TypeFallible

@dataclass
class StmtFor(Stmt):
    expr: Optional['Expr']
    stmts: Sequence[Stmt]
    label: Optional[str]

@dataclass
class StmtForLet(Stmt):
    bind: StmtBind
    stmts: Sequence[Stmt]
    label: Optional[str]

@dataclass
class StmtCall(Stmt):
    expr: 'ExprCall'

@dataclass
class Expr(ABC):
    ty: Optional[Union[Type, Name]]

@dataclass
class ExprNil(Expr):
    ty: TypeNil

@dataclass
class ExprSizeof(Expr):
    ty: TypeInteger
    rhs: Union[Type, Name]

@dataclass
class ExprLengthof(Expr):
    ty: TypeInteger
    rhs: Expr

@dataclass
class ExprName(Expr):
    name: Name

@dataclass
class ExprCast(Expr):
    ty: Union[Type, Name]
    expr: Expr

@dataclass
class ExprFallible(Expr):
    expr: Expr
    ok: bool

@dataclass
class ExprIndex(Expr):
    lhs: Expr
    rhs: Expr

@dataclass
class ExprSlice(Expr):
    lhs: Expr
    rhs: Expr

@dataclass
class ExprAccess(Expr):
    lhs: Expr
    field: str

@dataclass
class ExprCall(Expr):
    lhs: Expr
    args: Sequence[Expr]

@dataclass
class ExprBinOp(Expr):
    lhs: Expr
    op: str
    rhs: Expr

@dataclass
class ExprUnaryOp(Expr):
    op: str
    rhs: Expr

@dataclass
class ExprInteger(Expr):
    val: int

@dataclass
class ExprBool(Expr):
    val: bool

@dataclass
class ExprStruct(Expr):
    ty: Type
    vals: Sequence[Tuple[str, Expr]]

@dataclass
class ExprUnion(Expr):
    ty: Type
    vals: Sequence[Tuple[str, Expr]]

@dataclass
class ExprArray(Expr):
    ty: Type
    vals: Sequence[Expr]

@dataclass
class ExprFn(Expr):
    args: Sequence[FnArg]
    rets: Union[Type, Name]
    stmts: Sequence['Stmt']

make_pkg = sys.argv[1] == '-p'
filename = sys.argv[2]
parser = Parser()
with open(filename) as f:
    pkg = parser.parse(f.read())

ARRAY_FORWARDS: List[str] = []
ARRAY_STRUCTS: List[str] = []
QPOINTER_FORWARDS: List[str] = []
QPOINTER_STRUCTS: List[str] = []
FALLIBLE_FORWARDS: List[str] = []
FALLIBLE_STRUCTS: List[str] = []
VIEW_FORWARDS: List[str] = []
VIEW_STRUCTS: List[str] = []
LAMBDA_FORWARDS: List[str] = []
LAMBDAS: List[str] = []

PUB_ARRAY_FORWARDS: List[str] = []
PUB_ARRAY_STRUCTS: List[str] = []
PUB_QPOINTER_FORWARDS: List[str] = []
PUB_QPOINTER_STRUCTS: List[str] = []
PUB_FALLIBLE_FORWARDS: List[str] = []
PUB_FALLIBLE_STRUCTS: List[str] = []
PUB_VIEW_FORWARDS: List[str] = []
PUB_VIEW_STRUCTS: List[str] = []

def mangle_type_name(ty: Union[Type, Name], pub: bool) -> str:
    if isinstance(ty, Name):
        return emit_name(ty)
    if isinstance(ty, TypeInteger):
        return ty.ffi
    if isinstance(ty, TypeBool):
        return ty.ffi
    if isinstance(ty, TypePointer):
        if ty.mut:
            return f'__mutptr{mangle_type_name(ty.ty, pub)}'
        return f'__ptr{mangle_type_name(ty.ty, pub)}'
    if isinstance(ty, TypeQPointer):
        QPOINTER_FORWARDS.append(f'struct __qptr{mangle_type_name(ty.ty, pub)};')
        QPOINTER_STRUCTS.append(f'struct __qptr{mangle_type_name(ty.ty, pub)} {{ {emit_type_or_name(ty.ty, pub)} __ptr; }};')
        if pub:
            PUB_QPOINTER_FORWARDS.append(f'struct __qptr{mangle_type_name(ty.ty, pub)};')
            PUB_QPOINTER_STRUCTS.append(f'struct __qptr{mangle_type_name(ty.ty, pub)} {{ {emit_type_or_name(ty.ty, pub)} __ptr; }};')
        return f'__qptr{mangle_type_name(ty.ty, pub)}'
    if isinstance(ty, TypeFallible):
        FALLIBLE_FORWARDS.append(f'struct __fal{mangle_type_name(ty.ty, pub)};')
        FALLIBLE_STRUCTS.append(f'struct __fal{mangle_type_name(ty.ty, pub)} {{ union {{ Int __err; {emit_type_or_name(ty.ty, pub)} __ok; }} __as; UInt __var; }};')
        if pub:
            PUB_FALLIBLE_FORWARDS.append(f'struct __fal{mangle_type_name(ty.ty, pub)};')
            PUB_FALLIBLE_STRUCTS.append(f'struct __fal{mangle_type_name(ty.ty, pub)} {{ union {{ Int __err; {emit_type_or_name(ty.ty, pub)} __ok; }} __as; UInt __var; }};')
        return f'__fal{mangle_type_name(ty.ty, pub)}'
    if isinstance(ty, TypeNil):
        return ty.ffi
    if isinstance(ty, TypeArray):
        size = emit_expr(ty.size, pub)
        # hack: strip suffix
        if isinstance(ty.size, ExprInteger) and size.endswith('U') or size.endswith('L'):
            size = size[:-1]
        ARRAY_FORWARDS.append(f'struct __arr{size}{mangle_type_name(ty.ty, pub)};')
        ARRAY_STRUCTS.append(f'struct __arr{size}{mangle_type_name(ty.ty, pub)} {{ {emit_type_or_name(ty.ty, pub)} __items[{size}]; }};')
        if pub:
            PUB_ARRAY_FORWARDS.append(f'struct __arr{size}{mangle_type_name(ty.ty, pub)};')
            PUB_ARRAY_STRUCTS.append(f'struct __arr{size}{mangle_type_name(ty.ty, pub)} {{ {emit_type_or_name(ty.ty, pub)} __items[{size}]; }};')
        return f'__arr{size}{mangle_type_name(ty.ty, pub)}'
    if isinstance(ty, TypeSlice):
        VIEW_FORWARDS.append(f'struct __view{mangle_type_name(ty.ty, pub)};')
        VIEW_STRUCTS.append(f'struct __view{mangle_type_name(ty.ty, pub)} {{ {emit_type_or_name(ty.ty, pub)} __items; UInt __length; }};')
        if pub:
            PUB_VIEW_FORWARDS.append(f'struct __view{mangle_type_name(ty.ty, pub)};')
            PUB_VIEW_STRUCTS.append(f'struct __view{mangle_type_name(ty.ty, pub)} {{ {emit_type_or_name(ty.ty, pub)} __items; UInt __length; }};')
        return f'__view{mangle_type_name(ty.ty, pub)}'
    return f'{ty}'

def emit_name(name: Name) -> str:
    if name.pkg == '*':
        return name.ident
    return f'_ZN{len(name.pkg)}{name.pkg}{len(name.ident)}{name.ident}E'

def emit_type(ty: Type, apply_visibility=False, hide_all=False) -> str:
    if isinstance(ty, TypeInteger):
        return ty.ffi
    if isinstance(ty, TypeBool):
        return ty.ffi
    if isinstance(ty, TypePointer):
        if ty.mut:
            return f'{emit_type_or_name(ty.ty, not (apply_visibility and hide_all))} *'
        return f'{emit_type_or_name(ty.ty, not (apply_visibility and hide_all))} const *'
    if isinstance(ty, TypeNil):
        return ty.ffi
    if isinstance(ty, TypeStruct):
        fields = []
        for field in ty.fields:
            if apply_visibility and (hide_all or not field.pub):
                fields += [f'{emit_type_and_name(field.ty, "__hidden_" + field.name, mut=True, pub=True)};']
            else:
                fields += [f'{emit_type_and_name(field.ty, field.name, mut=True, pub=True)};']
        return f'{{ {"".join(fields)} }}'
    if isinstance(ty, TypeUnion):
        fields = []
        for field in ty.fields:
            fields += [f'{emit_type_and_name(field.ty, field.name, mut=True, pub=True)};']
        return f'{{ {"".join(fields)} }}'
    return f'{ty}'

def emit_type_or_name(ty: Union[Type, Name], pub: bool) -> str:
    if isinstance(ty, Type):
        if isinstance(ty, (TypeArray, TypeQPointer, TypeSlice, TypeFallible)):
            return f'struct {mangle_type_name(ty, pub)}'
        return emit_type(ty)
    return emit_name(ty)

def emit_type_and_name(ty: Union[Type, Name], name: str, mut: bool, pub: bool) -> str:
    if isinstance(ty, TypeFn):
        args = []
        for arg in ty.args:
            args += [f'{emit_type_or_name(arg, pub)}']
        if mut:
            return f'{emit_type_or_name(ty.rets, pub)}(* {name})({",".join(args)})'
        return f'{emit_type_or_name(ty.rets, pub)}(* const {name})({",".join(args)})'
    if mut:
        return f'{emit_type_or_name(ty, pub)} {name}'
    return f'{emit_type_or_name(ty, pub)} const {name}'

def emit_expr(expr: Expr, pub: bool) -> str:
    if isinstance(expr, ExprNil):
        return '((Nil){})'
    if isinstance(expr, ExprBinOp):
        return f'({emit_expr(expr.lhs, pub)} {expr.op} {emit_expr(expr.rhs, pub)})'
    if isinstance(expr, ExprUnaryOp):
        # TODO: these cases may cause problems for unresolved names
        #if expr.op == '&mut' or expr.op == '&':
        #    return f'(({emit_type_or_name(cast(Union[Type, Name], expr.ty))}) (& {emit_expr(expr.rhs)}))'
        if expr.op == '&mut': # for now, emit one that doesn't do const verification :(
            return f'(& {emit_expr(expr.rhs, pub)})'
        return f'({expr.op} {emit_expr(expr.rhs, pub)})'
    if isinstance(expr, ExprName):
        return emit_name(expr.name)
    if isinstance(expr, ExprInteger):
        suffix = ''
        ty = expr.ty
        if isinstance(ty, TypeInteger):
            if ty.ffi == 'U64' or ty.ffi == 'UInt':
                suffix = 'U'
            if ty.ffi == 'I64' or ty.ffi == 'Int':
                suffix = 'L'
        return f'{expr.val}{suffix}'
    if isinstance(expr, ExprSlice):
        expr_type(expr)
        return f'((struct {mangle_type_name(cast(Union[Type, Name], expr.ty), pub)}){{ .__items = {emit_expr(expr.lhs, pub)}, .__length = {emit_expr(expr.rhs, pub)} }})'
    if isinstance(expr, ExprFallible):
        ty = expr.ty
        expr_type(expr)
        if expr.ok:
            return f'((struct {mangle_type_name(cast(Union[Type, Name], ty), pub)}){{ .__as = {{ .__ok = {emit_expr(expr.expr, pub)} }}, .__var = 1 }})'
        return f'((struct {mangle_type_name(cast(Union[Type, Name], ty), pub)}){{ .__as = {{ .__err = {emit_expr(expr.expr, pub)} }}, .__var = 0 }})'
    if isinstance(expr, ExprCast):
        ty = expr.ty
        if isinstance(ty, TypeQPointer):
            expr_type(expr)
            if isinstance(expr.expr.ty, TypeInteger) and expr.expr.ty.ffi == 'UInt':
                return f'((struct {mangle_type_name(cast(Union[Type, Name], ty), pub)}){{ .__ptr = ((U8 *) {emit_expr(expr.expr, pub)}) }})'
            return f'((struct {mangle_type_name(cast(Union[Type, Name], ty), pub)}){{ .__ptr = {emit_expr(expr.expr, pub)} }})'
        return f'(({emit_type_and_name(expr.ty, "", mut=False, pub=pub)}) {emit_expr(expr.expr, pub)})'
    if isinstance(expr, ExprSizeof):
        return f'(sizeof ({emit_type_or_name(expr.rhs, pub)}))'
    if isinstance(expr, ExprLengthof):
        return f'(({emit_expr(expr.rhs, pub)}).__length)'
    if isinstance(expr, ExprBool):
        if expr.val:
            return '1'
        else:
            return '0'
    if isinstance(expr, ExprCall):
        args = []
        for arg in expr.args:
            args += [emit_expr(arg, pub)]
        return f'({emit_expr(expr.lhs, pub)}({",".join(args)}))'
    if isinstance(expr, ExprArray):
        vals = []
        for val in expr.vals:
            vals += [emit_expr(val, pub)]
        return f'((struct {mangle_type_name(expr.ty, pub)}){{ .__items = {{ {",".join(vals)} }} }})'
    if isinstance(expr, ExprIndex):
        return f'(({emit_expr(expr.lhs, pub)}).__items[{emit_expr(expr.rhs, pub)}])'
    if isinstance(expr, ExprAccess):
        if isinstance(expr.lhs.ty, TypePointer): # auto deref!
            return f'({emit_expr(expr.lhs, pub)}->{expr.field})'
        return f'({emit_expr(expr.lhs, pub)}.{expr.field})'
    if isinstance(expr, ExprStruct):
        fields = []
        for field in expr.vals:
            fields += [f'.{field[0]} = {emit_expr(field[1], pub)}']
        return f'(({emit_type_or_name(expr.ty, pub)}){{ {",".join(fields)} }})'
    if isinstance(expr, ExprUnion):
        fields = []
        for field in expr.vals:
            fields += [f'.{field[0]} = {emit_expr(field[1], pub)}']
        return f'(({emit_type_or_name(expr.ty, pub)}){{ {",".join(fields)} }})'
    if isinstance(expr, ExprFn):
        args = []
        for arg in expr.args:
            args += [f'{emit_type_and_name(arg.ty, arg.name, arg.mut, pub)}']
        name = f'__lambda_{str(len(LAMBDAS))}'
        LAMBDA_FORWARDS.append(f'static inline {emit_type_or_name(expr.rets, pub)} {name}({",".join(args)});')
        LAMBDAS.append(f'static inline {emit_type_or_name(expr.rets, pub)} {name}({",".join(args)}) {{')
        for stmt in expr.stmts:
            for line in emit_stmt(stmt, 4):
                LAMBDAS.append(line)
        LAMBDAS.append('}')
        return name
    return f'{expr}'

def emit_stmt(stmt: Stmt, indent: int) -> Sequence[str]:
    global COUNTER
    COUNTER += 1
    pad = ' ' * indent
    prefix = pad + f'#line {stmt.line} "{filename}"\n' + pad
    output = ['']
    if isinstance(stmt, StmtRet):
        output += [prefix + f'return {emit_expr(stmt.val, False)};']
        return output
    if isinstance(stmt, StmtIf):
        output += [prefix + f'if ({emit_expr(stmt.expr, False)}) {{']
        for s in stmt.stmts:
            output += emit_stmt(s, indent + 4)
        output += [pad + '}']
        if len(stmt.else_stmts) > 0:
            output += [pad + 'else {']
            for s in stmt.else_stmts:
                output += emit_stmt(s, indent + 4)
            output += [pad + '}']
        if stmt.label is not None:
            output += [prefix + f'__label_break_{stmt.label}: (void)0;']
        return output
    if isinstance(stmt, StmtBlock):
        output += [prefix + '{']
        for s in stmt.stmts:
            output += emit_stmt(s, indent + 4)
        output += [pad + '}']
        if stmt.label is not None:
            output += [prefix + f'__label_break_{stmt.label}: (void)0;']
        return output
    if isinstance(stmt, StmtIfLet):
        output += [prefix + '{']
        val = cast(Expr, stmt.taken.val)
        fal = f'__fal{COUNTER}'
        if isinstance(stmt.taken.ty, TypePointer):
            output += [prefix + f'{emit_type_and_name(stmt.taken.ty, stmt.taken.name, stmt.taken.mut, False)} = {emit_expr(val, False)}.__ptr;']
            output += [prefix + f'if ({stmt.taken.name} != 0) {{']
        else:
            output += [prefix + f'{emit_type_and_name(TypeFallible(ty=stmt.taken.ty), fal, mut=False, pub=False)} = {emit_expr(val, False)};']
            output += [prefix + f'{emit_type_and_name(stmt.taken.ty, stmt.taken.name, stmt.taken.mut, False)} = {fal}.__as.__ok;']
            output += [prefix + f'if ({fal}.__var != 0) {{']
        for s in stmt.stmts:
            output += emit_stmt(s, indent + 4)
        output += [pad + '}']
        if len(stmt.else_stmts) > 0:
            output += [pad + 'else {']
            if stmt.not_taken is not None:
                output += [prefix + f'    {emit_type_and_name(stmt.not_taken.ty, stmt.not_taken.name, stmt.not_taken.mut, False)} = {fal}.__as.__err;']
            for s in stmt.else_stmts:
                output += emit_stmt(s, indent + 4)
            output += [pad + '}']
        if stmt.label is not None:
            output += [prefix + f'__label_break_{stmt.label}: (void)0;']
        output += [pad + '}']
        return output
    if isinstance(stmt, StmtTryLet):
        val = cast(Expr, stmt.bind.val)
        fal = f'__fal{COUNTER}'
        if isinstance(stmt.bind.ty, TypePointer):
            output += [prefix + f'{emit_type_and_name(stmt.bind.ty, stmt.bind.name, stmt.bind.mut, False)} = {emit_expr(val, False)}.__ptr;']
            output += [prefix + f'if ({stmt.bind.name} == 0) {{']
            if stmt.else_bind is not None:
                output += [prefix + f'    {emit_type_and_name(stmt.else_bind.ty, stmt.else_bind.name, stmt.else_bind.mut, False)} = {stmt.bind.name};']
        else:
            output += [prefix + f'{emit_type_and_name(TypeFallible(ty=stmt.bind.ty), fal, mut=False, pub=False)} = {emit_expr(val, False)};']
            output += [prefix + f'{emit_type_and_name(stmt.bind.ty, stmt.bind.name, stmt.bind.mut, False)} = {fal}.__as.__ok;']
            output += [prefix + f'if ({fal}.__var == 0) {{']
            if stmt.else_bind is not None:
                output += [prefix + f'    {emit_type_and_name(stmt.else_bind.ty, stmt.else_bind.name, stmt.else_bind.mut, False)} = {fal}.__as.__err;']
        if len(stmt.else_stmts) > 0:
            for s in stmt.else_stmts:
                output += emit_stmt(s, indent + 4)
        else:
            if isinstance(stmt.bind.ty, TypePointer):
                output += [prefix + f'    return {stmt.bind.name};']
            else:
                output += [prefix + f'    return ((struct {mangle_type_name(stmt.func_return_type, False)}){{ .__as = {{ .__err = {fal}.__as.__err }}, .__var = 0 }});']
        output += [pad + '}']
        return output
    if isinstance(stmt, StmtBreak):
        if stmt.label is not None:
            output += [prefix + f'goto __label_break_{stmt.label};']
        else:
            output += [prefix + 'break;']
        return output;
    if isinstance(stmt, StmtCont):
        if stmt.label is not None:
            output += [prefix + f'goto __label_continue_{stmt.label};']
        else:
            output += [prefix + 'continue;']
        return output;
    if isinstance(stmt, StmtBind):
        if stmt.val is not None:
            output += [prefix + f'{emit_type_and_name(stmt.ty, stmt.name, stmt.mut, False)} = {emit_expr(stmt.val, False)};']
        else:
            output += [prefix + f'{emit_type_and_name(stmt.ty, stmt.name, stmt.mut, False)};']
        return output
    if isinstance(stmt, StmtAssign):
        output += [prefix + f'{emit_expr(stmt.lhs, False)} {stmt.op} {emit_expr(stmt.rhs, False)};']
        return output
    if isinstance(stmt, StmtFor):
        if stmt.expr is not None:
            output += [prefix + f'while ({emit_expr(stmt.expr, False)}) {{']
        else:
            output += [prefix + 'while (1) {']
        for s in stmt.stmts:
            output += emit_stmt(s, indent + 4)
        if stmt.label is not None:
            output += [prefix + f'    __label_continue_{stmt.label}: (void)0;']
        output += [pad + '}']
        if stmt.label is not None:
            output += [prefix + f'__label_break_{stmt.label}: (void)0;']
        return output
    if isinstance(stmt, StmtForLet):
        output += [prefix + '{']
        output += [prefix + f'while (1) {{']
        if isinstance(stmt.bind.ty, TypePointer):
            output += [prefix + f'    {emit_type_and_name(stmt.bind.ty, stmt.bind.name, stmt.bind.mut, False)} = {emit_expr(cast(Expr, stmt.bind.val), False)}.__ptr;']
            output += [prefix + f'    if ({stmt.bind.name} == 0) {{ break; }}']
            for s in stmt.stmts:
                output += emit_stmt(s, indent + 4)
        else:
            eprint('you probably dont want to use for let with errs since we dont propagate them unless we add an else block')
            fal = f'__fal{COUNTER}'
            output += [prefix + f'    {emit_type_and_name(TypeFallible(ty=stmt.bind.ty), fal, mut=False, pub=False)} = {emit_expr(cast(Expr, stmt.bind.val), False)};']
            output += [prefix + f'    {emit_type_and_name(stmt.bind.ty, stmt.bind.name, stmt.bind.mut, False)} = {fal}.__as.__ok;']
            output += [prefix + f'    if ({fal}.__var != 0) {{']
            for s in stmt.stmts:
                output += emit_stmt(s, indent + 8)
            output += [pad + '    }']
        if stmt.label is not None:
            output += [prefix + f'    __label_continue_{stmt.label}: (void)0;']
        output += [pad + '}']
        if stmt.label is not None:
            output += [prefix + f'__label_break_{stmt.label}: (void)0;']
        output += [pad + '}']
        return output
    if isinstance(stmt, StmtCall):
        output += [prefix + f'{emit_expr(stmt.expr, False)};']
        return output
    output += [f'{stmt}']
    return output

output: List[str] = []
forward_fns: List[str] = []
forward_structs: List[str] = []
structs: List[str] = []
typedefs: List[str] = []
pub_forward_fns: List[str] = []
pub_typedefs: List[str] = []
pub_forward_structs: List[str] = []
pub_structs: List[str] = []

for item in pkg.items:
    output += [f'#line {item.line} "{filename}"']
    if isinstance(item, PkgFn):
        args = []
        for arg in item.args:
            args += [f'{emit_type_and_name(arg.ty, arg.name, arg.mut, item.pub)}']
        if item.pub:
            forward_fns += [f'{emit_type_or_name(item.rets, item.pub)} {emit_name(item.name)}({",".join(args)});']
            pub_forward_fns += [f'extern {emit_type_or_name(item.rets, item.pub)} {emit_name(item.name)}({",".join(args)});']
            output += [f'{emit_type_or_name(item.rets, item.pub)} {emit_name(item.name)}({",".join(args)}) {{']
        else:
            if any(map(lambda attr: attr.items[0] == 'inline', item.attrs)):
                forward_fns += [f'static inline {emit_type_or_name(item.rets, item.pub)} {emit_name(item.name)}({",".join(args)});']
                output += [f'static inline {emit_type_or_name(item.rets, item.pub)} {emit_name(item.name)}({",".join(args)}) {{']
            else:
                forward_fns += [f'static {emit_type_or_name(item.rets, item.pub)} {emit_name(item.name)}({",".join(args)});']
                output += [f'static {emit_type_or_name(item.rets, item.pub)} {emit_name(item.name)}({",".join(args)}) {{']
        for stmt in item.stmts:
            output += emit_stmt(stmt, 4)
        output += ['}']
        continue
    if isinstance(item, PkgType):
        if isinstance(item.ty, TypeStruct):
            forward_structs += [f'struct {emit_name(item.name)};']
            typedefs += [f'typedef struct {emit_name(item.name)} {emit_name(item.name)};']
            structs += [f'struct {emit_name(item.name)} {emit_type(cast(Type, item.ty))};']
            # we always export structs, but hide all of their fields if the whole type is private
            # this allows public structs to refer to private ones
            # TODO: in the real compiler we'd just add padding instead
            pub_forward_structs += [f'struct {emit_name(item.name)};']
            pub_typedefs += [f'typedef struct {emit_name(item.name)} {emit_name(item.name)};']
            pub_structs += [f'struct {emit_name(item.name)} {emit_type(cast(Type, item.ty), apply_visibility=True, hide_all=not item.pub)};']
        elif isinstance(item.ty, TypeUnion):
            forward_structs += [f'union {emit_name(item.name)};']
            typedefs += [f'typedef union {emit_name(item.name)} {emit_name(item.name)};']
            structs += [f'union {emit_name(item.name)} {emit_type(cast(Type, item.ty))};']
            pub_forward_structs += [f'union {emit_name(item.name)};']
            pub_typedefs += [f'typedef union {emit_name(item.name)} {emit_name(item.name)};']
            pub_structs += [f'union {emit_name(item.name)} {emit_type(cast(Type, item.ty), apply_visibility=True, hide_all=not item.pub)};']
        elif isinstance(item.ty, TypeFn):
            typedefs += [f'typedef {emit_type_and_name(item.ty, emit_name(item.name), True, item.pub)};']
            if item.pub:
                pub_typedefs += [f'typedef {emit_type_and_name(item.ty, emit_name(item.name), True, item.pub)};']
        else:
            typedefs += [f'typedef {emit_type_and_name(item.ty, emit_name(item.name), True, item.pub)};']
            if item.pub:
                pub_typedefs += [f'typedef {emit_type_and_name(item.ty, emit_name(item.name), True, item.pub)};']
        continue
    if isinstance(item, PkgBind):
        if item.pub:
            if item.mut or item.val is None:
                forward_fns += [f'{emit_type_and_name(item.ty, emit_name(item.name), item.mut, item.pub)};']
                pub_forward_fns += [f'extern {emit_type_and_name(item.ty, emit_name(item.name), item.mut, item.pub)};']
                if item.val is not None:
                    output += [f'{emit_type_and_name(item.ty, emit_name(item.name), item.mut, item.pub)} = {emit_expr(item.val, item.pub)};']
            # TODO: maybe should make dedicated pub const syntax.
            # This is me attempting to optimize the output of shared constants.
            # another approach I tried was exporting statics, but they create cycles.
            # the better compiler could detect all transitive imports
            # during the package building phase. *.pkg should really
            # be chl files with externs. and transitive externs.
            # or really they need to be structured data with forwards and
            # typedefs in separate sections.
            # we should export constants with their literal values in pkg files
            # and allow expression evaluation for constants at compile time as the true solution.
            #elif item.val is not None:
            #    forward_fns += [f'static {emit_type_and_name(item.ty, emit_name(item.name), item.mut)};']
            #    pub_forward_fns += [f'static {emit_type_and_name(item.ty, emit_name(item.name), item.mut)} = {emit_expr(item.val)};']
            #    output += [f'static {emit_type_and_name(item.ty, emit_name(item.name), item.mut)} = {emit_expr(item.val)};']
            #elif item.val is not None:
                # this is naive. but we assume these sorts of constants are simple values or names
                #if isinstance(item.val, ExprName): # constants that are names
                #    pub_forward_fns += [f'extern {emit_type_or_name(item.ty)} {emit_expr(item.val)};']
                #    pub_forward_fns += [f'#define {emit_name(item.name)} ({emit_expr(item.val)})']
                #else: # constants that are hopefully just expressions
                #pub_forward_fns += [f'#define {emit_name(item.name)} (({emit_type_or_name(item.ty)}) {emit_expr(item.val)})']
                #forward_fns += [f'{emit_type_and_name(item.ty, emit_name(item.name), item.mut)};']
                #output += [f'{emit_type_and_name(item.ty, emit_name(item.name), item.mut)} = {emit_expr(item.val)};']
            else: # this always "works" but prevents constants from referring to other constants
                forward_fns += [f'{emit_type_and_name(item.ty, emit_name(item.name), item.mut, item.pub)};']
                pub_forward_fns += [f'extern {emit_type_and_name(item.ty, emit_name(item.name), item.mut, item.pub)};']
                output += [f'{emit_type_and_name(item.ty, emit_name(item.name), item.mut, item.pub)} = {emit_expr(item.val, item.pub)};']
        else:
            forward_fns += [f'static {emit_type_and_name(item.ty, emit_name(item.name), item.mut, item.pub)};']
            if item.val is not None:
                output += [f'static {emit_type_and_name(item.ty, emit_name(item.name), item.mut, item.pub)} = {emit_expr(item.val, item.pub)};']
        continue
    if isinstance(item, PkgUse) and not make_pkg:
        with open(os.path.dirname(filename) +'/' + item.name + '.pkg') as f:
            pkgfile = json.load(fp=f)
            QPOINTER_FORWARDS.extend(pkgfile['forward_qptrs'])
            VIEW_FORWARDS.extend(pkgfile['forward_views'])
            FALLIBLE_FORWARDS.extend(pkgfile['forward_fallible'])
            ARRAY_FORWARDS.extend(pkgfile['forward_arrays'])
            QPOINTER_STRUCTS.extend(pkgfile['qptr_structs'])
            VIEW_STRUCTS.extend(pkgfile['view_structs'])
            FALLIBLE_STRUCTS.extend(pkgfile['fallible_structs'])
            ARRAY_STRUCTS.extend(pkgfile['array_structs'])
            forward_structs += pkgfile['forward_structs']
            typedefs += pkgfile['typedefs']
            structs += pkgfile['structs']
            forward_fns += pkgfile['forward_fns']
        continue

if not make_pkg:
    # because the compiler is naive right now we sometimes need to resolve
    # declarations of structs in a particular order. We do with via HACK attributes
    final_structs = list(dict.fromkeys(ARRAY_STRUCTS)) + list(dict.fromkeys(QPOINTER_STRUCTS)) + list(dict.fromkeys(VIEW_STRUCTS)) + list(dict.fromkeys(FALLIBLE_STRUCTS)) + structs

    for attr in pkg.attrs:
        if attr.items[0] == 'HACK_typeorder':
            above = 'struct ' + attr.items[1]
            below = 'struct ' + attr.items[2]
            removed: Optional[Tuple[int, str]] = None
            for (i, struct) in enumerate(final_structs):
                if struct.startswith(above):
                    removed = (i, final_structs.pop(i))
                    break
            if removed is None:
                raise AssertionError(f'{above} was not found')
            for (i, struct) in enumerate(final_structs):
                if struct.startswith(below):
                    if removed[0] <= i:
                        raise AssertionError(f'{above} is already above {below}')
                    final_structs.insert(i, removed[1])
                    break

    with open(os.path.splitext(filename)[0] + '.c', 'w') as f:
        print('typedef struct{} Nil;', file=f)
        print('typedef unsigned char U8;', file=f)
        print('typedef signed char I8;', file=f)
        print('typedef unsigned short U16;', file=f)
        print('typedef signed short I16;', file=f)
        print('typedef unsigned int U32;', file=f)
        print('typedef signed int I32;', file=f)
        print('typedef unsigned long int U64;', file=f)
        print('typedef signed long int I64;', file=f)
        print('typedef U64 UInt;', file=f)
        print('typedef I64 Int;', file=f)
        print('typedef U8 Bool;', file=f)
        print('extern U8 _ZN2U813read_volatileE(U8 const * const);', file=f)
        print('extern Nil _ZN2U814write_volatileE(U8 * const, U8 const);', file=f)
        for line in forward_structs:
            print(line, file=f)

        for line in list(dict.fromkeys(ARRAY_FORWARDS)):
            print(line, file=f)

        for line in list(dict.fromkeys(QPOINTER_FORWARDS)):
            print(line, file=f)

        for line in list(dict.fromkeys(VIEW_FORWARDS)):
            print(line, file=f)

        for line in list(dict.fromkeys(FALLIBLE_FORWARDS)):
            print(line, file=f)

        for line in typedefs:
            print(line, file=f)

        for line in final_structs:
            print(line, file=f)

        for line in list(dict.fromkeys(LAMBDA_FORWARDS)):
            print(line, file=f)

        for line in forward_fns:
            print(line, file=f)

        for line in LAMBDAS:
            print(line, file=f)

        for line in output:
            print(line, file=f)

if make_pkg:
    pkgfile = {
        'forward_qptrs': list(dict.fromkeys(PUB_QPOINTER_FORWARDS)),
        'forward_views': list(dict.fromkeys(PUB_VIEW_FORWARDS)),
        'forward_fallible': list(dict.fromkeys(PUB_FALLIBLE_FORWARDS)),
        'forward_arrays': list(dict.fromkeys(PUB_ARRAY_FORWARDS)),
        'forward_structs': pub_forward_structs,
        'typedefs': pub_typedefs,
        'qptr_structs': list(dict.fromkeys(PUB_QPOINTER_STRUCTS)),
        'view_structs': list(dict.fromkeys(PUB_VIEW_STRUCTS)),
        'fallible_structs': list(dict.fromkeys(PUB_FALLIBLE_STRUCTS)),
        'array_structs': list(dict.fromkeys(PUB_ARRAY_STRUCTS)),
        'structs': pub_structs,
        'forward_fns': pub_forward_fns
    }
    with open(os.path.splitext(filename)[0] + '.pkg', 'w') as f:
        json.dump(pkgfile, fp=f)
