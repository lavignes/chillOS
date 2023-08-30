#!/usr/bin/env python3

from abc import ABC
from dataclasses import dataclass
import functools
import os
import sys
from typing import Mapping, Optional, Set, Tuple, Union, Sequence, cast

import ply.lex as lex
import ply.yacc as yacc

eprint = functools.partial(print, file=sys.stderr)

@dataclass(frozen=True)
class Name:
    pkg: str
    ident: str

@dataclass
class Type(ABC):
    pass

@dataclass
class TypeInteger(Type):
    ffi: str

@dataclass
class TypePointer(Type):
    ty: Union[Type, Name]
    mut: bool

@dataclass
class TypeArray(Type):
    ty: Union[Type, Name]
    size: 'Expr'

@dataclass
class TypeStruct(Type):
    fields: Sequence['Field']

@dataclass
class TypeFn(Type):
    args: Sequence[Union[Type, Name]]
    rets: Union[Type, Name]

@dataclass
class TypeBool(Type):
    ffi: str

@dataclass
class TypeUnit(Type):
    ffi: str

@dataclass
class TypeAny(Type):
    ffi: str

Unit = TypeUnit(ffi='Unit')
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
Any = TypeAny(ffi='Any')

# very basic and buggy type checker and inference
# in the real compiler we need to make multiple passes
# over the AST to infer types correctly. this fails in lots
# of places besides basic integers and expressions
def expr_type(expr: 'Expr') -> Optional[Union[Type, Name]]:
    if expr.ty is not None:
        return expr.ty
    if isinstance(expr, ExprIndex):
        lhs = expr_type(expr.lhs)
        if isinstance(lhs, TypeArray):
            expr.ty = lhs.ty
            return lhs.ty
        if isinstance(lhs, TypePointer):
            expr.ty = lhs.ty
            return lhs.ty
        return lhs
    if isinstance(expr, ExprCast):
        lhs = expr_type(expr.expr)
        ty = expr.ty
        if lhs == ty:
            return lhs
        if isinstance(lhs, TypeInteger) and isinstance(ty, TypeInteger):
            return ty
        if isinstance(lhs, TypePointer) and isinstance(ty, TypePointer) and isinstance(ty.ty, TypeAny):
            return ty
        if isinstance(lhs, TypePointer) and isinstance(lhs.ty, TypeAny) and isinstance(ty, TypePointer):
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

PRIMS: Mapping[Name, Type] = {
    Name('*', '()'): Unit,
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
    Name('*', 'Any'): Any,
}

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
    }

    tokens = list(reserved.values()) + ['ID', 'INTEGER', 'BOOL', 'BRACE_OPEN', 'BRACE_CLOSE',
            'BRACKET_OPEN', 'BRACKET_CLOSE', 'PLUS', 'MINUS', 'STAR', 'SOLIDUS', 'BANG', 'CARET',
            'AMPERSAND', 'PAREN_OPEN', 'PAREN_CLOSE', 'EQUAL', 'DOUBLE_EQUAL', 'LESS_THAN',
            'GREATER_THAN', 'LESS_EQUAL', 'GREATER_EQUAL', 'PIPE', 'COLON', 'SEMI', 'MODULUS',
            'DOT', 'DOUBLE_COLON', 'COMMA', 'STRING', 'SHIFT_LEFT', 'SHIFT_RIGHT', 'AND', 'OR',
            'NOT_EQUAL', 'PLUS_EQUAL', 'MINUS_EQUAL', 'STAR_EQUAL', 'SOLIDUS_EQUAL',
            'MODULUS_EQUAL', 'SHIFT_LEFT_EQUAL', 'SHIFT_RIGHT_EQUAL', 'AND_EQUAL', 'CARET_EQUAL',
            'PIPE_EQUAL']

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

    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        t.type = self.reserved.get(t.value, 'ID')
        return t

    def t_INTEGER(self, t):
        r"(0x[0-9a-fA-F_]+)|(0b[01_]+)|(\d[\d_]*)|('((\\x[0-9a-fA-F]{2})|[ -~])')"
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
        return t

    def t_BOOL(self, t):
        r'true|false'
        if cast(str, t.value) == 'true':
            t.value = True
        else:
            t.value = False
        return t

    def t_STRING(self, t):
        r'"((\\x[0-9a-fA-F]{2})|[ -~])*"'
        t.value = t.value[1:-1]
        value = ''
        i = 0
        while i < len(t.value):
            if cast(str, t.value)[i:].startswith('\\x'):
                value += chr(int(t.value[i+2:i+4], base=16))
                i += 4
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
        self.aliases = dict()

    def parse(self, text: str) -> 'Pkg':
        lexer = Lexer()
        return self.inner.parse(text, lexer=lexer.inner)

    precedence = (
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
        ('right', 'UMINUS', 'BANG', 'USTAR', 'UAMPERSAND', 'SIZEOF'),
        ('left', 'PAREN_OPEN', 'PPAREN_OPEN', 'DOT', 'PBRACKET_OPEN')
    )

    def p_pkg(self, p):
        '''
        pkg : PKG ID set_pkg_name SEMI pkg_items
        '''
        p[0] = Pkg(p[5])

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
        attrs : BRACKET_OPEN ID id_list BRACKET_CLOSE
        '''
        p[0] = [p[2]] + p[3]

    def p_id_list_1(self, p):
        '''
        id_list : empty
        '''
        p[0] = []

    def p_id_list_2(self, p):
        '''
        id_list : id_list ID
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
        pkg_bind : LET ID COLON type EQUAL BRACE_OPEN field_list BRACE_CLOSE SEMI
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

    def p_pkg_fn_1(self, p):
        '''
        pkg_fn : FN ID PAREN_OPEN begin_scope arg_list PAREN_CLOSE stmt_block end_scope
        '''
        name = Name(self.pkg_name, p[2])
        self.scopes[-1][name] = name
        p[0] = PkgFn(line=p.lineno(1), name=name, ty=name, pub=False, attrs=[], args=p[5], rets=Unit, stmts=p[7])

    def p_pkg_fn_2(self, p):
        '''
        pkg_fn : FN ID PAREN_OPEN begin_scope arg_list PAREN_CLOSE COLON type stmt_block end_scope
        '''
        name = Name(self.pkg_name, p[2])
        self.scopes[-1][name] = name
        p[0] = PkgFn(line=p.lineno(1), name=name, ty=name, pub=False, attrs=[], args=p[5], rets=p[8], stmts=p[9])

    def p_begin_scope(self, p):
        '''
        begin_scope :
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
        pkg_type : TYPE ID EQUAL type SEMI
        '''
        name = Name(self.pkg_name, p[2])
        self.scopes[-1][name] = p[4]
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

    def p_type_4(self, p):
        '''
        type : BRACKET_OPEN type SEMI expr BRACKET_CLOSE
        '''
        p[0] = TypeArray(p[2], size=p[4])

    def p_type_7(self, p):
        '''
        type : BRACE_OPEN field_list BRACE_CLOSE
        '''
        p[0] = TypeStruct(fields=p[2])

    def p_type_8(self, p):
        '''
        type : PAREN_OPEN PAREN_CLOSE
        '''
        p[0] = Unit

    def p_type_9(self, p):
        '''
        type : FN PAREN_OPEN type_list PAREN_CLOSE
        '''
        p[0] = TypeFn(args=p[3], rets=Unit)

    def p_type_10(self, p):
        '''
        type : FN PAREN_OPEN type_list PAREN_CLOSE COLON type
        '''
        p[0] = TypeFn(args=p[3], rets=p[6])

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
        field_list : ID COLON type
        '''
        p[0] = [Field(name=p[1], ty=p[3], pub=False)]

    def p_field_list_3(self, p):
        '''
        field_list : ID COLON type COMMA field_list
        '''
        p[0] = [Field(name=p[1], ty=p[3], pub=False)] + p[5]

    def p_field_list_4(self, p):
        '''
        field_list : PUB ID COLON type
        '''
        p[0] = [Field(name=p[1], ty=p[3], pub=True)]

    def p_field_list_5(self, p):
        '''
        field_list : PUB ID COLON type COMMA field_list
        '''
        p[0] = [Field(name=p[1], ty=p[3], pub=True)] + p[5]

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
        p[0] = StmtRet(line=p.lineno(1), val=ExprUnit(ty=Unit))

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
             | for_stmt
             | expr_stmt
        '''
        p[0] = p[1]

    def p_expr_stmt_1(self, p):
        '''
        expr_stmt : expr EQUAL expr SEMI
                  | expr PLUS_EQUAL expr SEMI
                  | expr MINUS_EQUAL expr SEMI
                  | expr STAR_EQUAL expr SEMI
                  | expr SOLIDUS_EQUAL expr SEMI
                  | expr MODULUS_EQUAL expr SEMI
                  | expr SHIFT_LEFT_EQUAL expr SEMI
                  | expr SHIFT_RIGHT_EQUAL expr SEMI
                  | expr AND_EQUAL expr SEMI
                  | expr CARET_EQUAL expr SEMI
                  | expr PIPE_EQUAL expr SEMI
        '''
        p[0] = StmtAssign(line=p.lineno(2), lhs=p[1], op=p[2], rhs=p[3])

    def p_expr_stmt_2(self, p):
        '''
        expr_stmt : expr PAREN_OPEN expr_list PAREN_CLOSE SEMI %prec PPAREN_OPEN
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
        self.scopes[-1][Name('*', p[3])] = p[4]
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
        self.scopes[-1][Name('*', p[3])] = p[3]
        p[0] = StmtFor(line=p.lineno(1), expr=p[4], stmts=p[5], label=p[3])

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
        self.scopes[-1][Name('*', p[3])] = p[3]
        p[0] = StmtIf(line=p.lineno(1), expr=p[4], stmts=p[5], else_stmts=[], label=p[3])

    def p_if_stmt_5(self, p):
        '''
        if_stmt : IF DOUBLE_COLON ID expr stmt_block ELSE stmt_block
        '''
        self.scopes[-1][Name('*', p[3])] = p[3]
        p[0] = StmtIf(line=p.lineno(1), expr=p[4], stmts=p[5], else_stmts=p[7], label=p[3])

    def p_if_stmt_6(self, p):
        '''
        if_stmt : IF DOUBLE_COLON ID expr stmt_block ELSE if_stmt
        '''
        self.scopes[-1][Name('*', p[3])] = p[3]
        p[0] = StmtIf(line=p.lineno(1), expr=p[4], stmts=p[5], else_stmts=[p[7]], label=p[3])

    def p_expr_1(self, p):
        '''
        expr : binary_expr
             | unary_expr
             | index_expr
             | call_expr
             | cast_expr
             | field_expr
             | sizeof_expr
             | primary_expr
        '''
        p[0] = p[1]

    def p_expr_2(self, p):
        '''
        expr : PAREN_OPEN expr PAREN_CLOSE
        '''
        p[0] = p[2]

    def p_index_expr(self, p):
        '''
        index_expr : expr BRACKET_OPEN expr BRACKET_CLOSE %prec PBRACKET_OPEN
        '''
        p[0] = ExprIndex(ty=None, lhs=p[1], rhs=p[3])

    def p_field_expr(self, p):
        '''
        field_expr : expr DOT ID
        '''
        p[0] = ExprAccess(ty=None, lhs=p[1], field=p[3])

    def p_call_expr(self, p):
        '''
        call_expr : expr PAREN_OPEN expr_list PAREN_CLOSE %prec PPAREN_OPEN
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

    def p_cast_expr(self, p):
        '''
        cast_expr : expr AS type
        '''
        p[0] = ExprCast(ty=p[3], expr=p[1])

    def p_sizeof_expr(self, p):
        '''
        sizeof_expr : SIZEOF type
        '''
        p[0] = ExprSizeof(ty=UInt, rhs=p[2])

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
        p[0] = ExprName(ty=None, name=p[1])

    def p_primary_expr_2(self, p):
        '''
        primary_expr : INTEGER
        '''
        # TODO: Need ambiguous integer type that we can resolve to that is not
        # illegal during type checking
        p[0] = ExprInteger(ty=None, val=p[1])

    def p_primary_expr_3(self, p):
        '''
        primary_expr : BOOL
        '''
        p[0] = ExprBool(ty=Bool, val=p[1])

    def p_primary_expr_4(self, p):
        '''
        primary_expr : STRING
        '''
        vals = []
        for b in cast(str, p[1]).encode():
            vals += [ExprInteger(ty=U8, val=b)]
        p[0] = ExprArray(ty=TypeArray(ty=U8, size=ExprInteger(ty=UInt, val=len(vals))), vals=vals)

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
class Pkg:
    items: Sequence['PkgItem']

@dataclass
class PkgItem(ABC):
    line: int
    attrs: Sequence[str]

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
class StmtIf(Stmt):
    expr: 'Expr'
    stmts: Sequence[Stmt]
    else_stmts: Sequence[Stmt]
    label: Optional[str]

@dataclass
class StmtFor(Stmt):
    expr: Optional['Expr']
    stmts: Sequence[Stmt]
    label: Optional[str]

@dataclass
class StmtCall(Stmt):
    expr: 'ExprCall'

@dataclass
class Expr(ABC):
    ty: Optional[Union[Type, Name]]

@dataclass
class ExprUnit(Expr):
    ty: TypeUnit

@dataclass
class ExprSizeof(Expr):
    ty: TypeInteger
    rhs: Union[Type, Name]

@dataclass
class ExprName(Expr):
    name: Name

@dataclass
class ExprCast(Expr):
    ty: Union[Type, Name]
    expr: Expr

@dataclass
class ExprIndex(Expr):
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
class ExprArray(Expr):
    ty: Type
    vals: Sequence[Expr]

make_pkg = sys.argv[1] == '-p'
filename = sys.argv[2]
parser = Parser()
with open(filename) as f:
    pkg = parser.parse(f.read())

ARRAY_FORWARDS: Set[str] = set();
ARRAY_STRUCTS: Set[str] = set();

def mangle_type_name(ty: Union[Type, Name]) -> str:
    if isinstance(ty, Name):
        return emit_name(ty)
    if isinstance(ty, TypeInteger):
        return ty.ffi
    if isinstance(ty, TypeBool):
        return ty.ffi
    if isinstance(ty, TypeAny):
        return ty.ffi
    if isinstance(ty, TypePointer):
        if ty.mut:
            return f'__mutptr{mangle_type_name(ty.ty)}'
        return f'__ptr{mangle_type_name(ty.ty)}'
    if isinstance(ty, TypeUnit):
        return 'Unit'
    if isinstance(ty, TypeArray):
        ARRAY_FORWARDS.add(f'struct __arr{emit_expr(ty.size)}{mangle_type_name(ty.ty)};')
        ARRAY_STRUCTS.add(f'struct __arr{emit_expr(ty.size)}{mangle_type_name(ty.ty)} {{ {emit_type_or_name(ty.ty)} __items[{emit_expr(ty.size)}]; }};')
        return f'__arr{emit_expr(ty.size)}{mangle_type_name(ty.ty)}'
    return f'{ty}'

def emit_name(name: Name) -> str:
    if name.pkg == '*':
        return name.ident
    return f'_ZN{len(name.pkg)}{name.pkg}{len(name.ident)}{name.ident}E'

def emit_type(ty: Type) -> str:
    if isinstance(ty, TypeInteger):
        return ty.ffi
    if isinstance(ty, TypeBool):
        return ty.ffi
    if isinstance(ty, TypeAny):
        return ty.ffi
    if isinstance(ty, TypePointer):
        if ty.mut:
            return f'{emit_type_or_name(ty.ty)} *'
        return f'{emit_type_or_name(ty.ty)} const *'
    if isinstance(ty, TypeUnit):
        return 'Unit'
    if isinstance(ty, TypeStruct):
        fields = []
        for field in ty.fields:
            fields += [f'{emit_type_or_name(field.ty)} {field.name};']
        return f'{{ {"".join(fields)} }}'
    return f'{ty}'

def emit_type_or_name(ty: Union[Type, Name]) -> str:
    if isinstance(ty, Type):
        if isinstance(ty, TypeArray):
            return f'struct {mangle_type_name(ty)}'
        return emit_type(ty)
    return emit_name(ty)

def emit_type_and_name(ty: Union[Type, Name], name: str, mut: bool) -> str:
    if isinstance(ty, TypeFn):
        args = []
        for arg in ty.args:
            args += [f'{emit_type_or_name(arg)}']
        if mut:
            return f'{emit_type_or_name(ty.rets)}(* {name})({",".join(args)})'
        return f'{emit_type_or_name(ty.rets)}(* const {name})({",".join(args)})'
    if mut:
        return f'{emit_type_or_name(ty)} {name}'
    return f'{emit_type_or_name(ty)} const {name}'

def emit_expr(expr: Expr) -> str:
    # TODO: expr_type(expr)

    if isinstance(expr, ExprUnit):
        return '((Unit){})'
    if isinstance(expr, ExprBinOp):
        return f'({emit_expr(expr.lhs)} {expr.op} {emit_expr(expr.rhs)})'
    if isinstance(expr, ExprUnaryOp):
        # TODO: these cases may cause problems for unresolved names
        #if expr.op == '&mut' or expr.op == '&':
        #    return f'(({emit_type_or_name(cast(Union[Type, Name], expr.ty))}) (& {emit_expr(expr.rhs)}))'
        if expr.op == '&mut': # for now, emit one that doesn't do const verification :(
            return f'(& {emit_expr(expr.rhs)})'
        return f'({expr.op} {emit_expr(expr.rhs)})'
    if isinstance(expr, ExprName):
        return emit_name(expr.name)
    if isinstance(expr, ExprInteger):
        return f'{expr.val}'
    if isinstance(expr, ExprCast):
        return f'(({emit_type_or_name(expr.ty)}) {emit_expr(expr.expr)})'
    if isinstance(expr, ExprSizeof):
        return f'(sizeof ({emit_type_or_name(expr.rhs)}))'
    if isinstance(expr, ExprBool):
        if expr.val:
            return '1'
        else:
            return '0'
    if isinstance(expr, ExprCall):
        args = []
        for arg in expr.args:
            args += [emit_expr(arg)]
        return f'({emit_expr(expr.lhs)}({",".join(args)}))'
    if isinstance(expr, ExprArray):
        vals = []
        for val in expr.vals:
            vals += [emit_expr(val)]
        return f'((struct {mangle_type_name(expr.ty)}){{ .__items = {{ {",".join(vals)} }} }})'
    if isinstance(expr, ExprIndex):
        return f'(({emit_expr(expr.lhs)}).__items[{emit_expr(expr.rhs)}])'
    if isinstance(expr, ExprAccess):
        return f'({emit_expr(expr.lhs)}.{expr.field})'
    if isinstance(expr, ExprStruct):
        fields = []
        for field in expr.vals:
            fields += [f'.{field[0]} = {emit_expr(field[1])}']
        return f'(({emit_type_or_name(expr.ty)}){{ {",".join(fields)} }})'
    return f'{expr}'

def emit_stmt(stmt: Stmt, indent: int) -> Sequence[str]:
    pad = ' ' * indent
    output = [pad + f'#line {stmt.line} "{filename}"']
    if isinstance(stmt, StmtRet):
        output += [pad + f'return {emit_expr(stmt.val)};']
        return output
    if isinstance(stmt, StmtIf):
        output += [pad + f'if ({emit_expr(stmt.expr)}) {{']
        for s in stmt.stmts:
            output += emit_stmt(s, indent + 4)
        output += [pad + '}']
        if len(stmt.else_stmts) > 0:
            output += [pad + 'else {{']
            for s in stmt.else_stmts:
                output += emit_stmt(s, indent + 4)
            output += [pad + '}']
        if stmt.label is not None:
            output += [pad + f'__label_break_{stmt.label}: (void)0;']
        return output
    if isinstance(stmt, StmtBreak):
        if stmt.label is not None:
            output += [pad + f'goto __label_break_{stmt.label};']
        else:
            output += [pad + 'break;']
        return output;
    if isinstance(stmt, StmtCont):
        if stmt.label is not None:
            output += [pad + f'goto __label_continue_{stmt.label};']
        else:
            output += [pad + 'continue;']
        return output;
    if isinstance(stmt, StmtBind):
        if stmt.val is not None:
            output += [pad + f'{emit_type_and_name(stmt.ty, stmt.name, stmt.mut)} = {emit_expr(stmt.val)};']
        else:
            output += [pad + f'{emit_type_and_name(stmt.ty, stmt.name, stmt.mut)};']
        return output
    if isinstance(stmt, StmtAssign):
        output += [pad + f'{emit_expr(stmt.lhs)} {stmt.op} {emit_expr(stmt.rhs)};']
        return output
    if isinstance(stmt, StmtFor):
        if stmt.expr is not None:
            output += [pad + f'while ({emit_expr(stmt.expr)}) {{']
        else:
            output += [pad + 'while (1) {']
        for s in stmt.stmts:
            output += emit_stmt(s, indent + 4)
        if stmt.label is not None:
            output += [pad + f'    __label_continue_{stmt.label}: (void)0;']
        output += [pad + '}']
        if stmt.label is not None:
            output += [pad + f'__label_break_{stmt.label}: (void)0;']
        return output
    if isinstance(stmt, StmtCall):
        output += [pad + f'{emit_expr(stmt.expr)};']
        return output
    output += [f'{stmt}']
    return output

output = []
forwards = []
structs = []
typedefs = []
pub_forwards = []
pub_typedefs = []
pub_structs = []
imports = []

for item in pkg.items:
    output += [f'#line {item.line} "{filename}"']
    if isinstance(item, PkgFn):
        args = []
        for arg in item.args:
            if not arg.mut:
                args += [f'{emit_type_or_name(arg.ty)} const {arg.name}']
            else:
                args += [f'{emit_type_or_name(arg.ty)} {arg.name}']
        if item.pub:
            forwards += [f'{emit_type_or_name(item.rets)} {emit_name(item.name)}({",".join(args)});']
            pub_forwards += [f'extern {emit_type_or_name(item.rets)} {emit_name(item.name)}({",".join(args)});']
            output += [f'{emit_type_or_name(item.rets)} {emit_name(item.name)}({",".join(args)}) {{']
        else:
            forwards += [f'static {emit_type_or_name(item.rets)} {emit_name(item.name)}({",".join(args)});']
            output += [f'static {emit_type_or_name(item.rets)} {emit_name(item.name)}({",".join(args)}) {{']
        for stmt in item.stmts:
            output += emit_stmt(stmt, 4)
        output += ['}']
        continue
    if isinstance(item, PkgType):
        if isinstance(item.ty, TypeStruct):
            structs += [f'struct {emit_name(item.name)};']
            typedefs += [f'typedef struct {emit_name(item.name)} {emit_name(item.name)};']
            output += [f'struct {emit_name(item.name)} {emit_type(cast(Type, item.ty))};']
            if item.pub:
                pub_structs += [f'struct {emit_name(item.name)};']
                pub_typedefs += [f'typedef struct {emit_name(item.name)} {emit_name(item.name)};']
                pub_typedefs += [f'struct {emit_name(item.name)} {emit_type(cast(Type, item.ty))};']
        elif isinstance(item.ty, TypeFn):
            typedefs += [f'typedef {emit_type_and_name(item.ty, emit_name(item.name), True)};']
            if item.pub:
                pub_typedefs += [f'typedef {emit_type_and_name(item.ty, emit_name(item.name), True)};']
        else:
            typedefs += [f'typedef {emit_type_and_name(item.ty, emit_name(item.name), True)};']
            if item.pub:
                pub_typedefs += [f'typedef {emit_type_and_name(item.ty, emit_name(item.name), True)};']
        continue
    if isinstance(item, PkgBind):
        if item.pub:
            forwards += [f'{emit_type_and_name(item.ty, emit_name(item.name), item.mut)};']
            pub_forwards += [f'extern {emit_type_and_name(item.ty, emit_name(item.name), item.mut)};']
            if item.val is not None:
                output += [f'{emit_type_and_name(item.ty, emit_name(item.name), item.mut)} = {emit_expr(item.val)};']
        else:
            forwards += [f'static {emit_type_and_name(item.ty, emit_name(item.name), item.mut)};']
            if item.val is not None:
                output += [f'static {emit_type_and_name(item.ty, emit_name(item.name), item.mut)} = {emit_expr(item.val)};']
        continue
    if isinstance(item, PkgUse) and not make_pkg:
        with open(os.path.dirname(filename) +'/' + item.name + '.pkg') as f:
            imports += f.readlines()
        continue

if not make_pkg:
    with open(os.path.splitext(filename)[0] + '.c', 'w') as f:
        print('typedef struct{} Unit;', file=f)
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
        print('typedef void Any;', file=f)

        for line in structs:
            print(line, file=f)

        for line in ARRAY_FORWARDS:
            print(line, file=f)

        for line in ARRAY_STRUCTS:
            print(line, file=f)

        for line in typedefs:
            print(line, file=f)

        for line in imports:
            print(line, file=f)

        for line in forwards:
            print(line, file=f)

        for line in output:
            print(line, file=f)

if make_pkg:
    with open(os.path.splitext(filename)[0] + '.pkg', 'w') as f:
        for line in pub_structs:
            print(line, file=f)

        for line in pub_typedefs:
            print(line, file=f)

        for line in pub_forwards:
            print(line, file=f)

