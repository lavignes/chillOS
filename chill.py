#!/usr/bin/env python3

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
import functools
import sys
from typing import Mapping, cast

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
    ty: Type
    mut: bool

@dataclass
class TypeStruct(Type):
    fields: Sequence['Field']

class TypeFn(Type):
    args: Sequence[Type]
    rets: Type

TYPES: Mapping[Name, Type] = {
    Name('*', '()'): TypeStruct(fields=[]),
    Name('*', 'Bool'): TypeInteger(ffi='uint8_t'),
    Name('*', 'U8'): TypeInteger(ffi='uint8_t'),
    Name('*', 'I8'): TypeInteger(ffi='int8_t'),
    Name('*', 'U16'): TypeInteger(ffi='uint16_t'),
    Name('*', 'I16'): TypeInteger(ffi='int16_t'),
    Name('*', 'U32'): TypeInteger(ffi='uint32_t'),
    Name('*', 'I32'): TypeInteger(ffi='int32_t'),
    Name('*', 'U64'): TypeInteger(ffi='uint64_t'),
    Name('*', 'I64'): TypeInteger(ffi='int64_t'),
    Name('*', 'UInt'): TypeInteger(ffi='size_t'),
    Name('*', 'Int'): TypeInteger(ffi='ssize_t'),
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
            'const': 'CONST',
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
            'NOT_EQUAL']

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

    t_ignore = ' \r\t'

    def t_comment(self, t):
        r'//.*\n'
        self.inner.lineno += len(t.value)

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
        r"(\d+)|(0x[0-9a-fA-F]+)|(0b[01]+)|('((\\x[0-9a-fA-F]{2})|[ -~])')"
        if cast(str, t.value).startswith('0x'):
            t.value = int(t.value[2:], base=16)
        elif cast(str, t.value).startswith('0b'):
            t.value = int(t.value[2:], base=2)
        elif cast(str, t.value).startswith("'"):
            t.value = t.value[1:-1]
            if cast(str, t.value).startswith('\\x'):
                t.value = int(t.value[2:], base=16)
            else:
                t.value = ord(t.value)
        else:
            t.value = int(t.value)
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
        self.inner = yacc.yacc(module=self, write_tables=False)

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
        ('right', 'UMINUS', 'BANG', 'USTAR', 'UAMPERSAND', 'SIZEOF')
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
        pkg_item : PUB pkg_const
                 | PUB pkg_fn
                 | PUB pkg_type
        '''
        p[2].public = True
        p[0] = p[2]

    def p_pkg_item_2(self, p):
        '''
        pkg_item : pkg_const
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

    def p_pkg_const(self, p):
        '''
        pkg_const : CONST ID COLON type EQUAL expr SEMI
        '''
        p[0] = PkgConst(name=Name(self.pkg_name, p[2]), ty=p[4], pub=False, val=p[6], attrs=[])

    def p_pkg_fn_1(self, p):
        '''
        pkg_fn : FN ID PAREN_OPEN arg_list PAREN_CLOSE stmt_block
        '''
        name = Name(self.pkg_name, p[2])
        p[0] = PkgFn(name=name, ty=name, pub=False, attrs=[], args=p[4], rets=TYPES[Name('*', '()')], stmts=p[6])

    def p_pkg_fn_2(self, p):
        '''
        pkg_fn : FN ID PAREN_OPEN arg_list PAREN_CLOSE COLON type stmt_block
        '''
        name = Name(self.pkg_name, p[2])
        p[0] = PkgFn(name=name, ty=name, pub=False, attrs=[], args=p[4], rets=p[7], stmts=p[8])

    def p_arg_list_1(self, p):
        '''
        arg_list : empty
        '''
        p[0] = []

    def p_arg_list_2(self, p):
        '''
        arg_list : ID COLON type
        '''
        p[0] = [FnArg(name=p[1], ty=p[3], mut=False)]

    def p_arg_list_3(self, p):
        '''
        arg_list : ID COLON type COMMA arg_list
        '''
        p[0] = [FnArg(name=p[1], ty=p[3], mut=False)] + p[5]

    def p_arg_list_4(self, p):
        '''
        arg_list : MUT ID COLON type
        '''
        p[0] = [FnArg(name=p[2], ty=p[4], mut=True)]

    def p_arg_list_5(self, p):
        '''
        arg_list : MUT ID COLON type COMMA arg_list
        '''
        p[0] = [FnArg(name=p[2], ty=p[4], mut=True)] + p[6]

    def p_pkg_type(self, p):
        '''
        pkg_type : TYPE ID EQUAL type SEMI
        '''
        p[0] = PkgType(name=Name(self.pkg_name, p[2]), ty=p[4], pub=False, attrs=[])

    def p_type_1(self, p):
        '''
        type : name
        '''
        p[0] = TYPES.get(p[1], p[1])

    def p_type_2(self, p):
        '''
        type : BRACE_OPEN field_list BRACE_CLOSE
        '''
        p[0] = TypeStruct(fields=p[2])

    def p_name_1(self, p):
        '''
        name : ID
        '''
        if Name('*', p[1]) in TYPES:
            p[0] = Name('*', p[1])
            return
        p[0] = Name(self.pkg_name, p[1])

    def p_name_2(self, p):
        '''
        name : ID DOUBLE_COLON ID
        '''
        p[0] = Name(p[1], p[2])

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
        stmt_block : BRACE_OPEN stmt_list BRACE_CLOSE
        '''
        p[0] = p[2]

    def p_stmt_list_1(self, p):
        '''
        stmt_list : empty
        '''
        p[0] = []

    def p_stmt_list_2(self, p):
        '''
        stmt_list : stmt_list stmt
        '''
        p[0] = p[1] + [p[2]]

    def p_stmt_1(self, p):
        '''
        stmt : LET ID COLON type EQUAL expr SEMI
        '''
        p[0] = StmtAssign(name=p[2], val=p[5], mut=False)

    def p_stmt_2(self, p):
        '''
        stmt : LET MUT ID COLON type EQUAL expr SEMI
        '''
        p[0] = StmtAssign(name=p[2], val=p[5], mut=True)

    def p_stmt_3(self, p):
        '''
        stmt : RETURN SEMI
        '''
        p[0] = StmtRet(val=ExprUnit(ty=TYPES[Name('*', '()')]))

    def p_stmt_4(self, p):
        '''
        stmt : RETURN expr SEMI
        '''
        p[0] = StmtRet(val=p[1])

    def p_stmt_5(self, p):
        '''
        stmt : CONTINUE SEMI
             | CONTINUE DOUBLE_COLON ID SEMI
             | BREAK SEMI
             | BREAK DOUBLE_COLON ID SEMI
             | if_stmt
             | FOR stmt_block
             | FOR DOUBLE_COLON ID stmt_block
             | FOR expr stmt_block
             | FOR DOUBLE_COLON ID expr stmt_block
        '''

    def p_if_stmt_1(self, p):
        '''
        if_stmt : IF expr stmt_block
                | IF expr stmt_block ELSE stmt_block
                | IF expr stmt_block ELSE if_stmt
        '''

    def p_if_stmt_2(self, p):
        '''
        if_stmt : IF DOUBLE_COLON ID expr stmt_block
                | IF DOUBLE_COLON ID expr stmt_block ELSE stmt_block
                | IF DOUBLE_COLON ID expr stmt_block ELSE if_stmt
        '''

    def p_expr_1(self, p):
        '''
        expr : binary_expr
             | unary_expr
             | index_expr
             | call_expr
             | cast_expr
             | field_expr
        '''
        p[0] = p[1]

    def p_expr_2(self, p):
        '''
        expr : PAREN_OPEN expr PAREN_CLOSE
        '''
        p[0] = p[2]

    def p_expr_3(self, p):
        '''
        expr : name
        '''
        p[0] = p[1]

    def p_expr_4(self, p):
        '''
        expr : INTEGER
             | BOOL
             | STRING
             | PAREN_OPEN PAREN_CLOSE
        '''
        p[0] = p[1]

    def p_expr_5(self, p):
        '''
        expr : BRACKET_OPEN expr_list BRACKET_CLOSE
        '''
        p[0] = p[2]

    def p_index_expr(self, p):
        '''
        index_expr : expr BRACE_OPEN expr BRACE_CLOSE
        '''

    def p_field_expr(self, p):
        '''
        field_expr : expr DOT ID
        '''

    def p_call_expr_1(self, p):
        '''
        call_expr : expr PAREN_OPEN expr_list PAREN_CLOSE
        '''

    def p_expr_list_1(self, p):
        '''
        expr_list : empty
        '''

    def p_expr_list_2(self, p):
        '''
        expr_list : expr
        '''

    def p_expr_list_3(self, p):
        '''
        expr_list : expr COMMA expr_list
        '''

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

    def p_cast_expr(self, p):
        '''
        cast_expr : expr AS type
        '''

    def p_unary_expression(self, p):
        '''
        unary_expr : MINUS expr %prec UMINUS
                   | AMPERSAND expr %prec UAMPERSAND
                   | AMPERSAND MUT expr %prec UAMPERSAND
                   | STAR expr %prec USTAR
                   | BANG expr
                   | SIZEOF type
        '''

    def p_error(self, p):
        eprint(f'ERROR: {p}')

@dataclass
class Pkg:
    items: Sequence['PkgItem']

@dataclass
class PkgItem(ABC):
    name: Name
    ty: Type | Name
    pub: bool
    attrs: Sequence[str]

@dataclass
class PkgConst(PkgItem):
    val: 'Expr'

@dataclass
class FnArg:
    name: str
    ty: Type | Name
    mut: bool

@dataclass
class Field:
    name: str
    ty: Type | Name
    pub: bool

@dataclass
class PkgFn(PkgItem):
    args: Sequence[FnArg]
    rets: Type | Name
    stmts: Sequence['Stmt']

@dataclass
class PkgType(PkgItem):
    pass

@dataclass
class Stmt(ABC):
    pass

@dataclass
class StmtAssign(Stmt):
    name: str
    val: 'Expr'
    mut: bool

@dataclass
class StmtRet(ABC):
    val: 'Expr'

@dataclass
class Expr(ABC):
    ty: Type | Name

class ExprUnit(Expr):
    pass

parser = Parser()
pkg = parser.parse('''
        pkg test;

        // this is a test
        [no_mangle]
        pub fn _start(mut test: U8) {
            return;
        }
''')

eprint(f'{pkg}')


