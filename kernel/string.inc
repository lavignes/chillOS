# vim: ft=riscv sw=8 ts=8 cc=80 noet
.ifndef STRING_INC
.equ STRING_INC, 1

.macro STRING_DEFINE label:req, value:req
1:	.ascii "\value"
\label\():
	.dword 1b, \label - 1b
	.align 4
.endm

.endif # STRING_INC
