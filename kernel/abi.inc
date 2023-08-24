# vim: ft=riscv sw=8 ts=8 cc=80 noet
.ifndef ABI_INC
.equ ABI_INC, 1

.macro ABI_PREAMBLE_RA_S0
	addi sp, sp, -16
	sd ra, 0(sp)
	sd s0, 8(sp)
.endm

.macro ABI_POSTAMBLE_RA_S0
	ld ra, 0(sp)
	ld s0, 8(sp)
    	addi sp, sp, 16
.endm

.macro ABI_PREAMBLE_RA_S1
	addi sp, sp, -24
	sd ra, 0(sp)
	sd s0, 8(sp)
	sd s1, 16(sp)
.endm

.macro ABI_POSTAMBLE_RA_S1
	ld ra, 0(sp)
	ld s0, 8(sp)
	ld s1, 16(sp)
    	addi sp, sp, 24
.endm

.macro ABI_PREAMBLE_RA_S2
	addi sp, sp, -32
	sd ra, 0(sp)
	sd s0, 8(sp)
	sd s1, 16(sp)
	sd s2, 24(sp)
.endm

.macro ABI_POSTAMBLE_RA_S2
	ld ra, 0(sp)
	ld s0, 8(sp)
	ld s1, 16(sp)
	ld s2, 24(sp)
    	addi sp, sp, 32
.endm

.macro ABI_PREAMBLE_RA_S3
	addi sp, sp, -40
	sd ra, 0(sp)
	sd s0, 8(sp)
	sd s1, 16(sp)
	sd s2, 24(sp)
	sd s3, 32(sp)
.endm

.macro ABI_POSTAMBLE_RA_S3
	ld ra, 0(sp)
	ld s0, 8(sp)
	ld s1, 16(sp)
	ld s2, 24(sp)
	ld s3, 32(sp)
    	addi sp, sp, 40
.endm

.macro ABI_PREAMBLE_RA_S4
	addi sp, sp, -48
	sd ra, 0(sp)
	sd s0, 8(sp)
	sd s1, 16(sp)
	sd s2, 24(sp)
	sd s3, 32(sp)
	sd s4, 40(sp)
.endm

.macro ABI_POSTAMBLE_RA_S4
	ld ra, 0(sp)
	ld s0, 8(sp)
	ld s1, 16(sp)
	ld s2, 24(sp)
	ld s3, 32(sp)
	ld s4, 40(sp)
    	addi sp, sp, 48
.endm

.endif # ABI_INC
