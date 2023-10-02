.section .text

.align 2
.global _ZN6Atomic12test_and_setE
_ZN6Atomic12test_and_setE:
    li t0, 1
    beqz a1, 1f
    amoswap.d.rl t0, t0, (a0)
    mv a0, t0
    ret
1:
    amoswap.d.aq t0, t0, (a0)
    mv a0, t0
	ret

.align 2
.global _ZN6Atomic5clearE
_ZN6Atomic5clearE:
    beqz a1, 1f
    amoswap.d.rl zero, zero, (a0)
    ret
1:
    amoswap.d.aq zero, zero, (a0)
	ret

