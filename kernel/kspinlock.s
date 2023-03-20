.global kspinlock_aquire
kspinlock_aquire:
    li t0, 1
1:
    amoswap.d.aq t0, t0, (a0)
    bnez t0, 1b
    ret

.global kspinlock_release
kspinlock_release:
    amoswap.d.rl zero, zero, (a0)
    ret
