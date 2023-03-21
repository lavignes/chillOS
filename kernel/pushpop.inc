.ifndef PUSHPOP_INC
.equ PUSHPOP_INC, 1

.macro pushd reg:req
    sd \reg, 0(sp)
    addi sp, sp, -16
.endm

.macro popd reg:req
    addi sp, sp, 16
    ld \reg, 0(sp)
.endm

.macro pushw reg:req
    sw \reg, 0(sp)
    addi sp, sp, -16
.endm

.macro popw reg:req
    addi sp, sp, 16
    lwu \reg, 0(sp)
.endm

.macro pushh reg:req
    sh \reg, 0(sp)
    addi sp, sp, -16
.endm

.macro poph reg:req
    addi sp, sp, 16
    lhu \reg, 0(sp)
.endm

.macro pushb reg:req
    sb \reg, 0(sp)
    addi sp, sp, -16
.endm

.macro popb reg:req
    addi sp, sp, 16
    lbu \reg 0(sp)
.endm

.macro pushrd stack:req reg:req
    sd \reg, 0(\stack)
    addi \stack, \stack, -16
.endm

.macro poprd stack:req reg:req
    addi \stack, \stack, 16
    ld \reg, 0(\stack)
.endm

.endif # PUSHPOP_INC
