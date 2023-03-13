.macro pushd reg:req
    sd \reg, 0(sp)
    addi sp, sp, -8
.endm

.macro popd reg:req
    addi sp, sp, 8
    ld \reg, 0(sp)
.endm

.macro pushw reg:req
    sw \reg, 0(sp)
    addi sp, sp, -4
.endm

.macro popw reg:req
    addi sp, sp, 4
    lwu \reg, 0(sp)
.endm

.macro pushh reg:req
    sh \reg, 0(sp)
    addi sp, sp, -2
.endm

.macro poph reg:req
    addi sp, sp, 2
    lhu \reg, 0(sp)
.endm

.macro pushb reg:req
    sb \reg, 0(sp)
    addi sp, sp, -1
.endm

.macro popb reg:req
    addi sp, sp, 1
    lbu \reg 0(sp)
.endm

.macro STRING_DEFINE label:req, value:req
1:
    .ascii "\value"
\label\():
    .dword 1b, \label - 1b
    .align 4
.endm
