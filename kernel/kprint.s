.include "asm-common.inc"

.global kprint
kprint:
    pushd ra

    popd ra
    ret
