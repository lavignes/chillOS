.include "pushpop.inc"

.global kprint
kprint:
    pushd ra

    popd ra
    ret
