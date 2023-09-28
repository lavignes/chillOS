ENTRY(_Kentry);

. = 0x80000000;

SECTIONS {
    .text : ALIGN(2) {
        *(.boot);
        *(.text);
        *(.rodata);

        . = ALIGN(16);
        . += 1M;
        PROVIDE(_KSTACK_TOP = .);
    }
}
