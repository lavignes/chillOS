ENTRY(_Kentry);

. = 0x80000000;

SECTIONS {
    .init : ALIGN(2) {
        *(.init);
    }
	.text : ALIGN(2) {
		*(.text);
        *(.rodata);

        . = ALIGN(16);
        . += 1M;
        PROVIDE(_KSTACK_TOP = .);
	}
}
