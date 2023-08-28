ENTRY(_kentry);

. = 0x80000000;

SECTIONS {
	.text : ALIGN(4K) {
		*(.init);
		*(.text);

        . = ALIGN(16);
        . += 4K;
        PROVIDE(_KSTACK_TOP = .);
	}
}
