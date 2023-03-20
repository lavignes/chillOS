ENTRY(kstart);

. = 0x80000000;

SECTIONS {
	.text : ALIGN(4K) {
		*(.init);
		*(.text);

        . = ALIGN(16);
        . += 4K;
        PROVIDE(KMEM_STACK_TOP = .);
	}
}
