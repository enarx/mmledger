# mmledger

_mmledger_ provides means to intervene mm-syscalls `mmap()`, `brk()` and
`mprotect()` with a fixed-size data structure to store descriptions of virtual
memory areas (VMAs) created. The library is geared towards confidential
computing environments, such as Intel SGX and AMD SEV-SNP.
