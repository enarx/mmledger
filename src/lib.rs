// SPDX-License-Identifier: Apache-2.0

//! A ledger for memory mappings.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use core::fmt::{Debug, Formatter};
use core::ops::BitAndAssign;

use const_default::ConstDefault;
use lset::Contains;
use primordial::{Address, Offset, Page};

/// A region of memory.
pub type Region = lset::Line<Address<usize, Page>>;
/// A span of memory.
pub type Span = lset::Span<Address<usize, Page>, Offset<usize, Page>>;

/// An access type for a region of memory.
pub trait LedgerAccess: Sized + ConstDefault + Default + Eq + BitAndAssign + Copy + Debug {
    /// The access type for a region of memory with all permissions.
    const ALL: Self;
}

/// A ledger record.
///
/// Note that this data type is designed to:
/// 1. be naturally aligned
/// 2. divide evenly into a single page
#[cfg_attr(target_pointer_width = "32", repr(C, align(16)))]
#[cfg_attr(target_pointer_width = "64", repr(C, align(32)))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Record<T: LedgerAccess> {
    /// The covered region of memory.
    pub region: Region,

    /// The access permissions.
    pub access: T,
}

impl<T: LedgerAccess> ConstDefault for Record<T> {
    const DEFAULT: Self = Record {
        region: Region::new(Address::NULL, Address::NULL),
        access: T::DEFAULT,
    };
}

/// Ledger error conditions.
#[derive(Debug)]
pub enum Error {
    /// Out of storage capacity
    OutOfCapacity,

    /// No space for the region
    OutOfSpace,
}

/// A virtual memory map ledger.
#[derive(Clone)]
pub struct Ledger<T: LedgerAccess, const N: usize> {
    /// Memory records stored into the ledger.
    records: [Record<T>; N],
    /// Address region that the ledger maintains.
    region: Region,
    /// Tail of the records currently in the ledger.
    tail: usize,
}

impl<T: LedgerAccess, const N: usize> Debug for Ledger<T, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ledger {{ records: ")?;
        f.debug_list()
            .entries(self.records[..self.tail].iter())
            .finish()?;
        write!(f, " }}")
    }
}

impl<T: LedgerAccess, const N: usize> Ledger<T, N> {
    /// Remove the record at index.
    fn remove(&mut self, index: usize) {
        assert!(self.tail > index);

        self.records[index] = Record::DEFAULT;
        self.records[index..].rotate_left(1);
        self.tail -= 1;
    }

    /// Insert a record at the index, shifting later records right.
    fn insert(&mut self, index: usize, record: Record<T>) -> Result<(), Error> {
        assert!(self.tail <= self.records.len());
        assert!(self.tail == self.records().len());
        assert!(self.tail >= index);

        if self.tail == self.records.len() {
            return Err(Error::OutOfCapacity);
        }

        self.records[index..].rotate_right(1);
        self.records[index] = record;
        self.tail += 1;

        Ok(())
    }

    /// Create a new instance.
    pub fn new(addr: Address<usize, Page>, length: Offset<usize, Page>) -> Self {
        let region = Span::new(addr, length).into();
        Self {
            records: [Record::<T>::DEFAULT; N],
            region,
            tail: 0,
        }
    }

    /// Check whether the ledger contains the given region, and return the
    /// maximum allowed access for it. Any empty space will result `None`.
    pub fn contains(&self, addr: Address<usize, Page>, length: Offset<usize, Page>) -> Option<T> {
        let region: Region = Span::new(addr, length).into();
        let mut access = T::ALL;
        let mut start = region.start;

        if !self.region.contains(&region) {
            return None;
        }

        for record in self.records() {
            if let Some(slice) = record.region.intersection(Region::new(start, region.end)) {
                if start != slice.start {
                    return None;
                }

                start = slice.end;
                access &= record.access;

                if start == region.end {
                    return Some(access);
                }
            }
        }

        None
    }

    /// Check whether the existing reserved addresses in the ledger overlap with the
    /// given region.
    pub fn overlaps(&self, addr: Address<usize, Page>, length: Offset<usize, Page>) -> bool {
        let region: Region = Span::new(addr, length).into();

        self.records()
            .iter()
            .any(|record| region.start < record.region.end && region.end > record.region.start)
    }

    /// Get an immutable view of the records.
    pub fn records(&self) -> &[Record<T>] {
        &self.records[..self.tail]
    }

    /// Get a mutable view of the records.
    ///
    /// This function MUST NOT be public.
    fn records_mut(&mut self) -> &mut [Record<T>] {
        &mut self.records[..self.tail]
    }

    /// Merge adjacent records.
    fn merge(&mut self) -> Result<(), Error> {
        let length = self.records().len();
        let mut merges = 0;
        for (p, n) in (0..length).zip(1..length) {
            let prev = self.records()[p - merges];
            let next = self.records()[n - merges];
            if prev.region.end == next.region.start && prev.access == next.access {
                self.records_mut()[n - merges].region.start = prev.region.start;
                self.remove(p - merges);
                merges += 1;
            }
        }
        Ok(())
    }

    /// Reserve an address range from the ledger. When overlapping with an
    /// existing record, the new access will be over-written. Conserves space by
    /// merging the adjacent records in the ledger after the reservation has
    /// been done.
    pub fn map(
        &mut self,
        addr: Address<usize, Page>,
        length: Offset<usize, Page>,
        access: T,
    ) -> Result<(), Error> {
        let region = Span::new(addr, length).into();

        // Clear out the possibly reserved space for the new record.
        self.unmap(addr, length)?;

        match self.records().len() {
            0 => self.insert(0, Record { region, access }).and(self.merge()),
            1 => {
                let record = self.records()[0];

                // Self-consistency check.
                assert!(record.region.start < record.region.end);

                if region.start < record.region.start {
                    self.insert(0, Record { region, access }).and(self.merge())
                } else {
                    self.insert(1, Record { region, access }).and(self.merge())
                }
            }
            _ => {
                for i in 0..self.records().len() {
                    let record = self.records()[i];

                    // Self-consistency check.
                    assert!(record.region.start < record.region.end);

                    if region.start < record.region.start {
                        return self.insert(i, Record { region, access }).and(self.merge());
                    }
                }

                self.insert(self.records().len(), Record { region, access })
                    .and(self.merge())
            }
        }
    }

    /// Find the smallest address where a region of given size fits.
    pub fn find_free_front(&self, length: Offset<usize, Page>) -> Option<Address<usize, Page>> {
        if length.bytes() == 0 || length > (self.region.end - self.region.start) {
            return None;
        }

        if self.tail == 0 {
            return Some(self.region.start);
        }

        // Front tail:
        let first = self.records().first().unwrap().region;
        if Address::new(length.bytes()) <= first.start {
            return Some(self.region.start);
        }

        // Gaps:
        for (prev, next) in (0..self.tail).zip(1..self.tail) {
            let prev = self.records[prev].region;
            let next = self.records[next].region;
            let gap = next.start - prev.end;
            if length <= gap {
                return Some(prev.end);
            }
        }

        // Back tail:
        let last = self.records().last().unwrap().region;
        let gap = self.region.end - last.end;
        if length <= gap {
            return Some(last.end);
        }

        None
    }

    /// Find the largest address where a region of given size fits.
    pub fn find_free_back(&self, length: Offset<usize, Page>) -> Option<Address<usize, Page>> {
        if length.bytes() == 0 || length > (self.region.end - self.region.start) {
            return None;
        }

        if self.tail == 0 {
            return Some(self.region.end - length);
        }

        // Back tail:
        let last = self.records().last().unwrap().region;
        let gap = self.region.end - last.end;
        if length <= gap {
            return Some(self.region.end - length);
        }

        // Gaps:
        for (prev, next) in (0..self.tail).zip(1..self.tail) {
            let prev = self.records[prev].region;
            let next = self.records[next].region;
            let gap = next.start - prev.end;
            if length <= gap {
                return Some(next.start - length);
            }
        }

        // Front tail:
        let first = self.records().first().unwrap().region;
        if self.tail == 0 || Address::new(length.bytes()) <= first.start {
            return Some(first.start - length);
        }

        None
    }

    /// Delete sub-regions.
    pub fn unmap(
        &mut self,
        addr: Address<usize, Page>,
        length: Offset<usize, Page>,
    ) -> Result<(), Error> {
        self.unmap_with(addr, length, |_| {})
    }

    /// Delete sub-regions and call a function on each deleted region.
    pub fn unmap_with(
        &mut self,
        addr: Address<usize, Page>,
        length: Offset<usize, Page>,
        mut f: impl FnMut(&Record<T>),
    ) -> Result<(), Error> {
        let region: Region = Span::new(addr, length).into();

        let mut index = 0;

        while index < self.tail {
            let record_start = self.records[index].region.start;
            let record_end = self.records[index].region.end;

            match (
                (region.start <= record_start),
                (region.end >= record_end),
                (region.start >= record_end),
                (region.end <= record_start),
            ) {
                (false, true, true, false) => {
                    // [   ]   XXXXX
                    // The record is fully outside the region.
                    // Try the next record.
                }
                (true, false, false, true) => {
                    // The record is fully outside the region.
                    // XXXXX   [   ]
                    // Any remaining records are after the region.
                    return Ok(());
                }

                (true, true, false, false) => {
                    // XX[XX]XX
                    // The record is fully contained in the region.
                    f(&self.records[index]);
                    self.remove(index);
                    // Jump without `index += 1` so that a left-shifted record will
                    // not be skipped:
                    continue;
                }
                (false, false, false, false) => {
                    // [   XXXXXX    ]
                    // The record fully contains the region.
                    let before = Record {
                        region: Region::new(record_start, region.start),
                        access: self.records[index].access,
                    };
                    let after = Record {
                        region: Region::new(region.end, record_end),
                        access: self.records[index].access,
                    };
                    f(&Record {
                        region,
                        access: self.records[index].access,
                    });
                    // Put `after` first because it will be right-shifted by `Self::commit()`.
                    self.records[index] = after;

                    // Any remaining records are after the region.
                    return self.insert(index, before);
                }
                (false, true, false, false) => {
                    // [  XXX]XXXX
                    f(&Record {
                        region: Region::new(region.start, record_end),
                        access: self.records[index].access,
                    });
                    self.records[index].region.end = region.start;
                }
                (true, false, false, false) => {
                    // XXX[XXXX   ]
                    f(&Record {
                        region: Region::new(record_start, region.end),
                        access: self.records[index].access,
                    });
                    self.records[index].region.start = region.end;
                    // Any remaining records are after the region.
                    return Ok(());
                }
                _ => unreachable!(
                    "unmap region {:#?} from {:#?}",
                    region, self.records[index].region
                ),
            }
            index += 1;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::cmp::max;
    use core::fmt;

    bitflags::bitflags! {
        /// Memory access permissions.
        #[derive(Default)]
        #[repr(transparent)]
        pub struct Access: usize {
            /// Access::READ access
            const READ = 1 << 0;

            /// Access::WRITE access
            const WRITE = 1 << 1;

            /// Execute access
            const EXECUTE = 1 << 2;
        }
    }

    impl ConstDefault for Access {
        const DEFAULT: Self = Self::empty();
    }

    impl LedgerAccess for Access {
        const ALL: Self = Self::all();
    }

    impl fmt::Display for Access {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "{}{}{}",
                if self.contains(Access::READ) {
                    'r'
                } else {
                    '-'
                },
                if self.contains(Access::WRITE) {
                    'w'
                } else {
                    '-'
                },
                if self.contains(Access::EXECUTE) {
                    'x'
                } else {
                    '-'
                }
            )
        }
    }

    const N: Access = Access::DEFAULT;
    const R: Access = Access::READ;
    const W: Access = Access::WRITE;

    const FULL: Record<Access> = Record {
        region: Region::new(Address::new(0), Address::new(0x10000)),
        access: Access::READ,
    };

    const LOWER_HALF_R: Record<Access> = Record {
        region: Region::new(Address::new(0), Address::new(0x8000)),
        access: Access::READ,
    };

    const UPPER_HALF_W: Record<Access> = Record {
        region: Region::new(Address::new(0x8000), Address::new(0x10000)),
        access: Access::WRITE,
    };

    const EMPTY_LEDGER: Ledger<Access, 5> = Ledger {
        records: [Record::DEFAULT; 5],
        region: Region::new(Address::new(0x0000), Address::new(0x10000)),
        tail: 0,
    };

    const FULL_LEDGER: Ledger<Access, 5> = Ledger {
        records: [
            FULL,
            Record::DEFAULT,
            Record::DEFAULT,
            Record::DEFAULT,
            Record::DEFAULT,
        ],
        region: Region::new(Address::new(0x0000), Address::new(0x10000)),
        tail: 1,
    };

    const MIXED_LEDGER: Ledger<Access, 5> = Ledger {
        records: [
            LOWER_HALF_R,
            UPPER_HALF_W,
            Record::DEFAULT,
            Record::DEFAULT,
            Record::DEFAULT,
        ],
        region: Region::new(Address::new(0x0000), Address::new(0x10000)),
        tail: 2,
    };

    fn records_from_rstest(maps: &[(usize, usize, Access)]) -> Vec<Record<Access>> {
        let maps = maps
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();
        maps
    }

    fn ledger_map_from_rstest<const N: usize>(
        ledger: &mut Ledger<Access, N>,
        maps: &[(usize, usize, Access)],
    ) {
        maps.iter().for_each(|r| {
            let span: Span = Region::new(Address::new(r.0 << 12), Address::new(r.1 << 12)).into();
            ledger.map(span.start, span.count, r.2).unwrap();
        });
    }

    fn regions_from_rstest(maps: &[(usize, usize)]) -> Vec<Region> {
        let maps = maps
            .iter()
            .cloned()
            .map(|record| Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)))
            .collect::<Vec<_>>();
        maps
    }

    fn trace_records(records: &[Record<Access>]) {
        for record in records {
            println!(
                "[{:>#08x}, {:>#08x} {}]",
                record.region.start, record.region.end, record.access
            );
        }
    }

    fn trace_regions(regions: &[Region]) {
        for region in regions {
            println!("[{:>#08x}, {:>#08x}]", region.start, region.end);
        }
    }

    fn trace_assert_records_eq(a: &[Record<Access>], b: &[Record<Access>]) {
        let max = max(a.len(), b.len());
        let mut equal = true;

        for i in 0..max {
            if i >= a.len() {
                equal = false;
                println!(
                    "[?, ?, ?] == [{:>#08x}, {:>#08x}, {}]",
                    b[i].region.start, b[i].region.end, b[i].access
                );
            } else if i >= b.len() {
                equal = false;
                println!(
                    "[{:>#08x}, {:>#08x}, {}] == [?, ?, ?]",
                    a[i].region.start, a[i].region.end, a[i].access
                );
            } else if a[i] == b[i] {
                println!(
                    "[{:>#08x}, {:>#08x}, {}] == [{:>#08x}, {:>#08x}, {}]",
                    a[i].region.start,
                    a[i].region.end,
                    a[i].access,
                    b[i].region.start,
                    b[i].region.end,
                    b[i].access
                );
            } else {
                equal = false;
                println!(
                    "[{:>#08x}, {:>#08x}, {}] != [{:>#08x}, {:>#08x}, {}]",
                    a[i].region.start,
                    a[i].region.end,
                    a[i].access,
                    b[i].region.start,
                    b[i].region.end,
                    b[i].access
                );
            }
        }

        assert!(equal);
    }

    #[rstest::rstest]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x0, 0x0d, None))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x3, 0x10, None))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x0, 0x10, None))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x3, 0xd, Some(N)))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x6, 0xa, Some(R)))]
    fn contains(
        #[case] maps: &[(usize, usize, Access)],
        #[case] expected: (usize, usize, Option<Access>),
    ) {
        let mut ledger = EMPTY_LEDGER.clone();
        ledger_map_from_rstest(&mut ledger, maps);

        println!("Maps:");
        trace_records(&ledger.records);
        println!("Region:");
        println!("({:>#08x}, {:>#08x})", expected.0 << 12, expected.1 << 12,);

        let access = ledger.contains(
            Address::new(expected.0 << 12),
            Offset::from_items(expected.1 - expected.0),
        );
        assert_eq!(access, expected.2);
    }

    #[rstest::rstest]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x0, 0x10, true))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x2, 0x7, true))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0xc, 0xe, true))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0xd, 0xe, false))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x2, 0x3, false))]
    fn overlaps(#[case] maps: &[(usize, usize, Access)], #[case] expected: (usize, usize, bool)) {
        let mut ledger = EMPTY_LEDGER.clone();
        ledger_map_from_rstest(&mut ledger, maps);

        println!("Maps:");
        trace_records(&ledger.records);
        println!("Region:");
        println!("({:>#08x}, {:>#08x})", expected.0 << 12, expected.1 << 12,);

        let access = ledger.overlaps(
            Address::new(expected.0 << 12),
            Offset::from_items(expected.1 - expected.0),
        );
        assert_eq!(access, expected.2);
    }

    #[rstest::rstest]
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // normal insert
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x0, 0x1, N)], &[(0x0, 0x1, N), (0x3, 0x6, N), (0xa, 0xd, N)])] // normal insert
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x7, 0x8, N)], &[(0x3, 0x6, N), (0x7, 0x8, N), (0xa, 0xd, N)])] // normal insert
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xe, 0xf, N)], &[(0x3, 0x6, N), (0xa, 0xd, N), (0xe, 0xf, N)])] // normal insert
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x2, 0x3, N)], &[(0x2, 0x6, N), (0xa, 0xd, N)])] // merge before
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x9, 0xa, N)], &[(0x3, 0x6, N), (0x9, 0xd, N)])] // merge before
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x2, 0x4, N)], &[(0x2, 0x6, N), (0xa, 0xd, N)])] // merge before overlap
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x9, 0xb, N)], &[(0x3, 0x6, N), (0x9, 0xd, N)])] // merge before overlap
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x6, 0x7, N)], &[(0x3, 0x7, N), (0xa, 0xd, N)])] // merge after
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xd, 0xe, N)], &[(0x3, 0x6, N), (0xa, 0xe, N)])] // merge after
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x5, 0x7, N)], &[(0x3, 0x7, N), (0xa, 0xd, N)])] // merge after overlap
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xc, 0xe, N)], &[(0x3, 0x6, N), (0xa, 0xe, N)])] // merge after overlap
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x2, 0x3, R)], &[(0x2, 0x3, R), (0x3, 0x6, N), (0xa, 0xd, N)])] // no merge before
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x9, 0xa, R)], &[(0x3, 0x6, N), (0x9, 0xa, R), (0xa, 0xd, N)])] // no merge before
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x6, 0x7, R)], &[(0x3, 0x6, N), (0x6, 0x7, R), (0xa, 0xd, N)])] // no merge after
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xd, 0xe, R)], &[(0x3, 0x6, N), (0xa, 0xd, N), (0xd, 0xe, R)])] // no merge after
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x3, 0x6, N)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // no update
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xa, 0xd, N)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // no update
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x3, 0x6, R)], &[(0x3, 0x6, R), (0xa, 0xd, N)])] // update
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xa, 0xd, R)], &[(0x3, 0x6, N), (0xa, 0xd, R)])] // update
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x0, 0xf, N)], &[(0x0, 0xf, N)])] // replace
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x0, 0xf, R)], &[(0x0, 0xf, R)])] // replace
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x4, 0x5, N)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // no split
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xb, 0xc, N)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // no split
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x4, 0x5, R)], &[(0x3, 0x4, N), (0x4, 0x5, R), (0x5, 0x6, N), (0xa, 0xd, N)])] // split
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xb, 0xc, R)], &[(0x3, 0x6, N), (0xa, 0xb, N), (0xb, 0xc, R), (0xc, 0xd, N)])] // split
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x2, 0x4, R)], &[(0x2, 0x4, R), (0x4, 0x6, N), (0xa, 0xd, N)])] // overlap before
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x9, 0xb, R)], &[(0x3, 0x6, N), (0x9, 0xb, R), (0xb, 0xd, N)])] // overlap before
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0x5, 0x7, R)], &[(0x3, 0x5, N), (0x5, 0x7, R), (0xa, 0xd, N)])] // overlap after
    #[case(&[(0x3, 0x6, N), (0xa, 0xd, N), (0xc, 0xe, R)], &[(0x3, 0x6, N), (0xa, 0xc, N), (0xc, 0xe, R)])] // overlap after
    fn map(#[case] maps: &[(usize, usize, Access)], #[case] expected: &[(usize, usize, Access)]) {
        let mut ledger = EMPTY_LEDGER.clone();
        ledger_map_from_rstest(&mut ledger, maps);

        let expected = records_from_rstest(expected);

        println!("Maps:");
        trace_records(&ledger.records);

        println!("Result:");
        trace_assert_records_eq(ledger.records(), &expected);
    }

    #[rstest::rstest]
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10)], &[(0x3, 0x6, R), (0xa, 0xd, R)], 3)] // split
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x1, 0xf)], &[], 5)] // clear
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x0, 0x1)], &[(0x3, 0x6, R), (0xa, 0xd, R)], 3)] // noop
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x7, 0x8)], &[(0x3, 0x6, R), (0xa, 0xd, R)], 3)] // noop
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0xe, 0xf)], &[(0x3, 0x6, R), (0xa, 0xd, R)], 3)] // noop
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x3, 0x6)], &[(0xa, 0xd, R)], 4)] // remove
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0xa, 0xd)], &[(0x3, 0x6, R)], 4)] // remove
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x2, 0x7)], &[(0xa, 0xd, R)], 4)] // remove oversized
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x9, 0xe)], &[(0x3, 0x6, R)], 4)] // remove oversized
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x2, 0x4)], &[(0x4, 0x6, R), (0xa, 0xd, R)], 4)] // overlap before
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x9, 0xb)], &[(0x3, 0x6, R), (0xb, 0xd, R)], 4)] // overlap before
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x5, 0x7)], &[(0x3, 0x5, R), (0xa, 0xd, R)], 4)] // overlap after
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0xc, 0xe)], &[(0x3, 0x6, R), (0xa, 0xc, R)], 4)] // overlap after
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x4, 0x5)], &[(0x3, 0x4, R), (0x5, 0x6, R), (0xa, 0xd, R)], 4)] // split
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0xb, 0xc)], &[(0x3, 0x6, R), (0xa, 0xb, R), (0xc, 0xd, R)], 4)] // split
    fn unmap(
        #[case] unmaps: &[(usize, usize)],
        #[case] expected: &[(usize, usize, Access)],
        #[case] expected_unmapped: usize,
    ) {
        let unmaps = regions_from_rstest(unmaps);
        let expected = records_from_rstest(expected);

        let mut ledger = FULL_LEDGER.clone();
        assert_eq!(ledger.records(), &[FULL]);

        println!("{:#?}", &ledger);
        println!("Unmaps:");
        trace_regions(&unmaps);

        let mut unmapped = 0;

        for region in unmaps {
            let addr = region.start;
            let length = region.end - region.start;
            ledger
                .unmap_with(addr, length, |record| {
                    unmapped += 1;
                    println!(
                        "Unmapped [{:>#08x}, {:>#08x}]",
                        record.region.start, record.region.end
                    );
                })
                .unwrap();
        }

        println!("Result:");
        println!("{:#?}", &ledger);
        trace_assert_records_eq(ledger.records(), &expected);
        assert_eq!(unmapped, expected_unmapped);
    }

    #[rstest::rstest]
    #[case(&[(0x8, 0x10)], &[(0x0, 0x8, R)], 1)] // start
    #[case(&[(0x0, 0x8)], &[(0x8, 0x10, W)], 1)] // end
    #[case(&[(0x8, 0x9)], &[(0x0, 0x8, R), (0x9, 0x10, W)], 1)] // split
    #[case(&[(0x8, 0x9), (0x8, 0x10)], &[(0x0, 0x8, R)], 2)] // split and end
    #[case(&[(0x7, 0x9)], &[(0x0, 0x7, R), (0x9, 0x10, W)], 2)] // split
    #[case(&[(0x7, 0x9), (0x0, 0x10)], &[], 4)] // empty
    fn unmap_mixed(
        #[case] unmaps: &[(usize, usize)],
        #[case] expected: &[(usize, usize, Access)],
        #[case] expected_unmapped: usize,
    ) {
        let unmaps = regions_from_rstest(unmaps);
        let expected = records_from_rstest(expected);

        let mut ledger = MIXED_LEDGER.clone();

        println!("Unmaps:");
        trace_regions(&unmaps);

        let mut unmapped = 0;

        for region in unmaps {
            let addr = region.start;
            let length = region.end - region.start;
            ledger
                .unmap_with(addr, length, |record| {
                    unmapped += 1;
                    println!(
                        "Unmapped [{:>#08x}, {:>#08x}]",
                        record.region.start, record.region.end
                    );
                })
                .unwrap();
        }

        println!("Result:");
        println!("{:#?}", &ledger);
        trace_assert_records_eq(ledger.records(), &expected);
        assert_eq!(unmapped, expected_unmapped);
    }

    #[rstest::rstest]
    #[case(0x1, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x0, 0x1, N), (0x3, 0x6, N), (0xa, 0xd, N)])]
    #[case(0x2, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x0, 0x2, N), (0x3, 0x6, N), (0xa, 0xd, N)])]
    #[case(0x3, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x0, 0x6, N), (0xa, 0xd, N)])]
    #[case(0x4, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x3, 0xd, N)])]
    #[case(0x5, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x3, 0x6, N), (0xa, 0xd, N)])]
    fn find_free_front(
        #[case] length: usize,
        #[case] maps: &[(usize, usize, Access)],
        #[case] expected: &[(usize, usize, Access)],
    ) {
        let length = Offset::from_items(length);
        let mut ledger = EMPTY_LEDGER.clone();
        ledger_map_from_rstest(&mut ledger, maps);

        let expected = records_from_rstest(expected);

        println!("Length: {}", length);
        println!("Maps:");
        trace_records(&ledger.records);

        if let Some(addr) = ledger.find_free_front(length) {
            ledger.map(addr, length, Access::empty()).unwrap();
        }

        println!("Result:");
        trace_assert_records_eq(ledger.records(), &expected);
    }

    #[rstest::rstest]
    #[case(0x1, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x3, 0x6, N), (0xa, 0xd, N), (0xf, 0x10, N)])]
    #[case(0x2, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x3, 0x6, N), (0xa, 0xd, N), (0xe, 0x10, N)])]
    #[case(0x3, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x3, 0x6, N), (0xa, 0x10, N)])]
    #[case(0x4, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x3, 0xd, N)])]
    #[case(0x5, &[(0x3, 0x6, N), (0xa, 0xd, N)], &[(0x3, 0x6, N), (0xa, 0xd, N)])]
    fn find_free_back(
        #[case] length: usize,
        #[case] maps: &[(usize, usize, Access)],
        #[case] expected: &[(usize, usize, Access)],
    ) {
        let length = Offset::from_items(length);
        let mut ledger = EMPTY_LEDGER.clone();
        ledger_map_from_rstest(&mut ledger, maps);
        let expected = records_from_rstest(expected);

        println!("Length: {}", length);
        println!("Maps:");
        trace_records(&ledger.records);

        if let Some(addr) = ledger.find_free_back(length) {
            ledger.map(addr, length, Access::empty()).unwrap();
        }

        println!("Result:");
        trace_assert_records_eq(ledger.records(), &expected);
    }

    #[test]
    fn record_size_align() {
        use core::mem::{align_of, size_of};
        assert_eq!(size_of::<Record<Access>>(), size_of::<usize>() * 4);
        assert_eq!(align_of::<Record<Access>>(), size_of::<Record<Access>>());
    }
}
