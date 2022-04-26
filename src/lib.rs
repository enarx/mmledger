//! A ledger for memory mappings.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use core::fmt;

use lset::Contains;
use primordial::{Address, Offset, Page};

/// A region of memory.
pub type Region = lset::Line<Address<usize, Page>>;

bitflags::bitflags! {
    /// Memory access permissions.
    #[derive(Default)]
    #[repr(transparent)]
    pub struct Access: usize {
        /// Read access
        const READ = 1 << 0;

        /// Write access
        const WRITE = 1 << 0;

        /// Execute access
        const EXECUTE = 1 << 0;
    }
}

impl Access {
    /// Creates a record for these permissions and the given region.
    pub const fn record(self, region: Region) -> Record {
        Record {
            region,
            access: self,
        }
    }
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

/// A ledger record.
///
/// Note that this data type is designed to:
/// 1. be naturally aligned
/// 2. divide evenly into a single page
#[cfg_attr(target_pointer_width = "32", repr(C, align(16)))]
#[cfg_attr(target_pointer_width = "64", repr(C, align(32)))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Record {
    /// The covered region of memory.
    pub region: Region,

    /// The access permissions.
    pub access: Access,
}

impl Record {
    const EMPTY: Record = Record {
        region: Region::new(Address::NULL, Address::NULL),
        access: Access::empty(),
    };
}

/// Ledger error conditions.
#[derive(Debug)]
pub enum Error {
    /// Out of storage capacity
    OutOfCapacity,

    /// No space for the region
    OutOfSpace,

    /// The given region has corrupted contents
    InvalidRegion,
}

/// A virtual memory map ledger.
#[derive(Clone, Debug)]
pub struct Ledger<const N: usize> {
    /// Memory records stored into the ledger.
    records: [Record; N],
    /// Address region that the ledger maintains.
    region: Region,
    /// Tail of the records currently in the ledger.
    tail: usize,
}

impl<const N: usize> Ledger<N> {
    /// Remove the record at index.
    fn remove(&mut self, index: usize) {
        assert!(self.tail > index);

        self.records[index] = Record::EMPTY;
        self.records[index..].rotate_left(1);
        self.tail -= 1;
    }

    /// Insert a record at the index, shifting later records right.
    fn insert(&mut self, index: usize, record: Record) -> Result<(), Error> {
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
    pub const fn new(region: Region) -> Self {
        Self {
            records: [Record::EMPTY; N],
            region,
            tail: 0,
        }
    }

    /// Check whether the ledger contains the given region, and return the
    /// maximum allowed access for it.
    pub fn contains(&self, region: Region) -> Option<Access> {
        if region.start >= region.end || !self.region.contains(&region) {
            return None;
        }

        let mut start = region.start;
        let mut access = Access::all();

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

    /// Check whether the existing addresses in the ledger overlap with the
    /// given region.
    pub fn overlaps(&self, region: Region) -> bool {
        if region.start >= region.end || !self.region.contains(&region) {
            return false;
        }

        self.records()
            .iter()
            .any(|record| region.start < record.region.end && region.end > record.region.start)
    }

    /// Get an immutable view of the records.
    pub fn records(&self) -> &[Record] {
        &self.records[..self.tail]
    }

    /// Get a mutable view of the records.
    ///
    /// This function MUST NOT be public.
    fn records_mut(&mut self) -> &mut [Record] {
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

    /// Adds a new record to the ledger, potentially merging with existing records.
    pub fn map(&mut self, region: Region, access: Access) -> Result<(), Error> {
        if region.start >= region.end || !self.region.contains(&region) {
            return Err(Error::InvalidRegion);
        }

        // Clear out the space for the new record.
        if let Err(err) = self.unmap(region) {
            return Err(err);
        }

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
    pub fn unmap(&mut self, region: Region) -> Result<(), Error> {
        if region.start >= region.end || !self.region.contains(&region) {
            return Err(Error::InvalidRegion);
        }

        let mut index = 0;

        while index < self.tail {
            let record_start = self.records[index].region.start;
            let record_end = self.records[index].region.end;

            if region.end < record_start || region.start > record_end {
                // Skip:
                index += 1;
                continue;
            }

            if region.start <= record_start && region.end >= record_end {
                self.remove(index);
                // Jump without `index += 1` so that a left-shifted record will
                // not be skipped:
                continue;
            }

            if region.start > record_start && region.end < record_end {
                let before: Record = Record {
                    region: Region::new(record_start, region.start),
                    access: Access::empty(),
                };
                let after: Record = Record {
                    region: Region::new(region.end, record_end),
                    access: Access::empty(),
                };
                // Put `after` first because it will be right-shifted by `Self::commit()`.
                self.records[index] = after;

                return self.insert(index, before);
            }

            if region.start > record_start {
                self.records[index].region.end = region.start;
            } else {
                self.records[index].region.start = region.end;
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

    const N: Access = Access::empty();
    const R: Access = Access::READ;

    const FULL: Record = Record {
        region: Region::new(Address::new(0), Address::new(0x10000)),
        access: Access::empty(),
    };

    const EMPTY_LEDGER: Ledger<5> = Ledger {
        records: [Record::EMPTY; 5],
        region: Region::new(Address::new(0x0000), Address::new(0x10000)),
        tail: 0,
    };

    const FULL_LEDGER: Ledger<5> = Ledger {
        records: [
            FULL,
            Record::EMPTY,
            Record::EMPTY,
            Record::EMPTY,
            Record::EMPTY,
        ],
        region: Region::new(Address::new(0x0000), Address::new(0x10000)),
        tail: 1,
    };

    fn trace_records(records: &[Record]) {
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

    fn trace_assert_records_eq(a: &[Record], b: &[Record]) {
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
                    b[i].access,
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
        for record in maps
            .iter()
            .cloned()
            .map(|r| Record {
                region: Region::new(Address::new(r.0 << 12), Address::new(r.1 << 12)),
                access: r.2,
            })
            .collect::<Vec<_>>()
        {
            ledger.map(record.region, record.access).unwrap();
        }

        println!("Maps:");
        trace_records(&ledger.records);
        println!("Region:");
        println!("({:>#08x}, {:>#08x})", expected.0 << 12, expected.1 << 12,);

        assert_eq!(
            ledger.contains(Region::new(
                Address::new(expected.0 << 12),
                Address::new(expected.1 << 12)
            )),
            expected.2
        );
    }

    #[rstest::rstest]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x0, 0x10, true))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x2, 0x7, true))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0xc, 0xe, true))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0xd, 0xe, false))]
    #[case(&[(0x3, 0x6, N), (0x6, 0xa, R), (0xa, 0xd, N)], (0x2, 0x3, false))]
    fn overlaps(#[case] maps: &[(usize, usize, Access)], #[case] expected: (usize, usize, bool)) {
        let mut ledger = EMPTY_LEDGER.clone();
        for record in maps
            .iter()
            .cloned()
            .map(|r| Record {
                region: Region::new(Address::new(r.0 << 12), Address::new(r.1 << 12)),
                access: r.2,
            })
            .collect::<Vec<_>>()
        {
            ledger.map(record.region, record.access).unwrap();
        }

        println!("Maps:");
        trace_records(&ledger.records);
        println!("Region:");
        println!("({:>#08x}, {:>#08x})", expected.0 << 12, expected.1 << 12,);

        assert_eq!(
            ledger.overlaps(Region::new(
                Address::new(expected.0 << 12),
                Address::new(expected.1 << 12)
            )),
            expected.2
        );
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
        let maps = maps
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();

        let expected = expected
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();

        let mut ledger = EMPTY_LEDGER.clone();

        println!("Maps:");
        trace_records(&maps);

        for record in maps {
            ledger.map(record.region, record.access).unwrap();
        }

        println!("Result:");
        trace_assert_records_eq(ledger.records(), &expected);
    }

    #[rstest::rstest]
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // split
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x1, 0xf)], &[])] // clear
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x0, 0x1)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // noop
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x7, 0x8)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // noop
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0xe, 0xf)], &[(0x3, 0x6, N), (0xa, 0xd, N)])] // noop
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x3, 0x6)], &[(0xa, 0xd, N)])] // remove
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0xa, 0xd)], &[(0x3, 0x6, N)])] // remove
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x2, 0x7)], &[(0xa, 0xd, N)])] // remove oversized
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x9, 0xe)], &[(0x3, 0x6, N)])] // remove oversized
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x2, 0x4)], &[(0x4, 0x6, N), (0xa, 0xd, N)])] // overlap before
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x9, 0xb)], &[(0x3, 0x6, N), (0xb, 0xd, N)])] // overlap before
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x5, 0x7)], &[(0x3, 0x5, N), (0xa, 0xd, N)])] // overlap after
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0xc, 0xe)], &[(0x3, 0x6, N), (0xa, 0xc, N)])] // overlap after
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0x4, 0x5)], &[(0x3, 0x4, N), (0x5, 0x6, N), (0xa, 0xd, N)])] // split
    #[case(&[(0x0, 0x3), (0x6, 0xa), (0xd, 0x10), (0xb, 0xc)], &[(0x3, 0x6, N), (0xa, 0xb, N), (0xc, 0xd, N)])] // split
    fn unmap(#[case] unmaps: &[(usize, usize)], #[case] expected: &[(usize, usize, Access)]) {
        let unmaps = unmaps
            .iter()
            .cloned()
            .map(|record| Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)))
            .collect::<Vec<_>>();

        let expected = expected
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();

        let mut ledger = FULL_LEDGER.clone();
        assert_eq!(ledger.records(), &[FULL]);

        println!("Unmaps:");
        trace_regions(&unmaps);

        for region in unmaps {
            ledger.unmap(region).unwrap();
        }

        println!("Result:");
        trace_assert_records_eq(ledger.records(), &expected);
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
        let maps = maps
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();
        let expected = expected
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();

        let mut ledger = EMPTY_LEDGER.clone();

        println!("Length: {}", length);
        println!("Maps:");
        trace_records(&maps);

        for record in maps {
            ledger.map(record.region, record.access).unwrap();
        }

        if let Some(addr) = ledger.find_free_front(length) {
            let end = addr + length;
            let region = Region::new(addr, end);
            ledger.map(region, Access::empty()).unwrap();
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
        let maps = maps
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();
        let expected = expected
            .iter()
            .cloned()
            .map(|record| Record {
                region: Region::new(Address::new(record.0 << 12), Address::new(record.1 << 12)),
                access: record.2,
            })
            .collect::<Vec<_>>();

        let mut ledger = EMPTY_LEDGER.clone();

        println!("Length: {}", length);
        println!("Maps:");
        trace_records(&maps);

        for record in maps {
            ledger.map(record.region, record.access).unwrap();
        }

        if let Some(addr) = ledger.find_free_back(length) {
            let end = addr + length;
            let region = Region::new(addr, end);
            ledger.map(region, Access::empty()).unwrap();
        }

        println!("Result:");
        trace_assert_records_eq(ledger.records(), &expected);
    }

    #[test]
    fn record_size_align() {
        use core::mem::{align_of, size_of};
        assert_eq!(size_of::<Record>(), size_of::<usize>() * 4);
        assert_eq!(align_of::<Record>(), size_of::<Record>());
    }
}
