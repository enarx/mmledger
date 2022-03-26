//! A ledger for memory mappings.

#![cfg_attr(not(test), no_std)]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use lset::Span;
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

    fn is_between(&self, lhs: &Record, rhs: &Record) -> bool {
        lhs.region < self.region && self.region < rhs.region
    }
}

/// Ledger error conditions.
#[derive(Debug)]
pub enum Error {
    /// Out of storage capacity
    OutOfCapacity,

    /// No space for the region
    OutOfSpace,

    /// Not inside the address space
    Overflow,

    /// Overlap with the existing regions
    Overlap,

    /// Invalid region
    InvalidRegion,
}

/// A virtual memory map ledger.
#[derive(Clone, Debug)]
pub struct Ledger<const N: usize> {
    records: [Record; N],
    region: Region,
    length: usize,
}

impl<const N: usize> Ledger<N> {
    /// Remove the record at index.
    fn remove(&mut self, index: usize) {
        assert!(self.length > index);

        self.records[index] = Record::EMPTY;
        self.records[index..].rotate_left(1);
        self.length -= 1;
    }

    /// Insert a record at the index, shifting later records right.
    fn insert(&mut self, index: usize, record: Record) -> Result<(), Error> {
        assert!(self.length <= self.records.len());
        assert!(self.length >= index);

        if self.length == self.records.len() {
            return Err(Error::OutOfCapacity);
        }

        self.records[index..].rotate_right(1);
        self.records[index] = record;
        self.length += 1;

        Ok(())
    }

    /// Create a new instance.
    pub const fn new(region: Region) -> Self {
        Self {
            records: [Record::EMPTY; N],
            region,
            length: 0,
        }
    }

    /// Get an immutable view of the records.
    pub fn records(&self) -> &[Record] {
        &self.records[..self.length]
    }

    /// Get a mutable view of the records.
    ///
    /// This function MUST NOT be public.
    fn records_mut(&mut self) -> &mut [Record] {
        &mut self.records[..self.length]
    }

    /// Adds a new record to the ledger, potentially merging with existing records.
    pub fn add(&mut self, record: Record) -> Result<(), Error> {
        // Make sure the record is valid.
        if record.region.start >= record.region.end {
            return Err(Error::InvalidRegion);
        }

        // Make sure the record fits in our adress space.
        if record.region.start < self.region.start || record.region.end > self.region.end {
            return Err(Error::Overflow);
        }

        // Loop over the records looking for merges.
        for i in 0..self.length + 1 {
            let (lhs, rhs) = self.records_mut().split_at_mut(i);
            let mut prev = lhs.last_mut();
            let mut this = rhs.first_mut();
            let mut merges = (false, false);

            // Check for overlap.
            if let Some(this) = this.as_mut() {
                if this.region.intersection(record.region).is_some() {
                    return Err(Error::Overlap);
                }
            }

            // Potentially merge after `prev`.
            if let Some(prev) = prev.as_mut() {
                if prev.access == record.access && prev.region.end == record.region.start {
                    prev.region.end = record.region.end;
                    merges.0 = true;
                }
            }

            // Potentially merge before `this`.
            if let Some(this) = this.as_mut() {
                if this.access == record.access && this.region.start == record.region.end {
                    this.region.start = record.region.start;
                    merges.1 = true;
                }
            }

            match (merges, (prev, this)) {
                // If there is a gap between two records, insert.
                ((false, false), (Some(prev), Some(this))) if record.is_between(prev, this) => {
                    return self.insert(i, record);
                }

                // If we are at the start, insert.
                ((false, false), (None, Some(this))) if record.region < this.region => {
                    return self.insert(i, record);
                }

                // If we are at the end, insert.
                ((false, false), (Some(prev), None)) if record.region > prev.region => {
                    return self.insert(i, record);
                }

                // If this is the first record, insert.
                ((false, false), (None, None)) => return self.insert(i, record),

                // We merged in both directions. Merge again.
                ((true, true), (Some(prev), Some(this))) => {
                    this.region.start = prev.region.start;
                    self.remove(i - 1);
                    return Ok(());
                }

                ((true, true), ..) => unreachable!(),
                ((true, false), ..) => return Ok(()),
                ((false, true), ..) => return Ok(()),
                ((false, false), ..) => (),
            }
        }

        unreachable!()
    }

    /// Find space for a free region.
    pub fn find_free(&self, len: Offset<usize, Page>, front: bool) -> Result<Region, Error> {
        let start = Record {
            region: Region::new(self.region.start, self.region.start),
            ..Default::default()
        };

        let end = Record {
            region: Region::new(self.region.end, self.region.end),
            ..Default::default()
        };

        // Synthesize a starting window.
        let first = [start, *self.records().first().unwrap_or(&end)];

        // Synthesize an ending window.
        let last = [*self.records().last().unwrap_or(&start), end];

        // Chain everything together.
        let mut iter = first
            .windows(2)
            .chain(self.records().windows(2))
            .chain(last.windows(2));

        // Iterate through the windows.
        if front {
            while let Some([l, r]) = iter.next() {
                let region = Region::from(Span::new(l.region.end, len));
                if region > l.region && region < r.region {
                    return Ok(region);
                }
            }
        } else {
            let mut iter = iter.rev();
            while let Some([l, r]) = iter.next() {
                let region = Region::from(Span::new(r.region.start - len, len));
                if region > l.region && region < r.region {
                    return Ok(region);
                }
            }
        }

        Err(Error::OutOfSpace)
    }

    /// Delete sub-regions.
    pub fn delete(&mut self, region: Region) -> Result<(), Error> {
        if region.start >= region.end {
            return Err(Error::InvalidRegion);
        }

        if region.start < self.region.start || region.end > self.region.end {
            return Err(Error::Overflow);
        }

        let mut index = 0;

        while index < self.length {
            let record_start = self.records[index].region.start;
            let record_end = self.records[index].region.end;

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

    use core::mem::{align_of, size_of};

    const PREV: Record = Record {
        region: Region::new(Address::new(0x2000), Address::new(0x3000)),
        access: Access::empty(),
    };

    const NEXT: Record = Record {
        region: Region::new(Address::new(0x7000), Address::new(0x8000)),
        access: Access::empty(),
    };

    const LEDGER: Ledger<4> = Ledger {
        records: [PREV, NEXT, Record::EMPTY, Record::EMPTY],
        region: Region::new(Address::new(0x0000), Address::new(0xa000)),
        length: 2,
    };

    #[test]
    fn record_size_align() {
        assert_eq!(size_of::<Record>(), size_of::<usize>() * 4);
        assert_eq!(align_of::<Record>(), size_of::<Record>());
    }

    #[test]
    fn insert_start() {
        const RECORD: Record = Record {
            region: Region::new(Address::new(0x0000), Address::new(0x1000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        assert_eq!(ledger.records(), &[PREV, NEXT]);

        ledger.add(RECORD).unwrap();
        assert_eq!(ledger.records(), &[RECORD, PREV, NEXT]);
    }

    #[test]
    fn insert_middle() {
        const RECORD: Record = Record {
            region: Region::new(Address::new(0x5000), Address::new(0x6000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        assert_eq!(ledger.records(), &[PREV, NEXT]);

        ledger.add(RECORD).unwrap();
        assert_eq!(ledger.records(), &[PREV, RECORD, NEXT]);
    }

    #[test]
    fn insert_end() {
        const RECORD: Record = Record {
            region: Region::new(Address::new(0x9000), Address::new(0xa000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        assert_eq!(ledger.records(), &[PREV, NEXT]);

        ledger.add(RECORD).unwrap();
        assert_eq!(ledger.records(), &[PREV, NEXT, RECORD]);
    }

    #[test]
    fn find_free_front() {
        let region = LEDGER.find_free(Offset::from_items(2), true).unwrap();
        assert_eq!(Address::new(0x0000)..Address::new(0x2000), region.into());

        let region = LEDGER.find_free(Offset::from_items(3), true).unwrap();
        assert_eq!(Address::new(0x3000)..Address::new(0x6000), region.into());
    }

    #[test]
    fn find_free_back() {
        let region = LEDGER.find_free(Offset::from_items(2), false).unwrap();
        assert_eq!(Address::new(0x8000)..Address::new(0xa000), region.into());

        let region = LEDGER.find_free(Offset::from_items(3), false).unwrap();
        assert_eq!(Address::new(0x4000)..Address::new(0x7000), region.into());
    }

    #[test]
    fn merge_before_prev() {
        const RECORD: Record = Record {
            region: Region::new(Address::new(0x1000), Address::new(0x2000)),
            access: Access::empty(),
        };

        const MERGED: Record = Record {
            region: Region::new(Address::new(0x1000), Address::new(0x3000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();

        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[0], MERGED);
        assert_eq!(ledger.records[1], NEXT);
    }

    #[test]
    fn merge_before_next() {
        const RECORD: Record = Record {
            region: Region::new(Address::new(0x6000), Address::new(0x7000)),
            access: Access::empty(),
        };

        const MERGED: Record = Record {
            region: Region::new(Address::new(0x6000), Address::new(0x8000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();

        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[0], PREV);
        assert_eq!(ledger.records[1], MERGED);
    }

    #[test]
    fn merge_after_prev() {
        const RECORD: Record = Record {
            region: Region::new(Address::new(0x3000), Address::new(0x4000)),
            access: Access::empty(),
        };

        const MERGED: Record = Record {
            region: Region::new(Address::new(0x2000), Address::new(0x4000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();

        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[0], MERGED);
        assert_eq!(ledger.records[1], NEXT);
    }

    #[test]
    fn merge_after_next() {
        const RECORD: Record = Record {
            region: Region::new(Address::new(0x8000), Address::new(0x9000)),
            access: Access::empty(),
        };

        const MERGED: Record = Record {
            region: Region::new(Address::new(0x7000), Address::new(0x9000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();

        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[0], PREV);
        assert_eq!(ledger.records[1], MERGED);
    }

    #[test]
    fn merge_before() {
        const RECORD: Record = Record {
            region: Region::new(Address::new(0x5000), Address::new(0x7000)),
            access: Access::empty(),
        };

        const MERGED: Record = Record {
            region: Region::new(Address::new(0x5000), Address::new(0x8000)),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();

        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[0], PREV);
        assert_eq!(ledger.records[1], MERGED);
    }

    #[test]
    fn merge_both() {
        const RECORD: Record = Record {
            region: Region::new(PREV.region.end, NEXT.region.start),
            access: Access::empty(),
        };

        const MERGED: Record = Record {
            region: Region::new(PREV.region.start, NEXT.region.end),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();

        assert_eq!(ledger.length, 1);
        assert_eq!(ledger.records[0], MERGED);
    }

    #[test]
    fn delete_after() {
        const RECORD: Record = Record {
            region: Region::new(PREV.region.end, NEXT.region.start),
            access: Access::empty(),
        };

        const MERGED: Record = Record {
            region: Region::new(PREV.region.start, NEXT.region.start),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();
        ledger.delete(NEXT.region).unwrap();

        assert_eq!(ledger.length, 1);
        assert_eq!(ledger.records[0], MERGED);
    }

    #[test]
    fn delete_all() {
        const RECORD: Record = Record {
            region: Region::new(LEDGER.region.start, LEDGER.region.end),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        assert_eq!(ledger.length, 2);

        ledger.delete(RECORD.region).unwrap();
        assert_eq!(ledger.length, 0);
    }

    #[test]
    fn delete_before() {
        const RECORD: Record = Record {
            region: Region::new(PREV.region.end, NEXT.region.start),
            access: Access::empty(),
        };

        const MERGED: Record = Record {
            region: Region::new(PREV.region.end, NEXT.region.end),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();
        ledger.delete(PREV.region).unwrap();

        assert_eq!(ledger.length, 1);
        assert_eq!(ledger.records[0], MERGED);
    }

    #[test]
    fn delete_split() {
        const RECORD: Record = Record {
            region: Region::new(PREV.region.end, NEXT.region.start),
            access: Access::empty(),
        };

        let mut ledger = LEDGER.clone();
        ledger.add(RECORD).unwrap();
        ledger.delete(RECORD.region).unwrap();

        assert_eq!(ledger.length, 2);
        assert_eq!(ledger.records[0], PREV);
        assert_eq!(ledger.records[1], NEXT);
    }
}
