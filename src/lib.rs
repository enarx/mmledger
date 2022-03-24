//! A ledger for memory mappings.

#![no_std]
#![deny(clippy::all)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use core::cmp::Ordering;

use lset::{Empty, Line, Span};
use primordial::{Address, Offset, Page};

bitflags::bitflags! {
    /// Memory permissions.
    #[derive(Default)]
    #[repr(transparent)]
    pub struct Permissions: usize {
        /// Read
        const READ = 1 << 0;
        /// Write
        const WRITE = 1 << 0;
        /// Execute
        const EXECUTE = 1 << 0;
    }
}

/// Represents an access controlled region of address space.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Record {
    /// The covered region of address space.
    pub region: Line<Address<usize, Page>>,

    /// The permissions enforced for the region.
    pub permissions: Permissions,

    length: usize,
}

impl Record {
    const EMPTY: Record = Record {
        region: Line::new(Address::NULL, Address::NULL),
        permissions: Permissions::empty(),
        length: 0,
    };

    fn new(region: Line<Address<usize, Page>>, permissions: Permissions) -> Self {
        Record {
            region,
            permissions,
            length: 0,
        }
    }
}

/// Ledger error conditions.
#[derive(Debug)]
pub enum Error {
    /// Out of storage capacity
    NoCapacity,
    /// No space for the region
    NoSpace,
    /// Not inside the address space
    Overflow,
    /// Overlap with the existing regions
    Overlap,
    /// Invalid region
    InvalidRegion,
}

/// A virtual memory map ledger.
//
// Developer Note: the first record is reserved for the ledger bounds. We
// structure it this way so that the user of the `Ledger` type has
// fine-grained controls over allocation. For example, to allocate a 4k page,
// the user can instantiate as `Ledger::<128>::new(..)`.
#[derive(Clone, Debug)]
pub struct Ledger<const N: usize> {
    records: [Record; N],
}

impl<const N: usize> Ledger<N> {
    /// Create a new instance.
    pub const fn new(region: Line<Address<usize, Page>>) -> Self {
        let mut records = [Record::EMPTY; N];
        records[0].region = region;
        Self { records }
    }

    /// Insert a new record into the ledger.
    pub fn insert(
        &mut self,
        region: impl Into<Line<Address<usize, Page>>>,
        permissions: impl Into<Option<Permissions>>,
    ) -> Result<(), Error> {
        // Make sure the record is valid.
        let record = Record::new(region.into(), permissions.into().unwrap_or_default());
        if record.region.start >= record.region.end {
            return Err(Error::InvalidRegion);
        }

        // Make sure the record fits in our adress space.
        let region = self.records[0].region;
        if record.region.start < region.start || record.region.end > region.end {
            return Err(Error::Overflow);
        }

        // Loop over the records looking for merges.
        let mut iter = self.records_mut().iter_mut().peekable();
        while let Some(prev) = iter.next() {
            if prev.region.intersection(record.region).is_some() {
                return Err(Error::Overlap);
            }

            if let Some(next) = iter.peek() {
                if next.region.intersection(record.region).is_some() {
                    return Err(Error::Overlap);
                }
            }

            // Potentially merge with the `prev` slot.
            if prev.permissions == record.permissions && prev.region.end == record.region.start {
                prev.region.end = record.region.end;

                return Ok(());
            }

            // Potentially merge with the `prev` slot
            if let Some(next) = iter.peek_mut() {
                if next.permissions == record.permissions && next.region.start == record.region.end
                {
                    next.region.start = record.region.start;

                    return Ok(());
                }
            }
        }

        // If there is room to append a new record.
        if self.records[0].length + 2 <= self.records.len() {
            self.records[0].length += 1;
            self.records[self.records[0].length] = record;
            self.sort();
            return Ok(());
        }

        Err(Error::NoCapacity)
    }

    /// Get an immutable view of the records.
    pub fn records(&self) -> &[Record] {
        let used = self.records[0].length;
        &self.records[1..][..used]
    }

    /// Sort the records.
    fn sort(&mut self) {
        self.records_mut().sort_unstable_by(|l, r| {
            if l.region == r.region {
                Ordering::Equal
            } else if l.region.is_empty() {
                Ordering::Greater
            } else if r.region.is_empty() {
                Ordering::Less
            } else {
                l.region.start.cmp(&r.region.start)
            }
        })
    }

    /// Get a mutable view of the records.
    fn records_mut(&mut self) -> &mut [Record] {
        let used = self.records[0].length;
        &mut self.records[1..][..used]
    }

    /// Find space for a free region.
    pub fn find_free(
        &self,
        len: Offset<usize, Page>,
        front: bool,
    ) -> Result<Line<Address<usize, Page>>, Error> {
        let region = self.records[0].region;

        let start = Record {
            region: Line::new(region.start, region.start),
            ..Default::default()
        };

        let end = Record {
            region: Line::new(region.end, region.end),
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
                if r.region.end - l.region.start > len {
                    return Ok(Span::new(l.region.end, len).into());
                }
            }
        } else {
            let mut iter = iter.rev();
            while let Some([l, r]) = iter.next() {
                if r.region.end - l.region.start > len {
                    return Ok(Span::new(r.region.start - len, len).into());
                }
            }
        }

        Err(Error::NoSpace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::mem::{align_of, size_of};

    const PREV: Record = Record {
        region: Line {
            start: Address::new(0x4000usize),
            end: Address::new(0x5000usize),
        },
        permissions: Permissions::empty(),
        length: 0,
    };

    const NEXT: Record = Record {
        region: Line {
            start: Address::new(0x8000),
            end: Address::new(0x9000),
        },
        permissions: Permissions::empty(),
        length: 0,
    };

    const INDX: Record = Record {
        region: Line {
            start: Address::new(0x1000),
            end: Address::new(0x10000),
        },
        permissions: Permissions::empty(),
        length: 2,
    };

    const LEDGER: Ledger<3> = Ledger {
        records: [INDX, PREV, NEXT],
    };

    #[test]
    fn record_size_align() {
        assert_eq!(size_of::<Record>(), size_of::<usize>() * 4);
        assert_eq!(align_of::<Record>(), align_of::<usize>());
    }

    #[test]
    fn insert() {
        let start = Address::from(0x1000usize).lower();
        let end = Address::from(0x10000usize).lower();
        let mut ledger = Ledger::<8>::new(Line::new(start, end));

        let region = Line {
            start: Address::from(0xe000usize).lower(),
            end: Address::from(0x10000usize).lower(),
        };

        assert_eq!(ledger.records(), &[]);
        ledger.insert(region, None,).unwrap();
        assert_eq!(
            ledger.records(),
            &[Record::new(region, Permissions::empty())]
        );
    }

    #[test]
    fn find_free_front() {
        let start = Address::from(0x1000).lower();
        let end = Address::from(0x10000).lower();
        let mut ledger = Ledger::<8>::new(Line::new(start, end));

        let region = ledger.find_free(Offset::from_items(2), true).unwrap();
        let answer = Line {
            start: Address::from(0x1000).lower(),
            end: Address::from(0x3000).lower(),
        };
        assert_eq!(region, answer);

        ledger.insert(answer, None,).unwrap();

        let region = ledger.find_free(Offset::from_items(2), true).unwrap();
        let answer = Line {
            start: Address::from(0x3000).lower(),
            end: Address::from(0x5000).lower(),
        };
        assert_eq!(region, answer);
    }

    #[test]
    fn find_free_back() {
        let start = Address::from(0x1000).lower();
        let end = Address::from(0x10000).lower();
        let mut ledger = Ledger::<8>::new(Line::new(start, end));

        let region = ledger.find_free(Offset::from_items(2), false).unwrap();
        let answer = Line {
            start: Address::from(0xe000).lower(),
            end: Address::from(0x10000).lower(),
        };
        assert_eq!(region, answer);

        ledger.insert(answer, None,).unwrap();

        let region = ledger.find_free(Offset::from_items(2), false).unwrap();
        let answer = Line {
            start: Address::from(0xc000).lower(),
            end: Address::from(0xe000).lower(),
        };
        assert_eq!(region, answer);
    }

    #[test]
    fn merge_after() {
        const REGION: Line<Address<usize, Page>> = Line {
            start: Address::new(0x5000),
            end: Address::new(0x6000),
        };

        const MERGED: Record = Record {
            region: Line {
                start: Address::new(0x4000),
                end: Address::new(0x6000),
            },
            permissions: Permissions::empty(),
            length: 0,
        };

        let mut ledger = LEDGER.clone();
        ledger
            .insert(REGION, Permissions::empty(),)
            .unwrap();

        assert_eq!(ledger.records[0].length, 2);
        assert_eq!(ledger.records[1], MERGED);
        assert_eq!(ledger.records[2], NEXT);
    }

    #[test]
    fn merge_before() {
        const REGION: Line<Address<usize, Page>> = Line {
            start: Address::new(0x7000),
            end: Address::new(0x8000),
        };

        const MERGED: Record = Record {
            region: Line {
                start: Address::new(0x7000),
                end: Address::new(0x9000),
            },
            permissions: Permissions::empty(),
            length: 0,
        };

        let mut ledger = LEDGER.clone();
        ledger
            .insert(REGION, Permissions::empty(),)
            .unwrap();

        assert_eq!(ledger.records[0].length, 2);
        assert_eq!(ledger.records[1], PREV);
        assert_eq!(ledger.records[2], MERGED);
    }
}
