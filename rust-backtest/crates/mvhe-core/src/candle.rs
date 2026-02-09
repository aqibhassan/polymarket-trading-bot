use std::path::Path;

/// Struct-of-Arrays candle storage for cache-efficient, SIMD-friendly access.
///
/// All vectors are parallel — index `i` across all fields represents one candle.
#[derive(Debug, Clone)]
pub struct CandleStore {
    pub timestamps: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

impl CandleStore {
    pub fn new() -> Self {
        Self {
            timestamps: Vec::new(),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(cap),
            open: Vec::with_capacity(cap),
            high: Vec::with_capacity(cap),
            low: Vec::with_capacity(cap),
            close: Vec::with_capacity(cap),
            volume: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    pub fn push(&mut self, ts: i64, o: f64, h: f64, l: f64, c: f64, v: f64) {
        self.timestamps.push(ts);
        self.open.push(o);
        self.high.push(h);
        self.low.push(l);
        self.close.push(c);
        self.volume.push(v);
    }

    /// Load candles from a CSV file using memory-mapped I/O.
    ///
    /// Expected CSV format: timestamp,open,high,low,close,volume
    /// Handles ISO8601 timestamps: `2025-11-08T00:00:00Z` or `2024-02-08T00:00:00+00:00`
    pub fn from_csv(path: &Path) -> Result<Self, CsvError> {
        let file = std::fs::File::open(path).map_err(|e| CsvError::Io(e.to_string()))?;
        let mmap =
            unsafe { memmap2::Mmap::map(&file) }.map_err(|e| CsvError::Io(e.to_string()))?;
        let data = &mmap[..];

        Self::parse_csv_bytes(data)
    }

    /// Parse CSV from raw bytes (testable without files).
    pub fn parse_csv_bytes(data: &[u8]) -> Result<Self, CsvError> {
        // Estimate row count for pre-allocation (avg ~50 bytes per row)
        let estimated_rows = data.len() / 50;
        let mut store = Self::with_capacity(estimated_rows);

        let len = data.len();

        // Skip header row
        let mut pos = if let Some(nl) = memchr::memchr(b'\n', data) {
            nl + 1
        } else {
            return Ok(store);
        };

        while pos < len {
            // Find end of line
            let line_end = memchr::memchr(b'\n', &data[pos..])
                .map(|i| pos + i)
                .unwrap_or(len);

            let line = &data[pos..line_end];
            // Skip empty lines or trailing whitespace
            let line = if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            };

            if !line.is_empty() {
                Self::parse_row(line, &mut store)?;
            }

            pos = line_end + 1;
        }

        // Sort by timestamp
        if !store.is_empty() {
            let mut indices: Vec<usize> = (0..store.len()).collect();
            indices.sort_unstable_by_key(|&i| store.timestamps[i]);

            let sorted = Self::reorder(&store, &indices);
            return Ok(sorted);
        }

        Ok(store)
    }

    fn parse_row(line: &[u8], store: &mut CandleStore) -> Result<(), CsvError> {
        let mut fields = [0usize; 7]; // start positions of up to 6 fields + end
        let mut field_count = 0;
        fields[0] = 0;

        for (i, &b) in line.iter().enumerate() {
            if b == b',' {
                field_count += 1;
                if field_count >= 6 {
                    break;
                }
                fields[field_count] = i + 1;
            }
        }

        if field_count < 5 {
            return Err(CsvError::Parse(format!(
                "Expected 6 columns, got {}",
                field_count + 1
            )));
        }

        // Field boundaries
        let ts_bytes = &line[fields[0]..Self::field_end(line, fields[0], fields[1] - 1)];
        let o_start = fields[1];
        let o_end = Self::find_comma_or_end(line, o_start);
        let h_start = o_end + 1;
        let h_end = Self::find_comma_or_end(line, h_start);
        let l_start = h_end + 1;
        let l_end = Self::find_comma_or_end(line, l_start);
        let c_start = l_end + 1;
        let c_end = Self::find_comma_or_end(line, c_start);
        let v_start = c_end + 1;
        let v_end = line.len();

        let ts = Self::parse_timestamp(ts_bytes)?;
        let o: f64 = fast_float::parse(&line[o_start..o_end])
            .map_err(|_| CsvError::Parse("bad open".into()))?;
        let h: f64 = fast_float::parse(&line[h_start..h_end])
            .map_err(|_| CsvError::Parse("bad high".into()))?;
        let l: f64 = fast_float::parse(&line[l_start..l_end])
            .map_err(|_| CsvError::Parse("bad low".into()))?;
        let c: f64 = fast_float::parse(&line[c_start..c_end])
            .map_err(|_| CsvError::Parse("bad close".into()))?;
        let v: f64 = fast_float::parse(&line[v_start..v_end])
            .map_err(|_| CsvError::Parse("bad volume".into()))?;

        store.push(ts, o, h, l, c, v);
        Ok(())
    }

    #[inline]
    fn field_end(_line: &[u8], _start: usize, comma_pos: usize) -> usize {
        comma_pos
    }

    #[inline]
    fn find_comma_or_end(line: &[u8], start: usize) -> usize {
        memchr::memchr(b',', &line[start..])
            .map(|i| start + i)
            .unwrap_or(line.len())
    }

    /// Parse ISO8601 timestamp to Unix epoch seconds.
    /// Handles: `2025-11-08T00:00:00Z` and `2024-02-08T00:00:00+00:00`
    /// Also handles plain unix timestamps (integer).
    fn parse_timestamp(bytes: &[u8]) -> Result<i64, CsvError> {
        // Try plain integer first
        if let Ok(ts) = fast_float::parse::<f64, _>(bytes) {
            if ts > 1_000_000_000.0 && !bytes.contains(&b'T') && !bytes.contains(&b'-') {
                return Ok(ts as i64);
            }
        }

        // ISO8601 parsing: YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DDTHH:MM:SS+00:00
        if bytes.len() < 19 {
            return Err(CsvError::Parse(format!(
                "timestamp too short: {}",
                String::from_utf8_lossy(bytes)
            )));
        }

        let s = std::str::from_utf8(bytes)
            .map_err(|_| CsvError::Parse("non-UTF8 timestamp".into()))?;

        // Parse components manually for speed
        let year: i32 = s[0..4]
            .parse()
            .map_err(|_| CsvError::Parse("bad year".into()))?;
        let month: u32 = s[5..7]
            .parse()
            .map_err(|_| CsvError::Parse("bad month".into()))?;
        let day: u32 = s[8..10]
            .parse()
            .map_err(|_| CsvError::Parse("bad day".into()))?;
        let hour: u32 = s[11..13]
            .parse()
            .map_err(|_| CsvError::Parse("bad hour".into()))?;
        let minute: u32 = s[14..16]
            .parse()
            .map_err(|_| CsvError::Parse("bad minute".into()))?;
        let second: u32 = s[17..19]
            .parse()
            .map_err(|_| CsvError::Parse("bad second".into()))?;

        // Convert to Unix timestamp
        let days = days_from_civil(year, month, day);
        let ts = days as i64 * 86400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64;

        Ok(ts)
    }

    fn reorder(store: &CandleStore, indices: &[usize]) -> CandleStore {
        let mut result = CandleStore::with_capacity(indices.len());
        for &i in indices {
            result.push(
                store.timestamps[i],
                store.open[i],
                store.high[i],
                store.low[i],
                store.close[i],
                store.volume[i],
            );
        }
        result
    }

    /// Get a sub-slice view as a new CandleStore (copies data).
    pub fn slice(&self, start: usize, end: usize) -> CandleStore {
        let end = end.min(self.len());
        let start = start.min(end);
        CandleStore {
            timestamps: self.timestamps[start..end].to_vec(),
            open: self.open[start..end].to_vec(),
            high: self.high[start..end].to_vec(),
            low: self.low[start..end].to_vec(),
            close: self.close[start..end].to_vec(),
            volume: self.volume[start..end].to_vec(),
        }
    }
}

impl Default for CandleStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert civil date to days since Unix epoch (Howard Hinnant algorithm).
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let m = if month <= 2 {
        month as i64 + 9
    } else {
        month as i64 - 3
    };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * m as u64 + 2) / 5 + day as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe as i64 - 719468) as i64
}

#[derive(Debug)]
pub enum CsvError {
    Io(String),
    Parse(String),
}

impl std::fmt::Display for CsvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CsvError::Io(e) => write!(f, "I/O error: {}", e),
            CsvError::Parse(e) => write!(f, "Parse error: {}", e),
        }
    }
}

impl std::error::Error for CsvError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_basic() {
        let csv = b"timestamp,open,high,low,close,volume\n\
                     2025-01-01T00:00:00Z,100.0,105.0,99.0,103.0,1000.0\n\
                     2025-01-01T00:01:00Z,103.0,106.0,102.0,105.0,1200.0\n";

        let store = CandleStore::parse_csv_bytes(csv).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.open[0], 100.0);
        assert_eq!(store.close[1], 105.0);
        assert_eq!(store.volume[0], 1000.0);
    }

    #[test]
    fn test_parse_timestamp_z() {
        let ts = CandleStore::parse_timestamp(b"2025-01-01T00:00:00Z").unwrap();
        // 2025-01-01 00:00:00 UTC
        assert_eq!(ts, 1735689600);
    }

    #[test]
    fn test_parse_timestamp_offset() {
        let ts = CandleStore::parse_timestamp(b"2025-01-01T00:00:00+00:00").unwrap();
        assert_eq!(ts, 1735689600);
    }

    #[test]
    fn test_days_from_civil() {
        // 1970-01-01 should be day 0
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        // 2025-01-01
        assert_eq!(days_from_civil(2025, 1, 1), 20089);
    }

    #[test]
    fn test_slice() {
        let csv = b"timestamp,open,high,low,close,volume\n\
                     2025-01-01T00:00:00Z,100.0,105.0,99.0,103.0,1000.0\n\
                     2025-01-01T00:01:00Z,103.0,106.0,102.0,105.0,1200.0\n\
                     2025-01-01T00:02:00Z,105.0,108.0,104.0,107.0,1100.0\n";

        let store = CandleStore::parse_csv_bytes(csv).unwrap();
        let sub = store.slice(1, 3);
        assert_eq!(sub.len(), 2);
        assert_eq!(sub.open[0], 103.0);
    }
}
