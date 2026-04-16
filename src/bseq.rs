use flate2::read::GzDecoder;
use std::io::{self, BufRead, Read};

/// A single sequence record from FASTA/FASTQ.
#[derive(Clone, Debug)]
pub struct BseqRecord {
    pub name: String,
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
    pub comment: String,
    pub l_seq: usize,
}

/// FASTA/FASTQ file reader supporting plain text and gzip.
pub struct BseqFile {
    reader: Box<dyn BufRead>,
    buf: String,
    is_fastq: bool,
    /// Stashed header line for FASTA multi-line reading.
    stashed_header: Option<String>,
    eof: bool,
}

impl BseqFile {
    /// Open a FASTA/FASTQ file (gzip auto-detected via magic bytes).
    pub fn open(path: &str) -> io::Result<Self> {
        let file: Box<dyn Read> = if path == "-" {
            Box::new(io::stdin())
        } else {
            let f = std::fs::File::open(path)?;
            let mut peek = [0u8; 2];
            let mut f = io::BufReader::new(f);
            let n = f.read(&mut peek)?;
            if n >= 2 && peek[0] == 0x1f && peek[1] == 0x8b {
                let chain = io::Cursor::new(peek[..n].to_vec()).chain(f);
                Box::new(GzDecoder::new(chain))
            } else {
                let chain = io::Cursor::new(peek[..n].to_vec()).chain(f);
                Box::new(chain)
            }
        };
        let mut reader = io::BufReader::new(file);

        // Read first line to detect format
        let mut first_line = String::new();
        let n = reader.read_line(&mut first_line)?;
        if n == 0 {
            return Ok(Self {
                reader: Box::new(reader),
                buf: String::new(),
                is_fastq: false,
                stashed_header: None,
                eof: true,
            });
        }
        let is_fastq = first_line.starts_with('@');
        let stashed_header = Some(first_line.trim_end().to_string());

        Ok(Self {
            reader: Box::new(reader),
            buf: String::new(),
            is_fastq,
            stashed_header,
            eof: false,
        })
    }

    /// Read a line from the reader into self.buf.
    fn read_line(&mut self) -> io::Result<usize> {
        self.buf.clear();
        self.reader.read_line(&mut self.buf)
    }

    /// Read one FASTA/FASTQ record. Returns None at EOF.
    pub fn read_record(&mut self) -> io::Result<Option<BseqRecord>> {
        if self.eof {
            return Ok(None);
        }

        // Get header line (from stash or read new)
        let header = match self.stashed_header.take() {
            Some(h) => h,
            None => {
                let n = self.read_line()?;
                if n == 0 {
                    self.eof = true;
                    return Ok(None);
                }
                self.buf.trim_end().to_string()
            }
        };

        if header.is_empty() {
            self.eof = true;
            return Ok(None);
        }

        // Parse header: strip leading > or @, split name from comment
        let header_content = if header.starts_with('>') || header.starts_with('@') {
            &header[1..]
        } else {
            &header
        };
        let (name, comment) = parse_header(header_content);

        if self.is_fastq {
            // FASTQ: single line seq, +, qual
            let n = self.read_line()?;
            if n == 0 {
                self.eof = true;
                return Ok(None);
            }
            let mut seq: Vec<u8> = self.buf.trim_end().as_bytes().to_vec();
            u_to_t(&mut seq);

            // Read + line
            self.read_line()?;

            // Read quality line
            let n = self.read_line()?;
            let qual = if n > 0 {
                self.buf.trim_end().as_bytes().to_vec()
            } else {
                Vec::new()
            };

            let l_seq = seq.len();
            Ok(Some(BseqRecord {
                name,
                seq,
                qual,
                comment,
                l_seq,
            }))
        } else {
            // FASTA: multi-line sequence until next > or EOF
            let mut seq = Vec::new();
            loop {
                let n = self.read_line()?;
                if n == 0 {
                    self.eof = true;
                    break;
                }
                let line = self.buf.trim_end().to_string();
                if line.starts_with('>') {
                    // Next record header — stash it
                    self.stashed_header = Some(line);
                    break;
                }
                let mut line_bytes = line.into_bytes();
                u_to_t(&mut line_bytes);
                seq.extend_from_slice(&line_bytes);
            }
            let l_seq = seq.len();
            Ok(Some(BseqRecord {
                name,
                seq,
                qual: Vec::new(),
                comment,
                l_seq,
            }))
        }
    }

    /// Read a batch of sequences up to chunk_size total bases.
    pub fn read_batch(&mut self, chunk_size: i64, with_qual: bool) -> io::Result<Vec<BseqRecord>> {
        let mut records = Vec::new();
        let mut total_len: i64 = 0;
        loop {
            match self.read_record()? {
                None => break,
                Some(mut rec) => {
                    if !with_qual {
                        rec.qual.clear();
                    }
                    total_len += rec.l_seq as i64;
                    records.push(rec);
                    if total_len >= chunk_size {
                        break;
                    }
                }
            }
        }
        Ok(records)
    }

    pub fn is_eof(&self) -> bool {
        self.eof
    }
}

/// Convert U/u to T/t in sequence.
fn u_to_t(seq: &mut [u8]) {
    for b in seq.iter_mut() {
        if *b == b'u' || *b == b'U' {
            *b -= 1;
        }
    }
}

fn parse_header(s: &str) -> (String, String) {
    match s.split_once(|c: char| c.is_whitespace()) {
        Some((name, comment)) => (name.to_string(), comment.to_string()),
        None => (s.to_string(), String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(content: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_read_fasta() {
        let content = b">seq1 comment1\nACGTACGT\nTGCA\n>seq2\nAAAA\n";
        let f = write_temp_file(content);
        let mut reader = BseqFile::open(f.path().to_str().unwrap()).unwrap();

        let rec1 = reader.read_record().unwrap().unwrap();
        assert_eq!(rec1.name, "seq1");
        assert_eq!(rec1.comment, "comment1");
        assert_eq!(rec1.seq, b"ACGTACGTTGCA");
        assert_eq!(rec1.l_seq, 12);

        let rec2 = reader.read_record().unwrap().unwrap();
        assert_eq!(rec2.name, "seq2");
        assert_eq!(rec2.seq, b"AAAA");

        assert!(reader.read_record().unwrap().is_none());
    }

    #[test]
    fn test_read_fastq() {
        let content = b"@read1 comment\nACGT\n+\nIIII\n@read2\nTGCA\n+\nHHHH\n";
        let f = write_temp_file(content);
        let mut reader = BseqFile::open(f.path().to_str().unwrap()).unwrap();

        let rec1 = reader.read_record().unwrap().unwrap();
        assert_eq!(rec1.name, "read1");
        assert_eq!(rec1.seq, b"ACGT");
        assert_eq!(rec1.qual, b"IIII");

        let rec2 = reader.read_record().unwrap().unwrap();
        assert_eq!(rec2.name, "read2");
        assert_eq!(rec2.seq, b"TGCA");
        assert_eq!(rec2.qual, b"HHHH");

        assert!(reader.read_record().unwrap().is_none());
    }

    #[test]
    fn test_u_to_t_conversion() {
        let content = b">seq1\nACGUacgu\n";
        let f = write_temp_file(content);
        let mut reader = BseqFile::open(f.path().to_str().unwrap()).unwrap();
        let rec = reader.read_record().unwrap().unwrap();
        assert_eq!(rec.seq, b"ACGTacgt");
    }

    #[test]
    fn test_read_batch() {
        let content = b">s1\nACGT\n>s2\nTGCA\n>s3\nAAAA\n";
        let f = write_temp_file(content);
        let mut reader = BseqFile::open(f.path().to_str().unwrap()).unwrap();

        let batch = reader.read_batch(5, false).unwrap();
        assert_eq!(batch.len(), 2); // 4 + 4 >= 5, stops after 2
        assert_eq!(batch[0].name, "s1");
        assert_eq!(batch[1].name, "s2");
    }
}
